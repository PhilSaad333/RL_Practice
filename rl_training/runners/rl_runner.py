# rl_training/runners/rl_runner.py
from __future__ import annotations

import os, json, math, gc, datetime, pathlib, shutil, yaml, torch, copy
import torch.distributed as dist
from collections import defaultdict
from subprocess import run as run_sync
from tqdm.auto import trange
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

from rl_training.runners.collect_rollouts import RolloutCollector
from rl_training.algs.dr_grpo import DRGRPO
from rl_training.algs.base import RolloutBatch
from rl_training.runners.eval_callback import EvalCallback

RUN_ROOT = pathlib.Path(os.environ.get("RUN_ROOT", "./rl_runs"))


def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(prompt_ids=rb.prompt_ids[sl],
                        gen_ids=rb.gen_ids[sl],
                        reward=rb.reward[sl],
                        logprobs=rb.logprobs[sl])

def _unwrap(model):
    # Changed 8/11: helper for DDP/unwrapped access
    return model.module if hasattr(model, "module") else model  # Changed 8/11


class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str):
        # ─── distributed init ────────────────────────────────────────────
        # Changed 8/11: make DDP optional (works in single-GPU Colab)
        if os.environ.get("WORLD_SIZE", "1") != "1":
            dist.init_process_group(backend=os.environ.get("DIST_BACKEND", "nccl"))  # Changed 8/11
            self.local_rank = int(os.getenv("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.rank = dist.get_rank()
            self.ddp = True  # Changed 8/11
        else:
            self.local_rank = 0
            self.rank = 0
            self.ddp = False  # Changed 8/11

        # ─── I/O setup ───────────────────────────────────────────────────
        self.cfg = yaml.safe_load(open(cfg_path))
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = RUN_ROOT / f"run_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, self.dir / "config.yaml")

        self.tb = SummaryWriter(str(self.dir)) if self.rank == 0 else None
        self.save_every = self.cfg.get("save_every", 100)
        self.step_id = 0

        # -----GNS----------------------------------------------------------
        self.gns_cfg = self.cfg.get("gns_probe", {})
        self._gns_state = {
            "ema_y_small": None, "ema_y_large": None,
            "B_small": int(self.gns_cfg.get("small_B", 8)),
            "B_large": int(self.gns_cfg.get("large_B", 64)),
            "ema": float(self.gns_cfg.get("ema", 0.9)),
        }


        # ─── model & tokenizer ───────────────────────────────────────────
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(self.cfg["backbone"],
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb)
        base = prepare_model_for_kbit_training(base)  # PEFT/QLoRA prep  # (PEFT docs)  # Changed 8/11
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        model = PeftModel.from_pretrained(base, lora_ckpt, is_trainable=True)
        model.enable_input_require_grads()
        model = model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        if self.ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )  # Changed 8/11
        self.model = model

        self.tok = AutoTokenizer.from_pretrained(self.cfg["backbone"])
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if _unwrap(self.model).config.pad_token_id is None:
            _unwrap(self.model).config.pad_token_id = self.tok.pad_token_id
        self.pad_id = self.tok.pad_token_id
        self.tok.padding_side = "left"

        # Changed 8/11: build ref model robustly for single or multi-GPU
        self.ref_model = copy.deepcopy(_unwrap(self.model)).eval().requires_grad_(False)  # Changed 8/11
        self.ref_model = self.ref_model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")  # Changed 8/11

        self.collector = RolloutCollector(self.model, self.tok, self.cfg,
                                          out_dir=self.dir,
                                          device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.algo = DRGRPO(self.model, self.cfg, pad_id=self.pad_id,
                           ratio_log_path=self.dir / "ratios.jsonl")
        self.accum = self.cfg["grad_accum_steps"]
        self.buffer_size = self.cfg["buffer_size"]
        self.eval_cb = EvalCallback(self.dir, self.cfg)

    # ─── training loop ────────────────────────────────────────────────────
    def train(self, total_updates: int):
        K = self.cfg["ppo_epochs"]
        ga_steps = self.accum
        B = self.cfg["microbatch_size"]
        outer_loops = math.ceil(total_updates / K)

        world = dist.get_world_size() if dist.is_initialized() else 1
        per_rank = math.ceil(self.buffer_size / world)

        for _ in trange(outer_loops, desc="outer collect loops", disable=(self.rank != 0)):
            rb = self.collector.collect_batch(batch_prompts=per_rank)
            # each rank trains on its shard; DDP averages grads for you
            self._train_one_buffer(rb, K, ga_steps, B)




#        for _ in trange(outer_loops, desc="outer collect loops",
#                        disable=(self.rank != 0)):
#            rb = self.collector.collect_batch(batch_prompts=self.buffer_size)
#            torch.cuda.empty_cache(); gc.collect()
#            self._train_one_buffer(rb, K, ga_steps, B)
            del rb
            torch.cuda.empty_cache(); gc.collect()
            if self.step_id % self.save_every == 0:
                self._save_ckpt()
        self._save_ckpt(final=True)

    def _train_one_buffer(self, rb, K, ga_steps, B):
        stats_sum, total_mb_cnt = defaultdict(float), 0
        for _ in range(K):
            micro_cnt = 0
            for idx in rb.iter_minibatches(B, shuffle=True):
                sync = ((micro_cnt + 1) % ga_steps == 0)
                mb = rb.get_batch(idx, device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
                stats = self.algo.step(mb, self.ref_model, sync_grads=sync)
                for k, v in stats.items():
                    stats_sum[k] += v
                micro_cnt += 1
                total_mb_cnt += 1
                if sync:
                    self.step_id += 1
                del mb, stats
                torch.cuda.empty_cache()
        stats_avg = {k: v / total_mb_cnt for k, v in stats_sum.items()}
        print(f"stats: {stats_avg}")
        self._log(stats_avg)

        # run the GNS probe every N optimiser steps using the current buffer
        every = int(self.gns_cfg.get("every", 0))
        if every > 0 and (self.step_id % every == 0):
            try:
                self._probe_gns(rb)
            except Exception as e:
                print(f"[GNS] probe failed: {e}")




    # ─── utils ────────────────────────────────────────────────────────────
    def _log(self, d):
        if self.rank != 0:
            return
        with open(self.dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps({"step": self.step_id, **d}) + "\n")
        if self.tb:
            for k, v in d.items():
                self.tb.add_scalar(k, float(v), self.step_id)
            self.tb.flush()

    def _save_ckpt(self, final: bool = False):
        tag = f"{self.step_id}" if not final else "final"
        save_dir = self.dir / f"step_{tag}"
        if self.rank == 0:
            _unwrap(self.model).save_pretrained(save_dir)  # Changed 8/11
            print(f"saved model to {save_dir}")
        if self.ddp:
            dist.barrier()  # Changed 8/11
        if self.rank == 0 and self.cfg.get("eval_every", 1) > 0 and (final or self.step_id % self.cfg.get("eval_every", 1) == 0):
            self._run_eval(save_dir)
        if self.ddp:
            dist.barrier()  # Changed 8/11

    def _run_eval(self, ckpt_dir: pathlib.Path):
        print(f"[Eval] starting eval for step {self.step_id} …")
        self.model.to("cpu"); torch.cuda.empty_cache(); gc.collect()

        cmd = [
            "python", "-m", "evals.eval_runner",
            "--backbone", self.cfg["eval_backbone"],
            "--ft_dataset", self.cfg["scheduler"]["dataset_name"],
            "--ckpt_path", str(ckpt_dir),
            "--ckpt_step", str(self.step_id),
            "--batch_size", str(self.cfg.get("eval_batch_size", 8)),
            "--subset_frac", str(self.cfg.get("eval_frac", 1.0)),
            "--eval_dataset", self.cfg["scheduler"]["dataset_name"],
            "--temperature", str(self.cfg.get("eval_temperature", 0.7)),
            "--top_p", "1.0",
            "--runs_root", str(self.dir.parent / "eval_runs")
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        run_sync(cmd, env=env, check=True)

        torch.cuda.empty_cache(); gc.collect()
        self.model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"[Eval] finished, resuming training.")


    def _probe_gns(self, rb):
        """Measure ||g||^2 at two prompt-batch sizes, estimate B_simple, log it."""
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        if self.rank != 0:
            return
        if not self.gns_cfg or int(self.gns_cfg.get("every", 0)) <= 0:
            return

        import random
        B_small = self._gns_state["B_small"]
        B_large = self._gns_state["B_large"]
        ema = self._gns_state["ema"]

        # sample prompt indices (without replacement) from the current rollout buffer
        N_prompts = len(rb)
        if N_prompts < max(B_small, B_large):
            return  # not enough to probe this time

        # index prompts we’ll use for the two effective batches
        import random
        idx_small = random.sample(range(N_prompts), B_small)
        idx_large = random.sample(range(N_prompts), B_large)


        # helper: split a list of prompt indices into microbatches that fit in VRAM
        def _make_microbatches(idxs, micro_size):
            mbs = []
            for s in range(0, len(idxs), micro_size):
                mbs.append(rb.get_batch(idxs[s:s+micro_size], device=device))
            return mbs

        micro_size = int(self.cfg.get("prompts_per_microbatch", 1))
        mbs_small = _make_microbatches(idx_small, micro_size)
        mbs_large = _make_microbatches(idx_large, micro_size)

        # emulate grad accumulation to measure ||g||^2 for the two effective batches
        y_small = self.algo._grad_sq_norm_for_effective_batch(
            mbs_small, self.ref_model, avoid_ddp_allreduce=True
        )
        y_large = self.algo._grad_sq_norm_for_effective_batch(
            mbs_large, self.ref_model, avoid_ddp_allreduce=True
        )


        # EWMA for stability (Appendix A.1 suggests smoothing) 
        #   E[||g_B||^2] ≈ a + c/B  =>  solve from two points (B1,y1),(B2,y2)
        es = self._gns_state["ema_y_small"]
        el = self._gns_state["ema_y_large"]
        es = (ema * es + (1 - ema) * y_small) if es is not None else y_small
        el = (ema * el + (1 - ema) * y_large) if el is not None else y_large
        self._gns_state["ema_y_small"], self._gns_state["ema_y_large"] = es, el

        B1, y1 = float(B_small), float(es)
        B2, y2 = float(B_large), float(el)
        if abs(B1 - B2) < 1e-9:
            return

        # Solve:
        #   a = (B1*y1 - B2*y2) / (B1 - B2)
        #   c = B1*B2*(y2 - y1) / (B1 - B2)
        a_hat = (B1 * y1 - B2 * y2) / (B1 - B2)
        c_hat = (B1 * B2) * (y2 - y1) / (B1 - B2)
        B_simple = float("nan")
        if a_hat > 0 and c_hat > 0:
            B_simple = c_hat / a_hat

        rec = {
            "gns_y_small": y_small, "gns_y_large": y_large,
            "gns_y_small_ema": es, "gns_y_large_ema": el,
            "gns_B_small": B_small, "gns_B_large": B_large,
            "gns_a_hat": a_hat, "gns_c_hat": c_hat,
            "gns_B_simple": B_simple
        }
        # file + TB
        self._log(rec)







# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint dir")
    # tolerate torchrun/launch variants; ignored by our code
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)
    # parse *known* only, to ignore any other launcher args
    args, _ = p.parse_known_args()

    cfg_dict = yaml.safe_load(open(args.cfg))
    runner = RLRunner(args.cfg, args.ckpt)
    runner.train(total_updates=cfg_dict["total_steps"])

