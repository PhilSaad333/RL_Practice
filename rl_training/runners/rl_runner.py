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
        per_rank = self.buffer_size // world
        # Handle remainder: rank 0 gets extra samples if buffer_size not evenly divisible
        if self.rank == 0:
            per_rank += self.buffer_size % world
        
        if self.rank == 0:
            print(f"Distributed collection: {world} ranks, buffer_size={self.buffer_size}, per_rank={per_rank}")

        for _ in trange(outer_loops, desc="outer collect loops", disable=(self.rank != 0)):
            # Memory monitoring before collection
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory before collection: {mem_before:.2f}GB")
            
            rb = self.collector.collect_batch(batch_prompts=per_rank)
            
            # Memory monitoring after collection
            if torch.cuda.is_available():
                mem_after_collect = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory after collection: {mem_after_collect:.2f}GB")
            
            # Add barrier after collection to ensure both ranks finish before training
            if self.ddp:
                print(f"[DEBUG] Rank {self.rank} entering post-collection barrier")
                dist.barrier()
                print(f"[DEBUG] Rank {self.rank} exited post-collection barrier")
            print(f"[DEBUG] Rank {self.rank} completed rollout collection")
            print(f"[DEBUG] Rank {self.rank} about to start training on buffer")
            
            # each rank trains on its shard; DDP averages grads for you
            self._train_one_buffer(rb, K, ga_steps, B)
            
            # Aggressive memory cleanup after training
            print(f"[DEBUG] Rank {self.rank} starting aggressive memory cleanup")
            del rb
            torch.cuda.empty_cache()
            gc.collect()
            
            # Memory monitoring after cleanup
            if torch.cuda.is_available():
                mem_after_cleanup = torch.cuda.memory_allocated() / 1024**3
                print(f"[MEMORY] Rank {self.rank} GPU memory after cleanup: {mem_after_cleanup:.2f}GB")




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
        print(f"[DEBUG] Rank {self.rank} entered _train_one_buffer with {len(rb)} prompts, K={K}, ga_steps={ga_steps}, B={B}")
        stats_sum, total_mb_cnt = defaultdict(float), 0
        for epoch in range(K):
            print(f"[DEBUG] Rank {self.rank} starting PPO epoch {epoch+1}/{K}")
            micro_cnt = 0
            for idx in rb.iter_minibatches(B, shuffle=True):
                print(f"[DEBUG] Rank {self.rank} processing microbatch {micro_cnt+1}, idx={len(idx)} samples")
                sync = ((micro_cnt + 1) % ga_steps == 0)
                print(f"[DEBUG] Rank {self.rank} microbatch {micro_cnt+1}: sync_grads={sync} (step_id will be {self.step_id + (1 if sync else 0)})")
                mb = rb.get_batch(idx, device=f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
                print(f"[DEBUG] Rank {self.rank} calling algo.step for microbatch {micro_cnt+1}")
                stats = self.algo.step(mb, self.ref_model, sync_grads=sync)
                print(f"[DEBUG] Rank {self.rank} completed algo.step for microbatch {micro_cnt+1}")
                for k, v in stats.items():
                    stats_sum[k] += v
                micro_cnt += 1
                total_mb_cnt += 1
                if sync:
                    self.step_id += 1
                    print(f"[DEBUG] Rank {self.rank} incremented step_id to {self.step_id}")
                del mb, stats
                torch.cuda.empty_cache()
            print(f"[DEBUG] Rank {self.rank} completed PPO epoch {epoch+1}/{K}")
        stats_avg = {k: v / total_mb_cnt for k, v in stats_sum.items()}
        print(f"[DEBUG] Rank {self.rank} computed stats_avg, about to log")
        print(f"stats: {stats_avg}")
        print(f"[DEBUG] Rank {self.rank} calling _log with stats")
        self._log(stats_avg)
        print(f"[DEBUG] Rank {self.rank} completed _log, moving to probe section")

        # run the GNS probe every N optimiser steps using the current buffer
        every = int(self.gns_cfg.get("every", 0))
        print(f"[DEBUG] Rank {self.rank} checking GNS probe: every={every}, step_id={self.step_id}, modulo={self.step_id % every if every > 0 else 'N/A'}")
        if every > 0 and (self.step_id % every == 0) and self.rank == 0:
            # Only rank 0 runs GNS probe - no barriers needed since it's rank-specific
            try:
                print(f"[DEBUG] Rank {self.rank} STARTING GNS probe (step_id={self.step_id})")
                self._probe_gns(rb)
                print(f"[DEBUG] Rank {self.rank} COMPLETED GNS probe (step_id={self.step_id})")
            except Exception as e:
                print(f"[GNS] probe failed: {e}")
        elif every > 0 and (self.step_id % every == 0):
            # Other ranks just note that GNS probe step is happening
            print(f"[DEBUG] Rank {self.rank} skipping GNS probe (rank 0 only, step_id={self.step_id})")
        else:
            print(f"[DEBUG] Rank {self.rank} no GNS probe this step (step_id={self.step_id})")




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
        # Evaluation disabled for distributed training to avoid barrier issues
        # Can run evaluation later on saved checkpoints
        print(f"[DEBUG] Rank {self.rank} skipping evaluation (disabled for distributed training)")

    def _run_eval(self, ckpt_dir: pathlib.Path):
        print(f"[Eval] starting eval for step {self.step_id} …")
        self.model.to("cpu"); torch.cuda.empty_cache(); gc.collect()

        cmd = [
            "python", "-m", "evals.eval_runner",
            "--backbone", self.cfg["eval_backbone"],
            "--ft-dataset", self.cfg["scheduler"]["dataset_name"],
            "--ckpt-path", str(ckpt_dir),
            "--ckpt-step", str(self.step_id),
            "--batch-size", str(self.cfg.get("eval_batch_size", 8)),
            "--subset-frac", str(self.cfg.get("eval_frac", 1.0)),
            "--eval-dataset", self.cfg["scheduler"]["dataset_name"],
            "--temperature", str(self.cfg.get("eval_temperature", 0.7)),
            "--top-p", "1.0",
            "--runs-root", str(self.dir.parent / "eval_runs")
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        run_sync(cmd, env=env, check=True)

        torch.cuda.empty_cache(); gc.collect()
        self.model.to(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"[Eval] finished, resuming training.")


    def _probe_gns(self, rb):
        """Measure ||g||^2 at two prompt-batch sizes, estimate B_simple, log it."""
        print(f"[GNS DEBUG] Rank {self.rank} entered _probe_gns")
        device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        if self.rank != 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns (not rank 0)")
            return
        if not self.gns_cfg or int(self.gns_cfg.get("every", 0)) <= 0:
            print(f"[GNS DEBUG] Rank {self.rank} exiting _probe_gns (config disabled)")
            return
        print(f"[GNS DEBUG] Rank {self.rank} starting GNS probe computation")

        import random
        B_small = self._gns_state["B_small"]
        B_large = self._gns_state["B_large"]
        ema = self._gns_state["ema"]
        print(f"[GNS DEBUG] B_small={B_small}, B_large={B_large}, ema={ema}")

        # sample prompt indices (without replacement) from the current rollout buffer
        N_prompts = len(rb)
        print(f"[GNS DEBUG] N_prompts={N_prompts}, need max({B_small}, {B_large})={max(B_small, B_large)}")
        if N_prompts < max(B_small, B_large):
            print(f"[GNS DEBUG] Not enough prompts for probe, exiting")
            return  # not enough to probe this time

        # index prompts we'll use for the two effective batches
        print(f"[GNS DEBUG] Sampling prompt indices")
        import random
        idx_small = random.sample(range(N_prompts), B_small)
        idx_large = random.sample(range(N_prompts), B_large)
        print(f"[GNS DEBUG] Sampled {len(idx_small)} small indices, {len(idx_large)} large indices")


        # helper: split a list of prompt indices into microbatches that fit in VRAM
        def _make_microbatches(idxs, micro_size):
            mbs = []
            for s in range(0, len(idxs), micro_size):
                mbs.append(rb.get_batch(idxs[s:s+micro_size], device=device))
            return mbs

        micro_size = int(self.cfg.get("prompts_per_microbatch", 1))
        print(f"[GNS DEBUG] Creating microbatches with micro_size={micro_size}")
        mbs_small = _make_microbatches(idx_small, micro_size)
        mbs_large = _make_microbatches(idx_large, micro_size)
        print(f"[GNS DEBUG] Created {len(mbs_small)} small microbatches, {len(mbs_large)} large microbatches")

        # emulate grad accumulation to measure ||g||^2 for the two effective batches
        print(f"[GNS DEBUG] Computing gradient norm for small batch")
        y_small = self.algo._grad_sq_norm_for_effective_batch(
            mbs_small, self.ref_model, avoid_ddp_allreduce=True
        )
        print(f"[GNS DEBUG] Computed y_small={y_small}")
        print(f"[GNS DEBUG] Computing gradient norm for large batch")
        y_large = self.algo._grad_sq_norm_for_effective_batch(
            mbs_large, self.ref_model, avoid_ddp_allreduce=True
        )
        print(f"[GNS DEBUG] Computed y_large={y_large}")


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

