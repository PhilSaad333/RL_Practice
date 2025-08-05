# rl_training/runners/rl_runner.py
from __future__ import annotations

import os, json, math, gc, datetime, pathlib, shutil, textwrap, yaml, torch
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


class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str):
        # ─── distributed init ────────────────────────────────────────────
        dist.init_process_group("nccl")
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.rank = dist.get_rank()

        # ─── I/O setup ───────────────────────────────────────────────────
        self.cfg = yaml.safe_load(open(cfg_path))
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = RUN_ROOT / f"run_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, self.dir / "config.yaml")

        self.tb = SummaryWriter(str(self.dir)) if self.rank == 0 else None
        self.save_every = self.cfg.get("save_every", 100)
        self.step_id = 0

        # ─── model & tokenizer ───────────────────────────────────────────
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(self.cfg["backbone"],
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb)
        base = prepare_model_for_kbit_training(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        model = PeftModel.from_pretrained(base, lora_ckpt, is_trainable=True)
        model = model.to(self.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank], output_device=self.local_rank)
        model.module.enable_input_require_grads()
        self.model = model

        self.tok = AutoTokenizer.from_pretrained(self.cfg["backbone"])
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if self.model.module.config.pad_token_id is None:
            self.model.module.config.pad_token_id = self.tok.pad_token_id
        self.pad_id = self.tok.pad_token_id
        self.tok.padding_side = "left"

        self.ref_model = torch.clone(self.model.module).eval().requires_grad_(False)
        self.collector = RolloutCollector(self.model, self.tok, self.cfg,
                                          out_dir=self.dir,
                                          device=f"cuda:{self.local_rank}")
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

        for _ in trange(outer_loops, desc="outer collect loops",
                        disable=(self.rank != 0)):
            rb = self.collector.collect_batch(batch_prompts=self.buffer_size)
            torch.cuda.empty_cache(); gc.collect()
            self._train_one_buffer(rb, K, ga_steps, B)
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
                mb = rb.get_batch(idx, device=f"cuda:{self.local_rank}")
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
            self.model.module.save_pretrained(save_dir)
            print(f"saved model to {save_dir}")
        dist.barrier()
        if self.rank == 0:
            self._run_eval(save_dir)
        dist.barrier()

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
        self.model.to(f"cuda:{self.local_rank}")
        print(f"[Eval] finished, resuming training.")

# ─── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint dir")
    args = p.parse_args()
    cfg_dict = yaml.safe_load(open(args.cfg))
    runner = RLRunner(args.cfg, args.ckpt)
    runner.train(total_updates=cfg_dict["total_steps"])
