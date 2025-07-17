# rl_training/runners/rl_runner.py
from __future__ import annotations
import json, math, pathlib, datetime, yaml, torch, shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange
from rl_training.runners.collect_rollouts import RolloutCollector
from rl_training.algs.grpo import GRPO
from rl_training.algs.base import RolloutBatch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel

RUN_ROOT = pathlib.Path("/content/drive/MyDrive/RL_Practice_Files/rl_runs")

def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(prompt_ids = rb.prompt_ids[sl],
                        gen_ids    = rb.gen_ids[sl],
                        reward     = rb.reward[sl],
                        logprobs   = rb.logprobs[sl])

class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str, save_every: int = 100):
        # ---------- I/O ---------------------------------------------------
        cfg = yaml.safe_load(open(cfg_path))
        stamp     = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir  = RUN_ROOT / f"run_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, self.dir / "config.yaml")       # save config

        self.tb   = SummaryWriter(log_dir=str(self.dir))
        self.save_every = save_every
        self.step_id    = 0

        # ---------- load model + adapters -------------------------------
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                                    torch_dtype=torch.bfloat16,
                                                    quantization_config=bnb)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        base.config.use_cache = False

        self.model = PeftModel.from_pretrained(base, lora_ckpt).to("cuda")
        self.tok   = AutoTokenizer.from_pretrained("microsoft/phi-2")
        
        if self.tok.pad_token_id is None:
            # safest practice is to duplicate eos so you don't expand embeddings
            self.tok.pad_token = self.tok.eos_token 

        # keep model & GRPO in sync
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tok.pad_token_id

        self.pad_id = self.tok.pad_token_id 
        self.tok.padding_side = "left"

        # ---------- subsystems ------------------------------------------
        self.collector = RolloutCollector(self.model, self.tok, cfg,
                                          out_dir=self.dir, device="cuda")
        self.algo      = GRPO(self.model, cfg, pad_id=self.tok.pad_token_id)
        self.accum = cfg["grad_accum_steps"]

        # for debug
        assert isinstance(self.tok.pad_token_id, int), "pad_token_id must not be None"


    # ---------------- main training loop ------------------------------
    def train(self, total_steps: int = 1000):
        for _ in trange(total_steps, desc="RL steps"):
            self._one_step()
            if self.step_id % self.save_every == 0:
                self._save_ckpt()

        self._save_ckpt(final=True)  # final full save

    def _one_step(self):
        rb = self.collector.collect_batch()
        mb = math.ceil(rb.prompt_ids.size(0) / self.accum)

        stats_acc = {}
        for i in range(0, rb.prompt_ids.size(0), mb):
            sync = (i + mb) >= rb.prompt_ids.size(0)
            stats = self.algo.step(slice_batch(rb, slice(i, i+mb)), sync_grads=sync)
            for k, v in stats.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v / self.accum

        self.step_id += 1
        self._log(stats_acc)

    def _log(self, d):
        json_out = {"step": self.step_id, **d}
        with open(self.dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(json_out) + "\n")
        for k, v in d.items():
            self.tb.add_scalar(k, v, self.step_id)
        self.tb.flush()

    def _save_ckpt(self, final: bool = False):
        tag = f"step-{self.step_id}" if not final else "final"
        save_dir = self.dir / f"ckpt_{tag}"
        self.model.save_pretrained(save_dir)        # adapter-only ✔️ :contentReference[oaicite:8]{index=8}
        # optional merged full model
        merged = self.model.merge_and_unload()      # combines LoRA → dense  :contentReference[oaicite:9]{index=9}
        merged.save_pretrained(save_dir / "merged")


# ------------------------------- CLI -------------------------------------- #
if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",  required=True, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, help="Path to LoRA adapter checkpoint dir")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    runner = RLRunner(args.cfg, args.ckpt)
    runner.train(args.steps)
