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
from copy import deepcopy


RUN_ROOT = pathlib.Path("/content/drive/MyDrive/RL_Practice_Files/rl_runs")

def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(prompt_ids = rb.prompt_ids[sl],
                        gen_ids    = rb.gen_ids[sl],
                        reward     = rb.reward[sl],
                        logprobs   = rb.logprobs[sl])

class RLRunner:
    def __init__(self, cfg_path: str, lora_ckpt: str):
        # ---------- I/O ---------------------------------------------------
        cfg = yaml.safe_load(open(cfg_path))
        stamp     = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir  = RUN_ROOT / f"run_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg_path, self.dir / "config.yaml")       # save config

        self.tb   = SummaryWriter(log_dir=str(self.dir))
        self.save_every = cfg.get("save_every", 100) #updated
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

        # frozen model for KL
        self.ref_model = deepcopy(self.model).eval().requires_grad_(False)
        # ensure pads match
        self.ref_model.config.pad_token_id = self.model.config.pad_token_id



        # ---------- subsystems ------------------------------------------
        self.collector = RolloutCollector(self.model, self.tok, cfg,
                                          out_dir=self.dir, device="cuda")
        self.algo      = GRPO(self.model, cfg, pad_id=self.tok.pad_token_id)
        self.accum = cfg["grad_accum_steps"]

        # for debug
        assert isinstance(self.tok.pad_token_id, int), "pad_token_id must not be None"


    # ---------------- main training loop ------------------------------
    def train(self, total_updates: int = 1000):
        """total_updates == number of *optimizer* steps (same definition GRPO uses)."""
        K         = self.collector.cfg["ppo_epochs"]
        ga_steps  = self.accum                     # == cfg["grad_accum_steps"]
        B         = self.collector.batch_size      # == cfg["microbatch_size"]

        outer_loops = math.ceil(total_updates / K)
        p_per_outer = K * ga_steps * B

        for _ in trange(outer_loops, desc="outer collect loops"):
            rb = self.collector.collect_batch(batch_prompts=p_per_outer)
            self._train_one_buffer(rb, K, ga_steps, B)

            if self.step_id % self.save_every == 0:
                self._save_ckpt()

        self._save_ckpt(final=True)

    def _train_one_buffer(self, rb, K, ga_steps, B):
        """Run K PPO epochs over RolloutBuffer `rb`."""
        stats_running = {}
        mb_counter    = 0

        for epoch in range(K):
            for idx in rb.iter_minibatches(B, shuffle=True):
                sync = ((mb_counter + 1) % ga_steps == 0)
                mb   = rb.get_batch(idx, device="cuda")

                stats = self.algo.step(mb, self.ref_model, sync_grads=sync)
                for k, v in stats.items():
                    stats_running[k] = stats_running.get(k, 0.0) + v / (K * ga_steps)

                if sync:
                    self.step_id += 1         # 1 optimiser update == 1 RL step
                mb_counter += 1

        self._log(stats_running)

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
