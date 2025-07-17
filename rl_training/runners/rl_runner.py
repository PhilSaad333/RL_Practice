# rl_training/runners/rl_runner.py
from __future__ import annotations
import math, time, json, pathlib
from typing import Dict, Any
import torch
from tqdm import trange
from rl_training.runners.collect_rollouts import RolloutCollector
from rl_training.algs.grpo import GRPO
from rl_training.algs.base import RolloutBatch

# --------------------------------------------------------------------------- #
# helper: slice a RolloutBatch along the prompt (batch) axis                  #
# --------------------------------------------------------------------------- #
def slice_batch(rb: RolloutBatch, sl: slice) -> RolloutBatch:
    return RolloutBatch(
        prompt_ids = rb.prompt_ids[sl],
        gen_ids    = rb.gen_ids[sl],
        reward     = rb.reward[sl],
        logprobs   = rb.logprobs[sl],
    )

class RLRunner:
    def __init__(
        self,
        policy, tokenizer,
        cfg: Dict[str, Any],
        out_dir: str | pathlib.Path = "runs/rl",
        device: torch.device | str | None = None,
    ):
        self.device   = device or torch.device("cuda")
        self.policy   = policy.to(self.device)

        # memory-saving toggles ------------------------------------------------
        if cfg.get("gradient_checkpointing", True):
            policy.gradient_checkpointing_enable()
            policy.enable_input_require_grads()
            policy.config.use_cache = False

        # instantiate subsystems ----------------------------------------------
        self.collector = RolloutCollector(policy, tokenizer, cfg, out_dir, device=self.device)
        self.algo      = GRPO(policy, cfg, pad_id=tokenizer.pad_token_id)

        self.accum    = cfg["grad_accum_steps"]
        self.global_t = 0
        self.log_file = pathlib.Path(out_dir) / "train_log.jsonl"


    def train_step(self) -> Dict[str, float]:
        rollouts = self.collector.collect_batch()          # big RolloutBatch
        B = rollouts.prompt_ids.size(0)
        mb = math.ceil(B / self.accum)                     # micro-batch size

        metrics_accum: Dict[str, float] = {}
        for i in range(0, B, mb):
            sync = ( (i + mb) >= B )                       # last micro-batch?
            stats = self.algo.step(
                slice_batch(rollouts, slice(i, i+mb)),
                sync_grads=sync
            )
            # running mean
            for k, v in stats.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v / self.accum

        self.global_t += 1
        self._log(metrics_accum)
        return metrics_accum

    # ----------------------------------------------------------------------- #
    def _log(self, d: Dict[str, float]):
        d = {"step": self.global_t, **d}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(d) + "\n")

# ----------------------------- convenience CLI ----------------------------- #
if __name__ == "__main__":
    import yaml, argparse
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out",   type=str, default="runs/rl")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tok.padding_side = "left"     # must match left-pad change earlier

    bnb_conf = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    mdl = AutoModelForCausalLM.from_pretrained(
              "microsoft/phi-2",
              quantization_config=bnb_conf,
              torch_dtype=torch.bfloat16)

    runner = RLRunner(mdl, tok, cfg, args.out)
    for _ in trange(args.steps):
        runner.train_step()
