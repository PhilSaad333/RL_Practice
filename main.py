# rl_verifiable_math/main.py
import argparse
from pathlib import Path

def finetune(cfg):
    from fine_tuning.sft import run_sft
    run_sft(cfg)

def evaluate(cfg):
    from evals.eval_harness import run_eval
    run_eval(cfg)

def rl_train(cfg):
    from rl_training.grpo import run_grpo
    run_grpo(cfg)

DISPATCH = dict(finetune=finetune, eval=evaluate, rl_train=rl_train)

def parse_args():
    p = argparse.ArgumentParser(prog="rl_verifiable_math")
    p.add_argument("--task", choices=DISPATCH, required=True)
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    DISPATCH[args.task](args.config)

if __name__ == "__main__":
    main()
