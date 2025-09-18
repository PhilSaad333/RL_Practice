# rl_training/rewards/dummy_zero.py
import torch
def reward_fn(prompt: str, answers: list[str]) -> torch.FloatTensor:
    return torch.zeros(len(answers))
