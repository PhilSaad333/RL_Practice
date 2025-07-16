# rl_training/algs/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class RolloutBatch:
    prompt_ids:    torch.LongTensor  # (B, T_p_max)
    gen_ids:       torch.LongTensor  # (B, G, T_gen_max)
    reward:        torch.FloatTensor # (B, G)
    logprobs:      torch.FloatTensor # (B, G, T_gen_max)

class RLAlgorithm(ABC):
    def __init__(self, policy, cfg):
        self.policy = policy        # transformers.PreTrainedModel (LoRA merged or not)
        self.cfg = cfg

    @abstractmethod
    def step(self, rollouts: RolloutBatch) -> dict[str, float]:
        """One optimisation step; returns metric dict for logging."""
