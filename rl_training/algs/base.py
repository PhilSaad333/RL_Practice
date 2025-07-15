# rl_training/algs/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class RolloutBatch:
    prompt_ids:    torch.LongTensor  # [B, T_prompt]
    gen_ids:       torch.LongTensor  # [B, G, T_gen]
    reward:        torch.FloatTensor # [B, G]

class RLAlgorithm(ABC):
    def __init__(self, policy, cfg):
        self.policy = policy        # transformers.PreTrainedModel (LoRA merged or not)
        self.cfg = cfg

    @abstractmethod
    def step(self, rollouts: RolloutBatch) -> dict[str, float]:
        """One optimisation step; returns metric dict for logging."""
