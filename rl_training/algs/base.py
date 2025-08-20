# rl_training/algs/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

@dataclass
class RolloutBatch:
    prompt_ids:    torch.LongTensor  # (B, T_p_max) (left-padded)
    gen_ids:       torch.LongTensor  # (B, G, T_gen_max)
    reward:        torch.FloatTensor # (B, G)
    logprobs:      torch.FloatTensor # (B, G, T_gen_max)
    tag_correct:   torch.FloatTensor # (B, G)
    think_len:     torch.IntTensor   # (B, G)
    
    @staticmethod
    def concatenate(batches: list['RolloutBatch']) -> 'RolloutBatch':
        """
        Concatenate multiple RolloutBatch objects along the batch dimension.
        
        Args:
            batches: List of RolloutBatch objects to concatenate
            
        Returns:
            Single RolloutBatch with all batches concatenated along dim 0
        """
        if not batches:
            raise ValueError("Cannot concatenate empty list of batches")
            
        if len(batches) == 1:
            return batches[0]
            
        return RolloutBatch(
            prompt_ids=torch.cat([batch.prompt_ids for batch in batches], dim=0),
            gen_ids=torch.cat([batch.gen_ids for batch in batches], dim=0),
            reward=torch.cat([batch.reward for batch in batches], dim=0),
            logprobs=torch.cat([batch.logprobs for batch in batches], dim=0),
            tag_correct=torch.cat([batch.tag_correct for batch in batches], dim=0),
            think_len=torch.cat([batch.think_len for batch in batches], dim=0)
        )


class RLAlgorithm(ABC):
    def __init__(self, policy, cfg):
        self.policy = policy        # transformers.PreTrainedModel (LoRA merged or not)
        self.cfg = cfg

    @abstractmethod
    def step(self, rollouts: RolloutBatch) -> dict[str, float]:
        """One optimisation step; returns metric dict for logging."""
