# rl_training/utils/rollout_buffer.py


from __future__ import annotations

from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from rl_training.algs.base import RolloutBatch


class RolloutBuffer:
    """
    One-shot buffer: you keep adding accepted prompt groups with `.add()` until
    you’re happy, then call `.to_batch()` once to pop a RolloutBatch.

    The buffer pads variable-length sequences so the trainer can consume a
    single contiguous tensor.  Prompts are padded on the right with 0; generated
    continuations are padded on the right with 0 across the *generation* axis.

    Note
    ----
    * We assert that every call to `.add()` uses the same ``G`` (#generations
      per prompt).  This is true for our collector.
    * Padding token ``0`` is safe because loss is never computed on prompt
      tokens, and the trainer masks out padded generation tokens by length.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._prompts: List[torch.LongTensor] = []
        self._gens: List[torch.LongTensor] = []
        self._rewards: List[torch.FloatTensor] = []
        self._G: int | None = None  # lazily fixed on first add

    # --------------------------------------------------------------------- #
    # mutating API
    # --------------------------------------------------------------------- #
    def add(
        self,
        *,
        prompt_ids: torch.LongTensor,          # [T_p]
        gen_ids: torch.LongTensor,             # [G, T_g]
        rewards: torch.FloatTensor,            # [G] or [G,1] or [1,G]
    ) -> None:
        assert prompt_ids.dim() == 1, "prompt_ids must be 1-D"
        assert gen_ids.dim() == 2, "gen_ids must be 2-D (G, T_gen)"
        G_here = gen_ids.shape[0]
        # flatten any extra dims so we get exactly shape (G_here,)
        rewards = rewards.squeeze()
        if self._G is None:
            self._G = G_here
        else:
            assert G_here == self._G, f"Inconsistent G: expected {self._G}, got {G_here}"
        assert rewards.shape == (G_here,), f"rewards shape mismatch: got {tuple(rewards.shape)}, expected ({G_here},)"

        assert len(self) < self.capacity, "Buffer already full"
        self._prompts.append(prompt_ids.cpu())   # keep on CPU; move later
        self._gens.append(gen_ids.cpu())
        self._rewards.append(rewards.cpu())

    # --------------------------------------------------------------------- #
    # read-only API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._prompts)

    def to_batch(self, device: torch.device | str | None = None) -> RolloutBatch:
        """
        Pads and stacks the stored sequences:

            prompt_ids : [B, T_prompt_max]
            gen_ids    : [B, G, T_gen_max]
            reward     : [B, G]

        Returns ``RolloutBatch`` (defined in algs/base.py).
        """
        assert len(self) > 0, "Buffer empty"
        B = len(self)
        G = self._G

        # ------------ pad prompts ------------------------------------------------
        padded_prompts = pad_sequence(self._prompts, batch_first=True, padding_value=0)  # (B, T_p_max)

        # ------------ pad generations --------------------------------------------
        # Each element in self._gens is (G, T_gen_i); we pad on dim=1, then stack.
        gens_padded_per_prompt = [
            pad_sequence(list(g), batch_first=True, padding_value=0)      # (G, T_gen_max_i)
            for g in self._gens
        ]
        # find global T_gen_max
        T_gen_max = max(g.shape[1] for g in gens_padded_per_prompt)
        # final pad so every prompt has the same T_gen_max
        gens_padded_per_prompt = [
            torch.nn.functional.pad(g, (0, T_gen_max - g.shape[1]))       # pad rhs
            for g in gens_padded_per_prompt
        ]
        padded_gens = torch.stack(gens_padded_per_prompt, dim=0)          # (B, G, T_gen_max)

        # ------------ rewards ----------------------------------------------------
        rewards = torch.stack(self._rewards, dim=0)                       # (B, G)

        if device is not None:
            padded_prompts = padded_prompts.to(device)
            padded_gens = padded_gens.to(device)
            rewards = rewards.to(device)

        # ------------ pack -------------------------------------------------------
        return RolloutBatch(
            prompt_ids=padded_prompts,
            gen_ids=padded_gens,
            reward=rewards,
        )
