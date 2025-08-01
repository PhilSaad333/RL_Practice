# rl_training/utils/rollout_buffer.py


from __future__ import annotations

import random
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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

    def __init__(self, capacity: int, pad_id: int):
        self.capacity = capacity
        self._prompts: List[torch.LongTensor] = []
        self._gens: List[torch.LongTensor] = []
        self._rewards: List[torch.FloatTensor] = []
        self._logprobs: List[torch.FloatTensor] = []
        self._tag_correct: List[torch.FloatTensor] = []
        self._think_len: List[torch.IntTensor] = []
        self._G: int | None = None
        self._pad_id: int = pad_id



    def iter_minibatches(self, B, shuffle=True):
        """Yield indices of size B*G without replacement."""
        idx = list(range(len(self)))           # prompts, not generations!
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), B):
            yield idx[i : i+B]

    def get_batch(self, idx, device=None):
        sub = RolloutBuffer(capacity=len(idx), pad_id=self._pad_id)
        for j in idx:  # copy refs, no padding work
            sub._prompts.append(self._prompts[j])
            sub._gens.append(self._gens[j])
            sub._rewards.append(self._rewards[j])
            sub._logprobs.append(self._logprobs[j])
            sub._tag_correct.append(self._tag_correct[j])
            sub._think_len.append(self._think_len[j])
        return sub.to_batch(device=device)

    def add(
        self,
        *,
        prompt_ids: torch.LongTensor,          # (T_p)
        gen_ids: torch.LongTensor,             # (G, T_g)
        rewards: torch.FloatTensor,            # (G) or (G,1) or (1,G)
        logprobs: torch.FloatTensor,           # (G, T_g)
        tag_correct: torch.FloatTensor,        # (G)
        think_len: torch.IntTensor,            # (G)
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
        self._logprobs.append(logprobs.cpu())
        self._tag_correct.append(tag_correct.cpu())
        self._think_len.append(think_len.cpu())
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
            logprobs   : [B, G, T_gen_max]
            tag_correct: [B, G]
            think_len  : [B, G]

        Returns ``RolloutBatch`` (defined in algs/base.py).
        """
        assert len(self) > 0, "Buffer empty"
        B = len(self)
        G = self._G

        # ------------ left-pad prompts ---------------------------------------------
        # Need left padding so we can cleanly concatenate with gens
        # Apparently pad_sequence doesn't allow left-padding so must do it manually
        T_max = max(p.size(0) for p in self._prompts)          # longest prompt length
        padded_prompts = torch.stack(
            [F.pad(p, (T_max - p.size(0), 0), value=self._pad_id)         # pad (left, right)
            for p in self._prompts],
            dim=0                                              # (B, T_max)
        )

        # ------------ pad generations --------------------------------------------
        # Each element in self._gens is (G, T_gen_i); we pad on dim=1, then stack.
        gens_padded_per_prompt = [
            pad_sequence(list(g), batch_first=True, padding_value=self._pad_id)      # (G, T_gen_max_i)
            for g in self._gens
        ]
        # find global T_gen_max
        T_gen_max = max(g.shape[1] for g in gens_padded_per_prompt)
        # final pad so every prompt has the same T_gen_max
        gens_padded_per_prompt = [
            torch.nn.functional.pad(g, (0, T_gen_max - g.shape[1]), value=self._pad_id)       # pad rhs
            for g in gens_padded_per_prompt
        ]
        padded_gens = torch.stack(gens_padded_per_prompt, dim=0)          # (B, G, T_gen_max)

        # ------------ pad logprobs ----------------------------------------------
        # Same as gens
        logprobs_padded_per_prompt = [
            pad_sequence(list(g), batch_first=True, padding_value=0)      # (G, T_gen_max_i)
            for g in self._logprobs
        ]
        logprobs_padded_per_prompt = [
            torch.nn.functional.pad(g, (0, T_gen_max - g.shape[1]))       # pad rhs
            for g in logprobs_padded_per_prompt
        ]

        padded_logprobs = torch.stack(logprobs_padded_per_prompt, dim=0)          # (B, G, T_gen_max)

        # ------------ rewards ----------------------------------------------------
        rewards = torch.stack(self._rewards, dim=0)                       # (B, G)
        tag_correct = torch.stack(self._tag_correct, dim=0)               # (B, G)
        think_len = torch.stack(self._think_len, dim=0)                   # (B, G)

        if device is not None:
            padded_prompts = padded_prompts.to(device)
            padded_gens = padded_gens.to(device)
            rewards = rewards.to(device)
            padded_logprobs = padded_logprobs.to(device)
            tag_correct = tag_correct.to(device)
            think_len = think_len.to(device)

        # ------------ pack -------------------------------------------------------
        return RolloutBatch(
            prompt_ids=padded_prompts, # right-padded, shape (B, T_p_max)
            gen_ids=padded_gens, # right-padded, shape (B, G, T_gen_max)
            reward=rewards, # shape (B, G)
            logprobs=padded_logprobs, # shape (B, G, T_gen_max)
            tag_correct=tag_correct,  # shape (B, G)
            think_len=think_len,      # shape (B, G)
        )
