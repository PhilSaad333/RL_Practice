# rl_training/runners/collect_rollouts.py
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from rl_training.utils.rollout_buffer import RolloutBuffer, RolloutBatch
from rl_training.rewards import get_reward_fns       # factory that imports by name
from rl_training.schedulers import get_prompt_sampler # factory for curriculum schedulers


# ──────────────────────────────────────────────────────────────────────────────
# Metadata stored per <prompt, generation> pair
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GenSample:
    prompt_id: int
    prompt_text: str
    gen_text: str
    reward: float
    include_in_batch: bool
    difficulty_tag: str                       # "easy" | "normal" | "hard"
    token_entropy: List[float]                # length == T_gen
    token_logprob: List[float]                # length == T_gen
    generation_time_s: float
    step_idx: int                             # global RL step this was collected


# ──────────────────────────────────────────────────────────────────────────────
# Rollout collector with acceptance-criteria & logging
# ──────────────────────────────────────────────────────────────────────────────
class RolloutCollector:
    """
    Handles *one* forward-generate-and-score loop.
    Keeps trying until it accumulates a batch that satisfies the user-defined
    acceptance rules (reward diversity, etc.).
    """

    def __init__(
        self,
        policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: Dict[str, Any],
        out_dir: str | Path,
        *,
        device: torch.device | str | None = None,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device or policy.device
        self.G: int = cfg["num_generations"]
        self.batch_size: int = cfg["num_prompts"]

        # factories so you can swap implementations via YAML
        self.reward_fns = get_reward_fns(cfg["reward_fns"])          # list[callable]
        self.prompt_sampler = get_prompt_sampler(cfg["scheduler"])   # yields prompt strings

        # rolling difficulty tracker
        self.win_rate_ema: Dict[int, float] = {}

        # persistence
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self.out_dir / "rollouts.jsonl"
        self._step_idx = 0

    # ──────────────────────────────────────────────────────────────────────────
    # public entry point
    # ──────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def collect_batch(self) -> RolloutBatch:
        """
        Keep sampling prompts → G generations each → rewards
        until                → we have `batch_size` prompts with
                               reward variance above threshold, etc.

        Returns:
            RolloutBatch ready for the trainer, plus side-effect:
            every attempt (accepted or rejected) is appended to rollouts.jsonl
        """
        buffer = RolloutBuffer(capacity=self.batch_size)

        while len(buffer) < self.batch_size:
            prompt = next(self.prompt_sampler)     # text str
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            prompt_id = hash(prompt) & 0xFFFFFFFF  # cheap stable-ish id

            # ------------------------------------------------------------------
            # generate G continuations with score capture
            # ------------------------------------------------------------------
            start_t = time.perf_counter()
            outputs = self.policy.generate(
                prompt_ids.repeat(self.G, 1),
                max_new_tokens=self.cfg["max_new_tokens"],
                do_sample=True,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                num_return_sequences=self.G,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            gen_ids: Tensor = outputs.sequences[:, prompt_ids.shape[1]:]   # (G, T_gen)
            scores: Sequence[Tensor] = outputs.scores                     # list len T_gen
            gen_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # ------------------------------------------------------------------
            # token-level stats
            # ------------------------------------------------------------------
            logprobs, entropies = _token_stats(gen_ids, scores)

            # ------------------------------------------------------------------
            # reward(s)
            # ------------------------------------------------------------------
            rewards = torch.stack([fn(prompt, gen_texts) for fn in self.reward_fns]).sum(0)
            # rewards: Tensor [G]

            # ------------------------------------------------------------------
            # acceptance test for this prompt group
            # ------------------------------------------------------------------
            accept = _accept_prompt_group(
                rewards,
                thresh=self.cfg["reward_var_thresh"],
                allow_all_zero=not self.cfg["reject_all_zero"],
                allow_all_max=not self.cfg["reject_all_max"],
            )

            end_t = time.perf_counter()

            # difficulty heuristic and EMA update
            success_rate = (rewards > 0).float().mean().item()
            prev = self.win_rate_ema.get(prompt_id, success_rate)
            self.win_rate_ema[prompt_id] = 0.95 * prev + 0.05 * success_rate
            diff_tag = (
                "easy" if self.win_rate_ema[prompt_id] > 0.8
                else "hard" if self.win_rate_ema[prompt_id] < 0.2
                else "normal"
            )

            # ------------------------------------------------------------------
            # trace every sample
            # ------------------------------------------------------------------
            records: List[GenSample] = []
            for g in range(self.G):
                records.append(
                    GenSample(
                        prompt_id=prompt_id,
                        prompt_text=prompt,
                        gen_text=gen_texts[g],
                        reward=rewards[g].item(),
                        include_in_batch=bool(accept),
                        difficulty_tag=diff_tag,
                        token_entropy=entropies[g],
                        token_logprob=logprobs[g],
                        generation_time_s=end_t - start_t,
                        step_idx=self._step_idx,
                    )
                )
            _append_jsonl(self._trace_file, records)

            # ------------------------------------------------------------------
            # if accepted, push to rollout buffer
            # ------------------------------------------------------------------
            if accept:
                buffer.add(
                    prompt_ids       = prompt_ids.squeeze(0),    #   [T_prompt]
                    gen_ids          = gen_ids,                 # G x T_gen
                    rewards          = rewards,                 #   [G]
                )

        self._step_idx += 1

        # pack tensor shapes to match RolloutBatch signature (B, …)
        return buffer.to_batch()


# ──────────────────────────────────────────────────────────────────────────────
# helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _token_stats(gen_ids: Tensor, scores: Sequence[Tensor]):
    """Compute per-token log-prob and entropy for each generated sequence."""
    logprobs: List[List[float]] = []
    entropies: List[List[float]] = []
    for t, score_t in enumerate(scores):                       # score_t : [G, vocab]
        probs = torch.softmax(score_t, dim=-1)                 #   (G, V)
        ent = -(probs * probs.log()).sum(-1)                   #   (G,)
        token_t = gen_ids[:, t]                                #   (G,)
        lp_t = torch.log_softmax(score_t, dim=-1).gather(
            -1, token_t.unsqueeze(-1)
        ).squeeze(-1)                                          # (G,)

        entropies.append(ent.tolist())
        logprobs.append(lp_t.tolist())

    # transpose :: list[G][T]
    entropies = list(map(list, zip(*entropies)))
    logprobs  = list(map(list, zip(*logprobs)))
    return logprobs, entropies


def _accept_prompt_group(
    rewards: Tensor,
    *,
    thresh: float,
    allow_all_zero: bool,
    allow_all_max: bool,
) -> bool:
    var = rewards.var(unbiased=False).item()
    if var < thresh:
        return False
    if not allow_all_zero and torch.all(rewards == 0):
        return False
    if not allow_all_max and torch.all(rewards == rewards.max()):
        return False
    return True


def _append_jsonl(file: Path, records: List[GenSample]):
    with file.open("a") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
