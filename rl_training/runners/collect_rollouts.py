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
    StoppingCriteria,
    StoppingCriteriaList,
)

from rl_training.utils.rollout_buffer import RolloutBuffer, RolloutBatch
from rl_training.rewards import get_reward_fns       # factory that imports by name
from importlib import import_module
from evals.utils_io import StopOnAnswer




# ──────────────────────────────────────────────────────────────────────────────
# Metadata stored per <prompt, generation> pair
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GenSample:
    prompt_id: int
    prompt_text: str
    gen_text: str
    think_len: int
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

        self.stopper = StoppingCriteriaList([StopOnAnswer(tokenizer)])

        # factories so you can swap implementations via YAML
        self.reward_fns = get_reward_fns(cfg["reward_fns"])          # list[callable]
        # dynamically load the right scheduler module by name
        sched_cfg = cfg["scheduler"]
        sched_mod = import_module(f"rl_training.schedulers.{sched_cfg['name']}")
        self.prompt_sampler = sched_mod.get_prompt_sampler(sched_cfg)  # yields prompt_id ints   # yields prompt_id ints

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
            prompt_id = next(self.prompt_sampler)     # int, don't get confused with tokens = prompt_ids
            question   = self.prompt_sampler.id2text[prompt_id]
            prompt_text = question + "\n<think>\n"

            batch = self.tokenizer(prompt_text, return_tensors="pt", padding=True)  # already left-pad
            prompt_ids     = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # ------------------------------------------------------------------
            # generate G continuations with score capture
            # ------------------------------------------------------------------
            start_t = time.perf_counter()
            outputs = self.policy.generate(
                prompt_ids,
                max_new_tokens=self.cfg["max_new_tokens"],
                do_sample=True,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                num_return_sequences=self.G,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                stopping_criteria=self.stopper,
                return_dict_in_generate=True,
            )
            full_ids: Tensor = outputs.sequences    # (G,T_total)
            gen_ids: Tensor = full_ids[:, prompt_ids.shape[1]:]   # (G, T_gen)
            scores: Sequence[Tensor] = outputs.scores                     # list len T_gen

            # Decode *generation only*  (for logging / buffer)
            try:
                gen_texts = self.tokenizer.batch_decode(
                    gen_ids,
                    skip_special_tokens=True,
                )
            except Exception as e:
                # fallback to empty strings so we don't crash
                print(f"[rollout] WARNING: decode failed: {e}")
                gen_texts = [""] * self.G

            # in case we somehow got an empty list
            if not isinstance(gen_texts, list) or len(gen_texts) != self.G:
                gen_texts = [""] * self.G

            # Immediately strip anything after </answer>
            gen_texts = [
                txt.split("</answer>", 1)[0] + "</answer>"
                if "</answer>" in txt else txt
                for txt in gen_texts
            ]

            # Decode *full prompt+gen* for think-length metric
            full_texts = self.tokenizer.batch_decode(
                full_ids, skip_special_tokens=True
            )

            # ------------------------------------------------------------------
            # token-level stats
            # ------------------------------------------------------------------
            logprobs, entropies = _token_stats(gen_ids, scores)

            lp_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(row) for row in logprobs],  # list[G] of 1-D
                batch_first=True, padding_value=0.0        # shape (G, T_gen_max_for_this_prompt)
            )

            # ------------------------------------------------------------------
            # reward(s)
            # ------------------------------------------------------------------
            rewards = torch.stack([fn(prompt_id, gen_texts) for fn in self.reward_fns]).sum(0)
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
                        prompt_text=prompt_text,
                        gen_text=gen_texts[g],
                        think_len=_count_think_tokens(full_texts[g], self.tokenizer),
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
                    logprobs         = lp_tensor
                )

        self._step_idx += 1

        # pack tensor shapes to match RolloutBatch signature (B, …)
        return buffer.to_batch(device=self.device)


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

def _count_think_tokens(text: str, tok: PreTrainedTokenizerBase) -> int:
    """
    Number of tokens between <think> ... </think>.
    Returns 0 if either tag is missing.
    """
    if "<think>" not in text or "</think>" not in text:
        return 0
    inner = text.split("<think>", 1)[1].split("</think>", 1)[0]
    return len(tok(inner, add_special_tokens=False).input_ids)


def _append_jsonl(file: Path, records: List[GenSample]):
    with file.open("a") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")
