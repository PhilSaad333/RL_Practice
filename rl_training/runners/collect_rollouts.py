# rl_training/runners/collect_rollouts.py
from __future__ import annotations

import re
import json
import math
import time
from tqdm.auto import tqdm
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
    LogitsProcessor,
    LogitsProcessorList,
)

from rl_training.utils.rollout_buffer import RolloutBuffer, RolloutBatch
from rl_training.rewards import get_reward_fns       # factory that imports by name
from importlib import import_module


TAG_STOP = "</answer>"

class StopAfterAnswer(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tag_ids = tokenizer("</answer>", add_special_tokens=False).input_ids
        self.L       = len(self.tag_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        # rows already finished → mask everything except pad/eos token
        tag = torch.tensor(self.tag_ids, device=input_ids.device)
        done = (input_ids[:, -self.L:] == tag).all(-1)             # (B,)
        if done.any():
            # keep only the pad_token for finished rows
            scores[done] = float("-inf")
            scores[done, self.tokenizer.pad_token_id] = 0.0
        return scores






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
    tag_correct: float
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
        assert tokenizer.padding_side == "left"
        self.cfg = cfg
        self.device = device or policy.device
        self.G: int = cfg["num_generations"]
        self.B: int = cfg["microbatch_size"]
        self.batch_size: int = cfg["microbatch_size"]

        # for trimming generation tokens
        self.TAG_IDS  = tokenizer("</answer>", add_special_tokens=False).input_ids
        self.L_TAG    = len(self.TAG_IDS)
        self.TAG_TENS = torch.tensor(self.TAG_IDS, device=self.device)


        self.logits_processor = LogitsProcessorList([StopAfterAnswer(tokenizer)])

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


    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        """
        Collect `need = batch_prompts or self.batch_size` prompt-groups.

        Generates in parallel over a mini-batch of size `self.B`
        and pushes accepted groups one-by-one into the RolloutBuffer.
        """
        need   = batch_prompts or self.batch_size            # total prompt-groups wanted
        buffer = RolloutBuffer(capacity=need)

        ans_pat = re.compile(
            r"\n</think>\n<answer>\n[\s\S]+?\n</answer>$"    # fast format check
        )


        # ----------------------------------------------------------------------
        bar = tqdm(total=need, desc="Collecting rollouts", leave=False)

        while len(buffer) < need:
            # ── 1) sample a *mini-batch* of B prompts ─────────────────────────
            
            # Hard coded amount assuming G=8            
            take = min(6, need - len(buffer))           # don't overshoot buffer.
            #take = min(self.B, need - len(buffer))           # don't overshoot buffer.
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler,
                self.tokenizer,
                self.device,
                take
            )                                                # prompt_ids : [B, T_p]

            # ── 2) batched generation ─────────────────────────────────────────

            gen_out = self.policy.generate(
                prompt_ids,
                attention_mask      = attn,
                num_return_sequences= self.G,
                max_new_tokens      = self.cfg["max_new_tokens"],
                do_sample           = True,
                temperature         = self.cfg["temperature"],
                #top_p               = self.cfg["top_p"],
                pad_token_id        = self.tokenizer.pad_token_id,
                output_scores       = True,
                logits_processor    = self.logits_processor,
                return_dict_in_generate=True,
            )

            # reshape: (B*G, T) → (B, G, T)
            full_ids = gen_out.sequences.view(take, self.G, -1)
            gen_ids  = full_ids[:, :, prompt_ids.shape[1]:]              # (B, G, T_g)
            scores   = [s.view(take, self.G, -1) for s in gen_out.scores]

            before = len(buffer)

            # ── 3) loop *per prompt-group*  (keeps old reward / accept logic) ──
            for b in range(take):
                pid    = pids[b]
                q_text = ptxts[b]
                g_ids  = gen_ids[b]                                      # (G, T_g)

                # --- decode & clean ---
                g_txts = self.tokenizer.batch_decode(
                    g_ids, skip_special_tokens=True
                )
                # HERE WE MANUALLY STRIP OFF ANY EXTRA TEXT AFTER THE FIRST </answer> TAG
                # THE STOP CRITERION SHOULD HAVE PREVENTED THIS BUT SEEMS INCONSISTENT?
                g_txts = [
                    t.split("</answer>", 1)[0] + "</answer>"
                    if "</answer>" in t else t
                    for t in g_txts
                ]
                full_txts = self.tokenizer.batch_decode(
                    full_ids[b], skip_special_tokens=True
                )

                # --- token-level stats ---
                # use trimming logic from evals/utils_io.py

                lp_rows, ent_rows, gid_rows = [], [], []
                keep_max = 0
                for g in range(self.G):
                    ids_full = g_ids[g]                      # (T_raw,)
                    # 1) find first "</answer>"
                    cut = next((i+self.L_TAG
                                for i in range(ids_full.size(0)-self.L_TAG+1)
                                if torch.equal(ids_full[i:i+self.L_TAG],
                                               self.TAG_TENS)), ids_full.size(0))
                    keep_max = max(keep_max, cut) 

                    ids_trim = ids_full[:cut]                # (T_keep)
                    gid_rows.append(ids_trim)

                # -- teacher-forcing forward pass once per prompt-group ------
                #   Stack -> (G, T_max) ; right-pad *before* TF so shape is rectangular
                g_padded = torch.nn.utils.rnn.pad_sequence(
                    gid_rows, batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                lp_tf, ent_tf = _teacher_forcing_lp_ent(
                    self.policy, g_padded, self.tokenizer.pad_token_id
                )                                             # lists length G

                lp_rows  = lp_tf
                ent_rows = ent_tf


                # 4) pad AFTER trimming so all G sequences line up
                pad = lambda lst: torch.nn.utils.rnn.pad_sequence(
                    lst, batch_first=True, padding_value=0.0
                )

                lp_t   = pad(lp_rows)        # shape (G, keep_max)
                ent_t  = pad(ent_rows)
                g_ids  = pad(gid_rows).to(g_ids.device)                                               # (G, T_g_max_b)


                print("lp_t row 0 (first 20):", lp_t[0][:20])


                # --- rewards ---
                r_vec  = torch.stack([fn(pid, g_txts) for fn in self.reward_fns]).sum(0)

                # --- accept / difficulty ---
                accept = _accept_prompt_group(
                    r_vec,
                    thresh          = self.cfg["reward_var_thresh"],
                    allow_all_zero  = not self.cfg["reject_all_zero"],
                    allow_all_max   = not self.cfg["reject_all_max"],
                )
                succ   = (r_vec > 0).float().mean().item()
                prev   = self.win_rate_ema.get(pid, succ)
                self.win_rate_ema[pid] = 0.95 * prev + 0.05 * succ
                diff_tag = ("easy"   if self.win_rate_ema[pid] > 0.8 else
                            "hard"   if self.win_rate_ema[pid] < 0.2 else
                            "normal")

                tag_ok = torch.tensor(
                    [bool(ans_pat.search(t)) for t in g_txts],
                    dtype=torch.float32, device=self.device
                )
                t_len  = torch.tensor(
                    [_count_think_tokens(t, self.tokenizer) for t in full_txts],
                    dtype=torch.int32, device=self.device
                )

                # --- trace JSONL ---
                now = time.perf_counter()
                samples = [
                    GenSample(
                        prompt_id       = pid,
                        prompt_text     = q_text,
                        gen_text        = g_txts[g],
                        think_len       = int(t_len[g]),
                        reward          = float(r_vec[g]),
                        tag_correct     = float(tag_ok[g]),
                        include_in_batch= bool(accept),
                        difficulty_tag  = diff_tag,
                        token_entropy   = ent_t[g].tolist(),
                        token_logprob   = lp_t[g].tolist(),
                        generation_time_s = 0.0,   # fill if you want timing
                        step_idx        = self._step_idx,
                    )
                    for g in range(self.G)
                ]
                _append_jsonl(self._trace_file, samples)

                if accept and len(buffer) < need:
                    buffer.add(
                        prompt_ids = prompt_ids[b].cpu(),
                        gen_ids    = g_ids.cpu(),         # already trimmed & padded
                        rewards    = r_vec.cpu(),
                        logprobs   = lp_t.cpu(),
                        tag_correct= tag_ok.cpu(),
                        think_len  = t_len.cpu(),
                    )
                    del g_ids, lp_t, tag_ok, t_len        # free references

            # ── Advance the progress bar by however many we just added ───────────
            added = len(buffer) - before
            if added:
                bar.update(added)

            # end for b
        # end while
        bar.close()
        self._step_idx += 1
        return buffer    


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

def _next_prompt_batch(sampler, tokenizer, device, B):
    ids, texts, toks = [], [], []
    for _ in range(B):
        pid = next(sampler)
        q   = sampler.id2text[pid]
        ids.append(pid)
        texts.append(q if q.rstrip().endswith("<think>") else q + "\n<think>\n")
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    return ids, texts, batch["input_ids"].to(device), batch["attention_mask"].to(device)

@torch.inference_mode()
def teacher_forcing_logprobs(model, ids):
    # ids : (G, T) on current device
    attn   = (ids != tokenizer.pad_token_id)
    logits = model(ids, attention_mask=attn).logits           # (G, T, V)
    lp     = logits.log_softmax(-1).gather(
                 -1, ids.unsqueeze(-1)).squeeze(-1)           # (G, T)
    ent    = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
    return lp, ent                                            # tensors, no padding

