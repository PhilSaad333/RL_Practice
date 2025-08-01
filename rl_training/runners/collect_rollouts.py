# rl_training/runners/collect_rollouts.py
from __future__ import annotations
import re, json, math, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase,
    LogitsProcessor, LogitsProcessorList
)

from rl_training.utils.logprob_entropy import compute_logprobs_and_entropy
from rl_training.utils.rollout_buffer import RolloutBuffer, RolloutBatch
from rl_training.rewards import get_reward_fns
from importlib import import_module


TAG_STOP = "</answer>"

# --------------------------------------------------------------------------
# Stop criterion: mask everything after the first </answer>
# --------------------------------------------------------------------------
class StopAfterAnswer(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tag_ids   = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L         = len(self.tag_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        tag   = torch.tensor(self.tag_ids, device=input_ids.device)
        done  = (input_ids[:, -self.L:] == tag).all(-1)
        if done.any():
            scores[done] = float("-inf")
            scores[done, self.tokenizer.pad_token_id] = 0.0
        return scores


# --------------------------------------------------------------------------
# JSON-serialisable metadata per generation
# --------------------------------------------------------------------------
@dataclass
class GenSample:
    prompt_id: int
    prompt_text: str
    gen_text: str
    think_len: int
    reward: float
    tag_correct: float
    include_in_batch: bool
    difficulty_tag: str
    token_entropy: List[float]
    token_logprob: List[float]
    generation_time_s: float
    step_idx: int


# --------------------------------------------------------------------------
# Roll-out collector
# --------------------------------------------------------------------------
class RolloutCollector:
    """
    Generates prompt groups, scores them, and fills a RolloutBuffer until it
    reaches the requested size.  Uses compute_transition_scores to obtain
    per-token log-probs in one pass (no teacher-forcing rerun needed).
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
        self.policy     = policy
        self.tokenizer  = tokenizer
        self.device     = device or policy.device
        self.cfg        = cfg

        assert tokenizer.padding_side == "left"
        self.pad_id     = tokenizer.pad_token_id
        self.G          = cfg["num_generations"]
        self.B_opt      = cfg["microbatch_size"]
        self.batch_size = cfg.get("rollout_batch_size",
                                  self.B_opt)  # different than B because rollout collection uses less RAM

        # tag look-ups for trimming
        self.TAG_IDS  = tokenizer(TAG_STOP,
                                  add_special_tokens=False).input_ids
        self.L_TAG    = len(self.TAG_IDS)
        self.TAG_TENS = torch.tensor(self.TAG_IDS, device=self.device)

        self.logits_processor = LogitsProcessorList([StopAfterAnswer(tokenizer)])

        # factories
        self.reward_fns     = get_reward_fns(cfg["reward_fns"])
        sched_cfg           = cfg["scheduler"]
        sched_mod           = import_module(f"rl_training.schedulers.{sched_cfg['name']}")
        self.prompt_sampler = sched_mod.get_prompt_sampler(sched_cfg)

        # bookkeeping
        self.win_rate_ema: Dict[int, float] = {}
        self._step_idx   = 0

        # persistence
        self.out_dir     = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self.out_dir / "rollouts.jsonl"

        # RAM/entropy mode (default "full"): choose "lite" for micro‑batched teacher forcing or "none" to skip entropy.
        self.entropy_mode: str = cfg.get("rollout_entropy_mode", "full").lower()
        self.tf_micro_batch: int = cfg.get("tf_micro_batch", 4)



    # ------------------------------------------------------------------
    # main public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        need   = batch_prompts or self.batch_size
        buffer = RolloutBuffer(capacity=need, pad_id=self.pad_id)
        ans_pat = re.compile(r"\n</think>\n<answer>\n[\s\S]+?\n</answer>$")

        bar = tqdm(total=need, desc="Collecting rollouts", leave=False)

        while len(buffer) < need:

            # -------- 1) sample prompt mini-batch ----------------------
            take = min(self.batch_size, need - len(buffer))   # CHANGE
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler, self.tokenizer, self.device, take
            )                                                  # prompt_ids: [B, T_p]

            # -------- 2) batched generation ---------------------------
            gen_out = self.policy.generate(
                prompt_ids,
                attention_mask       = attn,
                do_sample            = True,
                num_return_sequences = self.G,
                max_new_tokens       = self.cfg["max_new_tokens"],
                temperature          = self.cfg["temperature"],
                pad_token_id         = self.pad_id,
                logits_processor     = self.logits_processor,
                output_scores        = (self.entropy_mode == "full"),
                return_dict_in_generate = True,
            )

            # reshape helper
            full_ids = gen_out.sequences.view(take, self.G, -1)          # (B,G,T_tot)
            gen_ids  = full_ids[:, :, prompt_ids.size(1):]               # (B,G,T_gen_pad)

            # If we requested output_scores, keep them; otherwise set None
            scores = [s.view(take, self.G, -1) for s in gen_out.scores] if self.entropy_mode == "full" else None

            if self.entropy_mode == "full":
                # current fast path: one call to compute_transition_scores
                all_lp = self.policy.compute_transition_scores(
                    gen_out.sequences, gen_out.scores, normalize_logits=True
                )  # (B*G, Lgen_trim)
            else:
                # ram‑lite or no entropy: compute log‑probs (and entropies) via teacher forcing
                seqs_flat = gen_out.sequences.view(take * self.G, -1)
                lp_list, ent_list, keep_lens = compute_logprobs_and_entropy(
                    self.policy,
                    self.tokenizer,
                    seqs_flat,
                    prompt_ids.size(1),
                    self.tf_micro_batch,
                    temperature=self.cfg["temperature"],
                    compute_entropy=(self.entropy_mode != "none"),
                    stop_tag=TAG_STOP,
                )
                # lp_list/ent_list/keep_lens are Python lists; we'll index them below

            before = len(buffer)

            # -------- 4) iterate over prompt-groups --------------------
            for b in range(take):
                pid     = pids[b]
                q_text  = ptxts[b]
                g_ids_b = gen_ids[b]                                      # (G, T_gen_pad)

                # ~~~ decode & truncate at first </answer> tag ~~~
                g_txts = self.tokenizer.batch_decode(
                    g_ids_b, skip_special_tokens=True
                )
                g_txts = [t.split(TAG_STOP, 1)[0] + TAG_STOP
                          if TAG_STOP in t else t
                          for t in g_txts]

                full_txts = self.tokenizer.batch_decode(
                    full_ids[b], skip_special_tokens=True
                )

                # ~~~ trim token IDs + collect per-token stats ~~~
                gid_rows, lp_rows, ent_rows = [], [], []
                keep_max = 0
                row_off = b * self.G

                for g in range(self.G):
                    ids_full = g_ids_b[g]
                    if self.entropy_mode == "full":
                        # trim ids using original search and use all_lp
                        cut  = next((i + self.L_TAG for i in range(ids_full.size(0) - self.L_TAG + 1)
                                    if torch.equal(ids_full[i:i + self.L_TAG], self.TAG_TENS)),
                                    ids_full.size(0))
                        keep_max = max(keep_max, cut)
                        ids_trim = ids_full[:cut]
                        gid_rows.append(ids_trim)
                        lp_seq = all_lp[row_off + g, :cut].clone()
                        lp_seq[lp_seq == float("-inf")] = 0.0
                        lp_rows.append(lp_seq)
                    else:
                        # ram‑lite / no entropy: use precomputed keep_lens and lp_list
                        keep = keep_lens[row_off + g]
                        cut  = keep  # number of generation tokens (no TAG length here)
                        keep_max = max(keep_max, cut)
                        ids_trim = ids_full[:cut]
                        gid_rows.append(ids_trim)
                        lp_seq = torch.tensor(lp_list[row_off + g], device=self.device)
                        lp_rows.append(lp_seq)

                # compute entropies
                if self.entropy_mode == "full":
                    # original vectorised entropy computation:contentReference[oaicite:3]{index=3}
                    ent_steps = [(-torch.softmax(gen_out.scores[t], -1) *
                                torch.log_softmax(gen_out.scores[t], -1)).sum(-1)[row_off:row_off + self.G]
                                for t in range(len(gen_out.scores))]
                    ent_rows = [torch.stack([ent_steps[t][g] for t in range(len(ent_steps))])
                                for g in range(self.G)]
                elif self.entropy_mode == "lite":
                    # ent_list contains per‑token entropies in CPU tensors
                    ent_rows = [torch.tensor(ent_list[row_off + g], device=self.device) for g in range(self.G)]
                else:
                    # "none": no entropy → use empty tensors of appropriate length
                    ent_rows = [torch.tensor([], device=self.device) for _ in range(self.G)]

                # now pad everything to keep_max
                g_ids_t = pad(gid_rows, batch_first=True, padding_value=self.pad_id)
                lp_t    = pad(lp_rows,  batch_first=True, padding_value=0.0)
                # ent_rows may be empty lists; pad gracefully by padding_value=0.0
                if ent_rows and ent_rows[0].numel() > 0:
                    ent_t = pad(ent_rows, batch_first=True, padding_value=0.0)
                else:
                    # create a zero tensor of shape (G, keep_max) for consistency
                    ent_t = torch.zeros((self.G, keep_max), device=self.device, dtype=torch.float32)


                # -------- 5) rewards & accept --------------------------
                r_vec = torch.stack([fn(pid, g_txts) for fn in self.reward_fns]).sum(0)

                accept = _accept_prompt_group(
                    r_vec,
                    thresh          = self.cfg["reward_var_thresh"],
                    allow_all_zero  = not self.cfg["reject_all_zero"],
                    allow_all_max   = not self.cfg["reject_all_max"],
                )
                succ     = (r_vec > 0).float().mean().item()
                prev     = self.win_rate_ema.get(pid, succ)
                self.win_rate_ema[pid] = 0.95 * prev + 0.05 * succ
                diff_tag = ("easy" if self.win_rate_ema[pid] > 0.8 else
                            "hard" if self.win_rate_ema[pid] < 0.2 else "normal")

                tag_ok = torch.tensor(
                    [bool(re.search(ans_pat, t)) for t in g_txts],
                    dtype=torch.float32, device=self.device
                )
                t_len = torch.tensor(
                    [_count_think_tokens(t, self.tokenizer) for t in full_txts],
                    dtype=torch.int32, device=self.device
                )

                # -------- 6) trace JSONL -------------------------------
                now = time.perf_counter()
                samples = [
                    GenSample(
                        prompt_id        = pid,
                        prompt_text      = q_text,
                        gen_text         = g_txts[g],
                        think_len        = int(t_len[g]),
                        reward           = float(r_vec[g]),
                        tag_correct      = float(tag_ok[g]),
                        include_in_batch = bool(accept),
                        difficulty_tag   = diff_tag,
                        token_entropy = ent_t[g].tolist() if self.entropy_mode != "none" else [],
                        token_logprob    = lp_t[g].tolist(),   # padded 0.0 where pad tokens
                        generation_time_s= 0.0,
                        step_idx         = self._step_idx,
                    ) for g in range(self.G)
                ]
                _append_jsonl(self._trace_file, samples)


                assert torch.isfinite(lp_t).all(), "non-finite old log-probs detected"


                # -------- 7) push to RolloutBuffer ---------------------
                if accept and len(buffer) < need:
                    buffer.add(
                        prompt_ids = prompt_ids[b].cpu(),
                        gen_ids    = g_ids_t.cpu(),
                        rewards    = r_vec.cpu(),
                        logprobs   = lp_t.cpu(),
                        tag_correct= tag_ok.cpu(),
                        think_len  = t_len.cpu(),
                    )
                    del g_ids_t, lp_t, tag_ok, t_len

            # -- progress bar ------------------------------------------
            bar.update(len(buffer) - before)

        bar.close()
        self._step_idx += 1
        return buffer


# ------------------------------------------------------------------
# helpers – unchanged except where noted
# ------------------------------------------------------------------
def _accept_prompt_group(
    rewards: Tensor, *, thresh: float,
    allow_all_zero: bool, allow_all_max: bool
) -> bool:
    if rewards.var(unbiased=False).item() < thresh:
        return False
    if not allow_all_zero and torch.all(rewards == 0):
        return False
    if not allow_all_max and torch.all(rewards == rewards.max()):
        return False
    return True


def _count_think_tokens(text: str, tok: PreTrainedTokenizerBase) -> int:
    if "<think>" not in text or "</think>" not in text:
        return 0
    inner = text.split("<think>", 1)[1].split("</think>", 1)[0]
    return len(tok(inner, add_special_tokens=False).input_ids)


def _append_jsonl(file: Path, records: List[GenSample]):
    with file.open("a") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")


def _next_prompt_batch(sampler, tokenizer, device, B):
    ids, texts = [], []
    for _ in range(B):
        pid = next(sampler)
        q   = sampler.id2text[pid]
        ids.append(pid)
        texts.append(q if q.rstrip().endswith("<think>") else q + "\n<think>\n")

    batch = tokenizer(texts, return_tensors="pt", padding=True)
    return ids, texts, batch["input_ids"].to(device), batch["attention_mask"].to(device)

# ---------------------  END OF FILE  ---------------------------------
