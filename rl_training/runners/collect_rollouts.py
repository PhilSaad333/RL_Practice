# rl_training/runners/collect_rollouts.py
from __future__ import annotations
import re, json, math, time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase,
    LogitsProcessor, LogitsProcessorList
)

# Removed unused import; we no longer compute full-vocab entropy  # Changed 8/11
# from rl_training.utils.logprob_entropy import compute_logprobs_and_entropy
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
    token_entropy: List[float]           # sampled entropy = -logprob(sampled token)  # Changed 8/11
    token_logprob: List[float]
    generation_time_s: float
    step_idx: int
    seq_logprob: float = 0.0             # Added 8/11: sum of token logprobs (for later analysis)


# --------------------------------------------------------------------------
# Roll-out collector
# --------------------------------------------------------------------------
class RolloutCollector:
    """
    Generates prompt groups, scores them, and fills a RolloutBuffer until it
    reaches the requested size.  Uses compute_transition_scores to obtain
    per-token log-probs from generation in a RAM-friendly microbatched way.
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
        self.cfg        = cfg

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        trace_name = f"rollouts_rank{self.rank}.jsonl"
        self._trace_file = Path(out_dir) / trace_name

        # Device handling
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        assert tokenizer.padding_side == "left"
        self.pad_id     = tokenizer.pad_token_id
        self.G          = cfg["num_generations"]
        self.B_opt      = cfg["microbatch_size"]
        self.batch_size = cfg.get("rollout_batch_size", self.B_opt)

        # tag look-ups for trimming
        self.TAG_IDS  = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
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

        # Removed old entropy modes; always compute sampled entropy cheaply  # Changed 8/11
        self.tf_micro_batch: int = max(1, int(cfg.get("tf_micro_batch", 8)))  # still used to slice scores  # Changed 8/11

    # ------------------------------------------------------------------
    # main public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        need   = batch_prompts or self.batch_size
        buffer = RolloutBuffer(capacity=need, pad_id=self.pad_id)
        ans_pat = re.compile(r"</think>\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)

        bar = tqdm(total=need, desc="Collecting rollouts", leave=False, disable=(self.rank != 0))

        while len(buffer) < need:
            # -------- 1) sample prompt mini-batch ----------------------
            take = self.batch_size
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler, self.tokenizer, self.device, take
            )

            # -------- 2) fast KV-cached batched generation -------------
            m = _unwrap(self.policy)
            was_training = m.training
            m.eval()
            old_cache = getattr(m.config, "use_cache", False)
            m.config.use_cache = True

            print(f"[DEBUG] Rank {self.rank}: Starting generation, synced_gpus={bool(dist.is_initialized())}")
            t0 = time.time()
            gen_out = m.generate(
                prompt_ids,
                attention_mask       = attn,
                do_sample            = True,
                num_return_sequences = self.G,
                max_new_tokens       = int(self.cfg["max_new_tokens"]),
                temperature          = float(self.cfg["temperature"]),
                pad_token_id         = self.pad_id,
                eos_token_id         = self.tokenizer.eos_token_id,
                logits_processor     = self.logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
                synced_gpus          = bool(dist.is_initialized()),
            )
            gen_time = time.time() - t0
            print(f"[DEBUG] Rank {self.rank}: Completed generation, took {gen_time:.2f}s")

            m.config.use_cache = old_cache
            if was_training: m.train()

            # reshape helper
            full_ids = gen_out.sequences.view(take, self.G, -1)          # (B,G,T_tot)
            gen_ids  = full_ids[:, :, prompt_ids.size(1):]               # (B,G,T_gen_pad)

            # -------- 3) microbatched transition logprobs --------------
            # We slice scores to reduce peak RAM. For each micro-batch of rows,
            # compute_transition_scores returns logprob(sampled_token) per step.
            seqs_flat   = gen_out.sequences.view(take * self.G, -1)      # (BG, L_full)
            scores_full = gen_out.scores                                 # list[L_gen] of (BG, vocab)

            lp_list: list[torch.Tensor] = []
            keep_lens: list[int] = []

            mb = self.tf_micro_batch
            P  = prompt_ids.size(1)

            for start_idx in range(0, seqs_flat.size(0), mb):
                rows = slice(start_idx, min(start_idx + mb, seqs_flat.size(0)))
                seqs_mb   = seqs_flat[rows]
                scores_mb = [s[rows] for s in scores_full]
                lp_mb = _unwrap(self.policy).compute_transition_scores(   # Changed 8/11
                    seqs_mb, scores_mb, normalize_logits=True
                )                                                         # (mb, L_gen_trim)
                for r in range(lp_mb.size(0)):
                    lp_seq = lp_mb[r].detach().to(torch.float32).cpu()    # numeric safety  # Changed 8/11
                    lp_seq = torch.nan_to_num(lp_seq, neginf=-80.0, posinf=0.0)            # Changed 8/11
                    lp_seq = torch.clamp(lp_seq, min=-80.0, max=0.0)                      # Changed 8/11
                    lp_list.append(lp_seq)
                    keep_lens.append(lp_seq.size(0))

            before = len(buffer)

            # -------- 4) iterate over prompt-groups --------------------
            for b in range(take):
                if len(buffer) >= need:
                    break
                print(f"[DEBUG] Rank {self.rank}: Processing sample {b+1}/{take}, buffer has {len(buffer)}/{need}")
                pid     = pids[b]
                q_text  = ptxts[b]
                g_ids_b = gen_ids[b]                                      # (G, T_gen_pad)

                # decode & truncate at first </answer>
                g_txts = self.tokenizer.batch_decode(g_ids_b, skip_special_tokens=True)
                g_txts = [t.split(TAG_STOP, 1)[0] + TAG_STOP if TAG_STOP in t else t for t in g_txts]

                # trim token IDs + collect per-token stats
                gid_rows, lp_rows, ent_rows = [], [], []
                keep_max = 0
                row_off = b * self.G

                for g in range(self.G):
                    ids_full = g_ids_b[g]
                    keep = keep_lens[row_off + g]
                    cut  = keep
                    keep_max = max(keep_max, cut)
                    ids_trim = ids_full[:cut]
                    gid_rows.append(ids_trim)

                    lp_seq = torch.tensor(lp_list[row_off + g], device=self.device)
                    lp_seq = torch.nan_to_num(lp_seq, neginf=-80.0, posinf=0.0)           # Changed 8/11
                    lp_seq = torch.clamp(lp_seq, min=-80.0, max=0.0)                      # Changed 8/11
                    lp_rows.append(lp_seq)

                    ent_rows.append(-lp_seq)  # sampled entropy per token  # Changed 8/11

                # pad to keep_max
                g_ids_t = pad(gid_rows, batch_first=True, padding_value=self.pad_id)
                lp_t    = pad(lp_rows,  batch_first=True, padding_value=0.0)
                ent_t   = pad(ent_rows, batch_first=True, padding_value=0.0)

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

                tag_ok = torch.tensor([bool(re.search(ans_pat, t)) for t in g_txts],
                                      dtype=torch.float32, device=self.device)
                t_len = torch.tensor([_count_think_tokens(t, self.tokenizer) for t in g_txts],
                                     dtype=torch.int32, device=self.device)

                # -------- 6) trace JSONL -------------------------------
                samples = []
                for g in range(self.G):
                    seq_lp = float(lp_t[g].sum().item())
                    samples.append(GenSample(
                        prompt_id        = pid,
                        prompt_text      = q_text,
                        gen_text         = g_txts[g],
                        think_len        = int(t_len[g]),
                        reward           = float(r_vec[g]),
                        tag_correct      = float(tag_ok[g]),
                        include_in_batch = bool(accept),
                        difficulty_tag   = diff_tag,
                        token_entropy    = (-lp_t[g]).tolist(),           # Changed 8/11
                        token_logprob    = lp_t[g].tolist(),
                        generation_time_s= 0.0,
                        step_idx         = self._step_idx,
                        seq_logprob      = seq_lp,                        # Added 8/11
                    ))
                _append_jsonl(self._trace_file, samples)

                # safety checks
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

            bar.update(len(buffer) - before)

        bar.close()
        self._step_idx += 1
        return buffer


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
    if "</think>" not in text:
        return 0
    inner = text.split("</think>", 1)[0].strip()
    return len(tok(inner, add_special_tokens=False).input_ids)


def _append_jsonl(file_path: Path, records: List[GenSample]):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    with open(file_path, "a") as fh:
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

def _unwrap(m):  # Added 8/11
    return m.module if hasattr(m, "module") else m  # Added 8/11
