# rl_training/runners/collect_rollouts.py (rewritten for faster collection)
from __future__ import annotations
import re, json, math, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizerBase,
    LogitsProcessor, LogitsProcessorList
)

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

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        # input_ids: (batch, cur_len)
        B, T = input_ids.size()
        if T < self.L:  # cannot contain the stop tag yet
            return scores
        tag = torch.tensor(self.tag_ids, device=input_ids.device)
        # check last L tokens for exact match; if so, set next-token logits to -inf for all tokens
        tail = input_ids[:, T-self.L:T]
        mask = (tail == tag).all(dim=1)
        if mask.any():
            scores[mask] = torch.finfo(scores.dtype).min
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
    token_entropy: List[float]      # -log p(token) for sampled tokens
    token_logprob: List[float]      # log p(token) for sampled tokens
    generation_time_s: float
    step_idx: int
    seq_logprob: float = 0.0

# --------------------------------------------------------------------------
# Roll-out collector
# --------------------------------------------------------------------------
class RolloutCollector:
    """
    Generates prompt groups, scores them, and fills a RolloutBuffer until it
    reaches the requested size. This version:
      • Enables KV cache during generation for speed
      • Drops generate(output_scores=True) and recomputes log-probs in a single
        teacher-forcing pass over (prompt || trimmed generation)
      • Optionally writes rollouts.jsonl only every N collect calls (or never)
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

        # Device handling
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # tokenizer setup
        assert tokenizer.padding_side == "left"
        self.pad_id     = tokenizer.pad_token_id
        self.G          = cfg["num_generations"]
        self.B_opt      = cfg["microbatch_size"]
        self.batch_size = cfg.get("rollout_batch_size", self.B_opt)

        # stop-tag lookups for trimming
        self.TAG_IDS  = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L_TAG    = len(self.TAG_IDS)
        self.TAG_TENS = torch.tensor(self.TAG_IDS, device=self.device)

        self.logits_processor = LogitsProcessorList([StopAfterAnswer(tokenizer)])

        # factories
        self.reward_fns     = get_reward_fns(cfg["reward_fns"])   # list of callables
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
        self.save_rollouts_every: int = int(cfg.get("save_rollouts_every", 1))  # 0 disables

        # teacher-forcing micro-batch (forward pass only)
        self.tf_micro_batch: int = max(1, int(cfg.get("tf_micro_batch", 8)))

    # ------------------------------------------------------------------
    # main public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        need   = batch_prompts or self.batch_size
        buffer = RolloutBuffer(capacity=need, pad_id=self.pad_id)
        ans_pat = re.compile(r"</think>\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)

        bar = tqdm(total=need, desc="Collecting rollouts", leave=False, disable=(self.rank != 0))

        should_trace = (self.save_rollouts_every > 0 and (self._step_idx % self.save_rollouts_every == 0))

        while len(buffer) < need:
            # -------- 1) sample prompt mini-batch ----------------------
            take = min(self.batch_size, need - len(buffer))
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler, self.tokenizer, self.device, take
            )

            # -------- 2) fast batched generation with KV cache --------
            m = _unwrap(self.policy)
            was_training = m.training
            m.eval()
            old_cache = getattr(m.config, "use_cache", False)
            m.config.use_cache = True

            t0 = time.time()
            gen_out = m.generate(
                prompt_ids,
                attention_mask       = attn,
                do_sample            = True,
                num_return_sequences = self.G,
                max_new_tokens       = self.cfg["max_new_tokens"],
                temperature          = self.cfg["temperature"],
                pad_token_id         = self.pad_id,
                logits_processor     = self.logits_processor,
                return_dict_in_generate=True,
                synced_gpus          = bool(dist.is_initialized()),
            )
            gen_time = time.time() - t0

            m.config.use_cache = old_cache
            if was_training: m.train()

            # -------- 3) cut out generated tokens & trim to first tag --
            full_ids = gen_out.sequences.view(take, self.G, -1)       # (B, G, P+Tpad)

            gid_rows: List[Tensor] = []
            keep_lens: List[int]   = []
            g_txts_by_b: List[List[str]] = []      # for rewards & logging

            for b in range(take):
                g_txts_b: List[str] = []
                Pmax   = prompt_ids.size(1)        # left-padded to same width
                g_rows = []
                for g in range(self.G):
                    raw = full_ids[b, g, Pmax:]    # drop prompt (uses fixed width)
                    eff = raw[: int((raw != self.pad_id).sum().item())]  # strip right-pad
                    cut = _first_tag_pos(eff, self.TAG_TENS)
                    trimmed = eff[:cut]
                    g_rows.append(trimmed)
                    keep_lens.append(int(trimmed.size(0)))
                # texts for rewards
                g_txts = self.tokenizer.batch_decode(pad(g_rows, True, self.pad_id), skip_special_tokens=True)
                g_txts_by_b.append(g_txts)
                gid_rows += g_rows

            # pad all generated rows to same length for one forward pass
            g_ids_t = pad(gid_rows, batch_first=True, padding_value=self.pad_id)  # (B*G, T_keep_max)

            # -------- 4) single TF forward to get token log-probs ------
            prompts_rep = prompt_ids.repeat_interleave(self.G, dim=0)            # (B*G, Pmax)
            seqs_flat   = torch.cat([prompts_rep, g_ids_t], dim=1)               # (B*G, Pmax+T)
            attn_mask   = (seqs_flat != self.pad_id).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = _unwrap(self.policy)(seqs_flat, attention_mask=attn_mask).logits
            logp_all = torch.log_softmax(logits.float(), dim=-1)
            targets  = seqs_flat[:, 1:]
            logp_tok = logp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B*G, Pmax+T-1)

            # keep only generated region (last T cols)
            T_keep = g_ids_t.size(1)
            logp_gen = logp_tok[:, -T_keep:]

            # split back per (b,g)
            lp_rows: List[Tensor] = []
            ent_rows: List[Tensor] = []
            row = 0
            for b in range(take):
                for g in range(self.G):
                    T = keep_lens[row]
                    lp = logp_gen[row, :T].to(torch.float32)
                    lp = torch.nan_to_num(lp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
                    lp_rows.append(lp)
                    ent_rows.append(-lp)
                    row += 1

            # -------- 5) rewards & accept ------------------------------
            before = len(buffer)
            row = 0
            for b in range(take):
                if len(buffer) >= need:
                    break
                pid     = pids[b]
                q_text  = ptxts[b]
                g_txts  = g_txts_by_b[b]

                r_vec = torch.stack([fn(pid, g_txts) for fn in self.reward_fns]).sum(0)  # (G,)
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

                # metadata derived from text
                tag_ok = torch.tensor([bool(re.search(ans_pat, t)) for t in g_txts],
                                      dtype=torch.float32, device=self.device)
                t_len = torch.tensor([_count_think_tokens(t, self.tokenizer) for t in g_txts],
                                     dtype=torch.int32, device=self.device)

                # pack per-(B,G) tensors for buffer & jsonl
                lp_t  = pad(lp_rows[row:row+self.G],  batch_first=True, padding_value=0.0)
                ent_t = pad(ent_rows[row:row+self.G], batch_first=True, padding_value=0.0)
                # reconstruct gen ids for this prompt from gid_rows slice
                g_slice = gid_rows[row:row+self.G]
                g_ids_t_b = pad(g_slice, batch_first=True, padding_value=self.pad_id)
                row += self.G

                # -------- 6) optional trace JSONL ----------------------
                if should_trace:
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
                            token_entropy    = (-lp_t[g]).tolist(),
                            token_logprob    = lp_t[g].tolist(),
                            generation_time_s= gen_time / max(1, self.G * take),
                            step_idx         = self._step_idx,
                            seq_logprob      = seq_lp,
                        ))
                    _append_jsonl(self._trace_file, samples)

                # -------- 7) push to RolloutBuffer ---------------------
                if accept and len(buffer) < need:
                    buffer.add(
                        prompt_ids = prompt_ids[b].cpu(),
                        gen_ids    = g_ids_t_b.cpu(),
                        rewards    = r_vec.cpu(),
                        logprobs   = lp_t.cpu(),
                        tag_correct= tag_ok.cpu(),
                        think_len  = t_len.cpu(),
                    )

            pushed = len(buffer) - before
            for _ in range(pushed):
                bar.update(1)

        bar.close()
        self._step_idx += 1
        return buffer

# ------------------------------------------------------------------
# helpers
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

def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def _first_tag_pos(ids_row: Tensor, tag: Tensor) -> int:
    # returns end position (i.e., index after the tag) or len(ids_row) if not found
    L = tag.numel()
    T = ids_row.numel()
    if T < L:
        return T
    # simple scan; T is at most ~200 so Python-level loop is fine
    for i in range(0, T - L + 1):
        if torch.equal(ids_row[i:i+L], tag):
            return i + L
    return T
