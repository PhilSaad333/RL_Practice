# rl_training/runners/collect_rollouts.py
from __future__ import annotations
import re, json, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

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
# Safe logits processor: if a row has just produced '</answer>', force EOS.
# This avoids "all -inf" logits (which break softmax).
# --------------------------------------------------------------------------
class ForceEosAfterAnswer(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError(f"Tokenizer could not encode stop tag {TAG_STOP!r}")
        self.tag_ids = ids
        self.L = len(ids)
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None; set a valid EOS token.")

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        # input_ids: (B, T), scores: (B, V)
        B, T = input_ids.size()
        if T < self.L:
            return scores
        tag = torch.tensor(self.tag_ids, device=input_ids.device)
        tail = input_ids[:, T - self.L : T]
        mask = (tail == tag).all(dim=1)  # (B,)
        if mask.any():
            scores[mask] = torch.finfo(scores.dtype).min
            scores[mask, self.eos_id] = 0  # make EOS the only viable choice
        return scores

# --------------------------------------------------------------------------
# JSON-serialisable metadata per generation (optional)
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
# Rollout collector
# --------------------------------------------------------------------------
class RolloutCollector:
    """
    Fast rollout collection:
      • Enables KV cache during generation for speed
      • Drops generate(output_scores=True) and recomputes token log-probs with
        one teacher-forcing forward over (prompt || trimmed generation)
      • Optional JSONL tracing every N collect calls (or never)
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
        self.rank       = dist.get_rank() if dist.is_initialized() else 0
        self.device     = torch.device(device) if device is not None else torch.device("cpu")

        # tokenizer / padding sanity
        if tokenizer.pad_token_id is None:
            # common practice with chat models
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.pad_id = tokenizer.pad_token_id

        # sizes
        self.G          = int(cfg["num_generations"])
        self.batch_size = int(cfg.get("rollout_batch_size", cfg.get("microbatch_size", 1)))

        # stop-tag encodings
        self.TAG_IDS  = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.TAG_TENS = torch.tensor(self.TAG_IDS, device=self.device)

        # logits processor to force EOS when '</answer>' is seen at tail
        self.logits_processor = LogitsProcessorList([ForceEosAfterAnswer(tokenizer)])

        # factories
        self.reward_fns = get_reward_fns(cfg["reward_fns"])   # list[Callable]
        sched_cfg       = cfg["scheduler"]
        sched_mod       = import_module(f"rl_training.schedulers.{sched_cfg['name']}")
        self.prompt_sampler = sched_mod.get_prompt_sampler(sched_cfg)

        # persistence
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self.out_dir / "rollouts.jsonl"
        # 0 = never, 1 = every call, N = every Nth call
        self.save_rollouts_every: int = int(cfg.get("save_rollouts_every", 1))
        self._collect_calls = 0

        # light EMA for difficulty tagging
        self.win_rate_ema: Dict[int, float] = {}

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def collect_batch(self, batch_prompts: int | None = None) -> RolloutBatch:
        need   = batch_prompts or self.batch_size
        buffer = RolloutBuffer(capacity=need, pad_id=self.pad_id)
        ans_pat = re.compile(r"</think>\s*<answer>\s*(.*?)\s*</answer>\s*$", re.DOTALL)

        bar = tqdm(total=need, desc="Collecting rollouts", leave=False, disable=(self.rank != 0))

        # should we JSONL-trace this call?
        self._collect_calls += 1
        should_trace = (self.save_rollouts_every > 0 and (self._collect_calls % self.save_rollouts_every == 0))

        while len(buffer) < need:
            # -------- 1) sample prompt mini-batch ----------------------
            take = min(self.batch_size, need - len(buffer))
            pids, ptxts, prompt_ids, attn = _next_prompt_batch(
                self.prompt_sampler, self.tokenizer, self.device, take
            )

            # -------- 2) fast KV-cached batched generation -------------
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
                max_new_tokens       = int(self.cfg["max_new_tokens"]),
                temperature          = float(self.cfg["temperature"]),
                pad_token_id         = self.pad_id,
                eos_token_id         = self.tokenizer.eos_token_id,
                logits_processor     = self.logits_processor,
                return_dict_in_generate=True,
                synced_gpus          = bool(dist.is_initialized()),
            )
            gen_time = time.time() - t0

            m.config.use_cache = old_cache
            if was_training: m.train()

            # -------- 3) split out generations & trim at first tag -----
            full_ids = gen_out.sequences.view(take, self.G, -1)       # (B, G, P+Tpad)
            Pmax = prompt_ids.size(1)                                 # fixed prompt width

            gid_rows: List[Tensor] = []    # flattened list of trimmed gen ids
            keep_lens: List[int]   = []    # lengths per (b,g)
            g_txts_by_b: List[List[str]] = []

            for b in range(take):
                rows_b: List[Tensor] = []
                for g in range(self.G):
                    raw = full_ids[b, g, Pmax:]                                   # drop prompt
                    eff = raw[: int((raw != self.pad_id).sum().item())]           # strip right pad
                    cut = _first_tag_pos(eff, self.TAG_TENS)
                    trimmed = eff[:cut]
                    rows_b.append(trimmed)
                    keep_lens.append(int(trimmed.size(0)))
                g_txts = self.tokenizer.batch_decode(
                    pad(rows_b, batch_first=True, padding_value=self.pad_id),
                    skip_special_tokens=True
                )
                g_txts_by_b.append(g_txts)
                gid_rows.extend(rows_b)

            # (BG, T_keep_max)
            g_ids_t = pad(gid_rows, batch_first=True, padding_value=self.pad_id)

            # -------- 4) single TF forward to get token log-probs ------
            prompts_rep = prompt_ids.repeat_interleave(self.G, dim=0)            # (BG, Pmax)
            seqs_flat   = torch.cat([prompts_rep, g_ids_t], dim=1)               # (BG, Pmax+T)
            attn_mask   = (seqs_flat != self.pad_id).long()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = _unwrap(self.policy)(seqs_flat, attention_mask=attn_mask).logits
            logp_all = torch.log_softmax(logits.float(), dim=-1)
            targets  = seqs_flat[:, 1:]
            logp_tok = logp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (BG, Pmax+T-1)

            # keep only generated region (last T columns)
            T_keep = g_ids_t.size(1)
            logp_gen = logp_tok[:, -T_keep:]

            # split back per (b,g)
            lp_rows: List[Tensor] = []
            ent_rows: List[Tensor] = []
            row = 0
            for _b in range(take):
                for _g in range(self.G):
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
                    thresh          = float(self.cfg["reward_var_thresh"]),
                    allow_all_zero  = not bool(self.cfg["reject_all_zero"]),
                    allow_all_max   = not bool(self.cfg["reject_all_max"]),
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

                lp_t  = pad(lp_rows[row:row+self.G],  batch_first=True, padding_value=0.0)
                ent_t = pad(ent_rows[row:row+self.G], batch_first=True, padding_value=0.0)
                g_slice = gid_rows[row:row+self.G]
                g_ids_t_b = pad(g_slice, batch_first=True, padding_value=self.pad_id)
                row += self.G

                # -------- 6) optional JSONL trace ----------------------
                if should_trace and (not dist.is_initialized() or self.rank == 0):
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
                            step_idx         = self._collect_calls,
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
    with open(file_path, "a") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r)) + "\n")

def _next_prompt_batch(sampler, tokenizer, device, B: int) -> Tuple[List[int], List[str], Tensor, Tensor]:
    ids, texts = [], []
    for _ in range(B):
        pid = next(sampler)
        q   = sampler.id2text[pid]
        # ensure we end in <think> to coax the chain-of-thought region
        texts.append(q if q.rstrip().endswith("<think>") else q + "\n<think>\n")
        ids.append(pid)
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    return ids, texts, batch["input_ids"].to(device), batch["attention_mask"].to(device)

def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def _first_tag_pos(ids_row: Tensor, tag: Tensor) -> int:
    # returns end position (index just after tag) or T if not found
    L = tag.numel()
    T = ids_row.numel()
    if T < L:
        return T
    # generation length is capped (e.g., 200), so a simple loop is fine and fast.
    for i in range(0, T - L + 1):
        if torch.equal(ids_row[i:i+L], tag):
            return i + L
    return T
