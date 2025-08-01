# rl_training/utils/logprob_entropy.py
"""
Memory‑efficient computation of per‑token log‑probabilities (and entropies)
using teacher forcing.  Designed for use during rollout collection when
output_scores=False to avoid storing large logits lists.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import torch

def compute_logprobs_and_entropy(
    model: torch.nn.Module,
    tokenizer,
    seqs: torch.Tensor,
    prompt_len: int,
    micro_batch_size: int,
    *,
    temperature: float = 1.0,
    compute_entropy: bool = True,
    stop_tag: str="</answer>",
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], List[int]]:
    """
    Arguments
    ---------
    model, tokenizer : the policy and its tokenizer.
    seqs            : (B, T_total) tensor of left‑padded prompt + right‑padded generation.
    prompt_len      : number of prompt tokens per row.
    micro_batch_size: number of rows to process per teacher‑forcing micro‑batch.
    temperature     : rescale logits before softmax.
    compute_entropy : whether to return per‑token entropies.

    Returns
    -------
    lp_list   : list of 1‑D tensors with per‑token log‑probs for each row.
    ent_list  : None or list of per‑token entropies for each row.
    keep_lens : list of ints indicating how many generation tokens were kept (trimmed at the first stop tag).
    """
    amp_dtype = torch.bfloat16 if getattr(model, "dtype", None) == torch.bfloat16 else torch.float16
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        att = (seqs != pad_id).int()
        seq_lens    = att.sum(-1)
        prompt_lens = att[:, :prompt_len].sum(-1)
        gen_lens    = seq_lens - prompt_lens

    tag_ids = tokenizer(stop_tag, add_special_tokens=False).input_ids
    L_tag = len(tag_ids)
    tag_tensor = torch.tensor(tag_ids, device=seqs.device) if L_tag > 0 else None

    lp_list: List[torch.Tensor] = []
    ent_list: Optional[List[torch.Tensor]] = [] if compute_entropy else None
    keep_lens: List[int] = []

    total_rows = seqs.size(0)
    for i in range(0, total_rows, micro_batch_size):
        blk = seqs[i : i + micro_batch_size]
        mb_size = blk.size(0)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(blk).logits
        if temperature != 1.0:
            logits = logits / temperature
        logp_all = logits.log_softmax(-1)
        if compute_entropy:
            probs_all = logp_all.exp()
            ent_all = -(probs_all * logp_all).sum(-1)
        targets_tok = blk[:, 1:].unsqueeze(-1)
        lp_all = logp_all[:, :-1].gather(2, targets_tok).squeeze(-1)
        for row in range(mb_size):
            idx = i + row
            g_len = int(gen_lens[idx])
            start = prompt_len - 1
            # default: keep all generation tokens
            keep = g_len
            if L_tag > 0:
                search_start = prompt_len
                search_end   = prompt_len + g_len - L_tag + 1
                row_ids = blk[row]
                for j in range(search_start, max(search_end, search_start)):
                    if torch.equal(row_ids[j : j + L_tag], tag_tensor):
                        keep = (j + L_tag) - prompt_len
                        break
            keep_lens.append(keep)
            lp_seq = lp_all[row, start : start + keep].detach().cpu()
            lp_list.append(lp_seq)
            if compute_entropy:
                ent_seq = ent_all[row, start : start + keep].detach().cpu()
                ent_list.append(ent_seq)  # type: ignore
        # free GPU memory early
        del logits, logp_all, lp_all
        if compute_entropy:
            del probs_all, ent_all
        torch.cuda.empty_cache()
    return lp_list, ent_list, keep_lens
