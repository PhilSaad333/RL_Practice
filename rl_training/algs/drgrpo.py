# rl_training/algs/grpo.py
from __future__ import annotations
import json, pathlib
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from transformers import get_cosine_schedule_with_warmup
from .base import RLAlgorithm, RolloutBatch



class DRGRPO(RLAlgorithm):
    def __init__(
        self,
        policy, 
        cfg, 
        *, 
        pad_id: int | None = None,
        ratio_log_path: str | pathlib.Path | None = None,
        ):

        super().__init__(policy, cfg)
        total_updates   = cfg["total_steps"] 
        warmup_steps    = int(0.05 * total_updates) 
        self.opt    = torch.optim.AdamW(policy.parameters(),
                              lr=cfg["lr"], weight_decay=0.01)
        self.lr_sched = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps = warmup_steps,
            num_training_steps = total_updates,
            )
        self.pad_id = pad_id if pad_id is not None else getattr(policy.config, "pad_token_id", 0)
        self.accum_steps  = cfg["grad_accum_steps"]
        self._accum_ctr   = 0
        self.cfg = cfg
        self.device = None # set in step
        self.actual_opt_step = 0

        # if path is None → feature disabled (default in unit tests)
        self._ratio_log_path = pathlib.Path(ratio_log_path) \
                               if ratio_log_path else None   # CHANGE

    def step(self, rollouts: RolloutBatch, ref_model, *, sync_grads: bool = True) -> dict[str, float]:
        B, G, T_g = rollouts.gen_ids.shape
        device    = rollouts.gen_ids.device
        self.device = device
        pad_id    = self.pad_id
        scale      = 1.0 / self.accum_steps

        # compute GRPO advantage from rewards
        mean_r = rollouts.reward.mean(dim=1, keepdim=True)
        # std_r  = rollouts.reward.std (dim=1, keepdim=True) + 1e-8 # Not needed for DRGRPO
        adv    = (rollouts.reward - mean_r)                        # (B,G)

        prompt_rep = rollouts.prompt_ids.unsqueeze(1).expand(-1, G, -1)
        seq_ids    = torch.cat((prompt_rep, rollouts.gen_ids), dim=-1)    # (B,G,T_tot)
        seq_flat   = seq_ids.reshape(B * G, -1)                           # (BG,T_tot)

        attn_mask  = (seq_flat != pad_id).long()


        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.cfg["bf16"]):
            logits = self.policy(seq_flat, attention_mask=attn_mask).logits


        logp_all = F.log_softmax(logits.to(torch.float16), dim=-1).to(torch.float16)     # (BG,T_tot,V)
        targets   = seq_flat[:, 1:]                                       # (BG,T_tot-1)


        logp_tok  = logp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        new_logp  = logp_tok[:, -T_g:].view(B, G, T_g)          # last T_g tokens


        old_logp  = rollouts.logprobs                          # (B,G,T_g)
        gen_mask  = (rollouts.gen_ids != pad_id).float()       # (B,G,T_g)

        ratios = torch.exp((new_logp - old_logp).clamp(-80, 80)) * gen_mask
        log_r   = (new_logp - old_logp) * gen_mask # for metrics

        # -------------- save full ratio list -------------
        if sync_grads and self._ratio_log_path is not None:
            flat = ratios[gen_mask.bool()].detach().cpu().tolist()
            rec  = {"step": self.actual_opt_step, "ratios": flat}
            with self._ratio_log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")


        # Compute entropy for logging
        probs_gen = torch.exp(logp_all[:, -T_g:])               # only gen slice
        H_gen     = -(probs_gen * logp_all[:, -T_g:]).sum(-1)  # (BG,T_g)
        ent_tok   = H_gen.view(B, G, T_g) * gen_mask
        entropy   = ent_tok.sum() / (gen_mask.sum() + 1e-8)


        # To-Do:
        # Add variations here related to clipping

        clip_p = self.cfg.get('clip_+', self.cfg['clip_eps'])
        clip_m = self.cfg['clip_eps']

        surr1 = ratios * adv.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1 - clip_m,
                                    1 + clip_p) * adv.unsqueeze(-1)
        token_loss = -torch.min(surr1, surr2) * gen_mask                 # (B,G,T_g)

        # ---- per-prompt normalisation (Dr-GRPO) -------------
        tokens_per_gen = gen_mask.sum(dim=2)             # (B,G)
        max_lens, _    = torch.max(tokens_per_gen, dim=-1)  # (B)
        loss_per_prompt = token_loss.sum(dim=(1,2)) / (G * max_lens + 1e-8)
        loss            = loss_per_prompt.mean() * scale


        # ratio statistics over *non-pad* tokens only
        clip = 8.0
        mask = gen_mask.bool()                          # (B,G,T_g) True for real tokens

        flat_ratios = ratios[mask]                      # 1D tensor of all valid ratios
        if flat_ratios.numel() > 0:
            ratio_mean_val   = flat_ratios.mean().item()
            ratio_median_val = flat_ratios.median().item()
            ratio_p90_val    = flat_ratios.quantile(0.90).item()
            ratio_p99_val    = flat_ratios.quantile(0.99).item()
            ratio_max_val    = flat_ratios.max().item()
        else:
            # no valid tokens? unlikely, but safe fallback
            ratio_mean_val = ratio_median_val = ratio_p90_val = ratio_p99_val = ratio_max_val = 0.0

        # clip-fraction and log-ratio std also over real tokens
        flat_logr = log_r[mask]
        if flat_logr.numel() > 0:
            ratio_clip_frac_val = (flat_logr.abs() > clip).float().mean().item()
            logr_std_val        = flat_logr.std().item()
        else:
            ratio_clip_frac_val = logr_std_val = 0.0


        entropy_val = entropy.item()

        # ---- free big policy tensors before ref forward ----
        del logits, logp_all, probs_gen, H_gen
        torch.cuda.empty_cache()


        
        # ── metric-only KL divergence ─────────────────────────────
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.cfg["bf16"]):
            ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
        ref_logp_all  = F.log_softmax(ref_logits.to(torch.float16), dim=-1)
        ref_logp_tok  = ref_logp_all[..., :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ref_logp      = ref_logp_tok[:, -T_g:].view(B, G, T_g)

        delta_lp      = new_logp - ref_logp                      # shape (B,G,T_g)
        kl_per_tok    = (delta_lp.exp() + delta_lp - 1.0) * gen_mask
        kl_mean       = (kl_per_tok.sum() / (gen_mask.sum() + 1e-8)).item()
        # ──────────────────────────────────────────────────────────
        

        del seq_flat, attn_mask, targets
        del ent_tok
        del ref_logits, ref_logp_all, ref_logp_tok, ref_logp, kl_per_tok
        torch.cuda.empty_cache()


        # Optimization  with gradient checkpointing


        maybe = (
            self.policy.no_sync
            if (hasattr(self.policy, "no_sync") and not sync_grads)
            else nullcontext
        )
        with maybe():
            loss.backward()

        if sync_grads:                          # ← step *only* when caller says so
            clip_grad_norm_(self.policy.parameters(), self.cfg["grad_clip"])
            self.opt.step()
            self.lr_sched.step()
            self.opt.zero_grad(set_to_none=True)
            self.actual_opt_step += 1
            #print(f"Actual Opt Steps = {self.actual_opt_step}")

        loss_val  = loss.detach().float().item()

        # To-Do:
        # More metrics
        return {
            "loss"            : loss_val,
            "entropy"         : entropy_val,
            "kl"              : kl_mean,
            "ratio_mean"      : ratio_mean_val,
            "ratio_median"    : ratio_median_val,
            "ratio_p90"       : ratio_p90_val,
            "ratio_p99"       : ratio_p99_val,
            "ratio_max"       : ratio_max_val,
            "ratio_clip_frac" : ratio_clip_frac_val,
            "logr_std"        : logr_std_val,
            "r_mean"          : rollouts.reward.mean(dim=(0,1)).item(),
            "tag_correct"     : rollouts.tag_correct.float().mean(dim=(0, 1)).item(),
            "think_len"       : rollouts.think_len.float().mean(dim=(0,1)).item(),
        }

