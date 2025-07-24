# rl_training/algs/grpo.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from transformers import get_cosine_schedule_with_warmup
from .base import RLAlgorithm, RolloutBatch



class GRPO(RLAlgorithm):
    def __init__(self, policy, cfg, *, pad_id: int | None = None):
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

        # for debug
        self.actual_op_step = 0

    def step(self, rollouts: RolloutBatch, ref_model, *, sync_grads: bool = True) -> dict[str, float]:
        B, G, T_g = rollouts.gen_ids.shape
        device    = rollouts.gen_ids.device
        self.device = device
        pad_id    = self.pad_id
        scale      = 1.0 / self.accum_steps

        # compute GRPO advantage from rewards
        mean_r = rollouts.reward.mean(dim=1, keepdim=True)
        std_r  = rollouts.reward.std (dim=1, keepdim=True) + 1e-8
        adv    = (rollouts.reward - mean_r) / std_r                       # (B,G)

        prompt_rep = rollouts.prompt_ids.unsqueeze(1).expand(-1, G, -1)
        seq_ids    = torch.cat((prompt_rep, rollouts.gen_ids), dim=-1)    # (B,G,T_tot)
        seq_flat   = seq_ids.reshape(B * G, -1)                           # (BG,T_tot)

        attn_mask  = (seq_flat != pad_id).long()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.cfg["bf16"]):
            logits = self.policy(seq_flat, attention_mask=attn_mask).logits

        logp_all  = F.log_softmax(logits, dim=-1)                         # (BG,T_tot,V)
        targets   = seq_flat[:, 1:]                                       # (BG,T_tot-1)


        logp_tok  = logp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        new_logp  = logp_tok[:, -T_g:].view(B, G, T_g)          # last T_g tokens

        old_logp  = rollouts.logprobs                          # (B,G,T_g)
        gen_mask  = (rollouts.gen_ids != pad_id).float()       # (B,G,T_g)

        ratios = torch.exp((new_logp - old_logp).clamp(-80, 80)) * gen_mask

        # -------------- KL term ----------------
        if self.cfg["kl_beta"] > 0:
            with torch.no_grad():
                ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
            ref_lp_all = F.log_softmax(ref_logits, -1)
            ref_lp_tok = ref_lp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            ref_logp   = ref_lp_tok[:, -T_g:].view(B, G, T_g)            # aligned slice
            delta_lp   = new_logp - ref_logp
            kl_per_tok = (torch.exp(delta_lp) + delta_lp - 1.0) * gen_mask * self.cfg["kl_beta"]
        else:
            kl_per_tok = torch.zeros_like(new_logp)

        # Compute entropy for logging
        probs_all = torch.exp(logp_all)
        H_all     = -(probs_all * logp_all).sum(-1)            # (BG,T_tot)
        ent_tok   = H_all[:, -T_g:].view(B, G, T_g) * gen_mask
        entropy   = ent_tok.sum() / (gen_mask.sum() + 1e-8)


        # To-Do:
        # Add variations here related to clipping
        # Log ratios for studying statistics

        surr1 = ratios * adv.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1 - self.cfg["clip_eps"],
                                    1 + self.cfg["clip_eps"]) * adv.unsqueeze(-1)
        ppo_loss = -torch.min(surr1, surr2) * gen_mask                 # (B,G,T_g)

        # add kl term per token
        kl_per_tok = kl_per_tok * gen_mask
        token_loss = ppo_loss - kl_per_tok


        # per-prompt normalisation
        tokens_per_prompt = gen_mask.sum(dim=(1,2))                       # (B)
        loss_per_prompt   = token_loss.sum(dim=(1,2)) / (tokens_per_prompt + 1e-8)
        kl_per_prompt     = kl_per_tok.sum(dim=(1,2)) / (tokens_per_prompt + 1e-8)
        loss              = (loss_per_prompt.mean() * scale)
        kl_term           = kl_per_prompt.mean()


        del logits, logp_all
        if self.cfg["kl_beta"] > 0:
            del ref_logits, ref_lp_all
        torch.cuda.empty_cache()


        # Optimization  with gradient checkpointing
        maybe = self.policy.no_sync if (hasattr(self.policy, "no_sync") and not sync_grads) else nullcontext
        with maybe():
            loss.backward()

        self._accum_ctr += 1
        if self._accum_ctr % self.accum_steps == 0:
            clip_grad_norm_(self.policy.parameters(), self.cfg["grad_clip"])
            self.opt.step()
            self.lr_sched.step()
            self.opt.zero_grad(set_to_none=True)
            self.actual_opt_step += 1
            print(f"Actual Opt Steps = {self.actual_opt_step}")

        loss_val  = loss.detach().float().item()
        kl_val    = kl_term.detach().float().item()

        # To-Do:
        # More metrics
        return {
            "loss"        : loss_val,
            "entropy"     : entropy.item(),
            "kl"          : kl_val,
            "ratio_mean"  : ratios.mean().item(),
            "r_mean": rollouts.reward.mean(dim=(0,1)).item(),
            "tag_correct" : rollouts.tag_correct.float().mean(dim=(0, 1)).item(),
            "think_len"   : rollouts.think_len.float().mean(dim=(0,1)).item(),
        }

