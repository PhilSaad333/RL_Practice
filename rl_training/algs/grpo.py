# rl_training/algs/grpo.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from .base import RLAlgorithm, RolloutBatch
from contextlib import nullcontext

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

    def step(self, rollouts: RolloutBatch, ref_model, *, sync_grads: bool = True) -> dict[str, float]:
        B, G, T_g = rollouts.gen_ids.shape
        device    = rollouts.gen_ids.device
        pad_id    = self.pad_id
        scale      = 1.0 / self.accum_steps

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

        plen   = (rollouts.prompt_ids != pad_id).sum(-1)                 # (B)
        plen   = plen[:, None].expand(-1, G).reshape(-1)                 # (BG)
        new_lp_list = []
        for i in range(B * G):
            p = plen[i].item()
            gen_len = min(T_g, logp_tok.size(1) - p)
            lp = logp_tok[i, p : p + gen_len]
            if gen_len < T_g:                        # right-pad with zeros
                lp = F.pad(lp, (0, T_g - gen_len), value=0.0)
            new_lp_list.append(lp)
        new_logp = torch.stack(new_lp_list).view(B, G, T_g)

        old_logp   = rollouts.logprobs                                   # (B,G,T_g)
        ratios     = torch.exp(new_logp - old_logp)                      # (B,G,T_g)
        gen_mask   = (rollouts.gen_ids != pad_id).float()                # (B,G,T_g)

        # To-Do:
        # Add variations here related to clipping
        # Log ratios for studying statistics

        surr1 = ratios * adv.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1 - self.cfg["clip_eps"],
                                    1 + self.cfg["clip_eps"]) * adv.unsqueeze(-1)
        ppo_loss = -torch.min(surr1, surr2) * gen_mask                 # (B,G,T_g)

        if getattr(self.cfg, "kl_beta", 0.0) > 0:
            with torch.no_grad():
                ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
            ref_logp_all = F.log_softmax(ref_logits, -1)
            ref_logp_tok = ref_logp_all[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            ref_lp_list = []
            for i in range(B * G):
                p = plen[i].item()
                gen_len = min(T_g, ref_logp_tok.size(1) - p)
                lp = ref_logp_tok[i, p : p + gen_len]
                if gen_len < T_g:                        # right-pad with zeros
                    lp = F.pad(lp, (0, T_g - gen_len), value=0.0)
                ref_lp_list.append(lp)
            ref_logp = torch.stack(ref_lp_list).view(B, G, T_g)
            delta_lp = new_logp - ref_logp
            kl_per_tok = torch.exp(delta_lp) + delta_lp - torch.ones(B,G,T_g)
            kl_per_tok = kl_per_tok * gen_mask * self.cfg["kl_beta"]
            token_loss = ppo_loss - kl_per_tok
        else:
            kl_per_tok = torch.zeros(B,G,T_g).to(device)
            token_loss = ppo_loss


        # per-prompt normalisation
        tokens_per_prompt = gen_mask.sum(dim=(1,2))                       # (B)
        loss_per_prompt   = token_loss.sum(dim=(1,2)) / (tokens_per_prompt + 1e-8)
        kl_per_prompt     = kl_per_tok.sum(dim=(1,2)) / (tokens_per_prompt + 1e-8)
        loss              = (loss_per_prompt.mean() * scale)
        kl_term           = kl_per_prompt.mean()


        # Compute entropy for logging
        # used to use idx earlier, just copied it here. maybe revist this
        idx        = plen[:, None] + torch.arange(T_g, device=device)
        probs      = torch.exp(logp_all)
        H_all      = -(probs * logp_all).sum(-1)
        ent_tok    = H_all.gather(1, idx).view(B, G, T_g)
        entropy    = (ent_tok * gen_mask).sum() / (gen_mask.sum() + 1e-8)


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


        # To-Do:
        # More metrics
        return {
            "loss"        : loss.item(),
            "entropy"     : entropy.item(),
            "kl"          : kl_term.item(),
            "ratio_mean"  : ratios.mean().item(),
            "r_mean": rollouts.reward.mean(dim=(0,1)).item(),
            "tag_correct" : rollouts.tag_correct.float().mean(dim=(0, 1)).item(),
            "think_len"   : rollouts.think_len.float().mean(dim=(0,1)).item(),
        }
