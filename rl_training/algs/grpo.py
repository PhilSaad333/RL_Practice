# rl_training/algs/grpo.py
import torch
from .base import RLAlgorithm, RolloutBatch

class GRPO(RLAlgorithm):
    def __init__(self, policy, cfg):
        super().__init__(policy, cfg)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    def step(self, rollouts: RolloutBatch):

        # rollouts.reward has shape (B,G)
        b = rollouts.reward.mean(dim=1, keepdim=True)      # (B,1)
        std = rollouts.reward.std(dim=1, keepdim=True)     # (B,1) 
        adv = (rollouts.reward - b)/(std + 1e-8)           # (B,G)


        # log-probs for the generated tokens
        B, G, T = rollouts.gen_ids.shape
        ids = torch.cat((rollouts.prompt_ids.unsqueeze(1).expand(-1,G,-1), rollouts.gen_ids), dim=-1)
        ids = ids.view(B*G, -1)
        with torch.no_grad():
            logits = self.policy(ids).logits[:, :-1]

        logp = torch.log_softmax(logits, -1)
        tgt = ids[:, 1:].unsqueeze(-1)
        lp_token = logp.gather(-1, tgt).squeeze(-1)         # [B*G, T_total-1]
        lp_seq = lp_token.sum(dim=-1).view(B, G)            # [B,G]








        loss = -(adv * lp_seq).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return {"loss": loss.item(), "adv_mean": adv.mean().item()}
