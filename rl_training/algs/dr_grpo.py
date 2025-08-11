# rl_training/algs/dr_grpo.py
from __future__ import annotations

import json
import pathlib
from contextlib import nullcontext
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from .base import RLAlgorithm, RolloutBatch


class DRGRPO(RLAlgorithm):
    """
    Clear, modular PPO/GRPO-style updater with optional differentiable KL.
    """

    def __init__(
        self,
        policy,
        cfg: dict,
        *,
        pad_id: int | None = None,
        ratio_log_path: str | pathlib.Path | None = None,
    ):
        super().__init__(policy, cfg)
        self.cfg = cfg

        total_updates = cfg["total_steps"]
        warmup_steps = int(0.05 * total_updates)

        self.opt = torch.optim.AdamW(
            policy.parameters(),
            lr=cfg["lr"],
            weight_decay=0.01,
        )

        match cfg.get("lr_scheduler", "none"):
            case "cosine":
                self.lr_sched = get_cosine_schedule_with_warmup(
                    self.opt,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_updates,
                )
            case "const" | "constant":
                self.lr_sched = get_constant_schedule_with_warmup(
                    self.opt, num_warmup_steps=warmup_steps
                )
            case _:
                self.lr_sched = None

        self.pad_id = pad_id if pad_id is not None else getattr(policy.config, "pad_token_id", 0)
        self.accum_steps: int = cfg["grad_accum_steps"]
        self._accum_ctr: int = 0
        self.actual_opt_step: int = 0
        self.device = None

        self._ratio_log_path = pathlib.Path(ratio_log_path) if ratio_log_path else None

        # Always compute sampled entropy (cheap and correct estimator)  # Changed 8/11
        self.compute_entropy = True  # Changed 8/11

    @torch.no_grad()
    def _compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return centred (but *not* normalised) advantage for Dr-GRPO."""
        mean_r = rewards.mean(dim=1, keepdim=True)
        return rewards - mean_r  # (B, G)

    def step(
        self,
        rollouts: RolloutBatch,
        ref_model,
        *,
        sync_grads: bool = True,
    ) -> Dict[str, float]:
        """
        One optimisation step (or gradient-accumulation micro-step).
        """
        self.device = rollouts.gen_ids.device
        B, G, T_g = rollouts.gen_ids.shape

        # 1) prepare tensors & advantages
        seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences(rollouts)
        adv = self._compute_advantage(rollouts.reward)  # (B, G)

        # 2) forward & gather log-probs
        new_logp = self._policy_logp(seq_flat, attn_mask, targets_tok, B, G, T_g).view(B, G, T_g)
        old_logp = rollouts.logprobs  # (B, G, T_g)

        if torch.isinf(old_logp).any() or torch.isnan(old_logp).any():
            bad = torch.nonzero(~torch.isfinite(old_logp))
            print("‼️ old_logp contains non-finite values at", bad[:5], "…")
        if torch.isinf(new_logp).any() or torch.isnan(new_logp).any():
            bad = torch.nonzero(~torch.isfinite(new_logp))
            print("‼️ new_logp contains non-finite values at", bad[:5], "…")

        # 3) PPO clipped loss & entropy (entropy from sampled tokens)
        ratios, token_loss = self._ppo_surrogate(new_logp, old_logp, adv, gen_mask)

        # sampled entropy per token = -log p(sampled token)
        entropy = (-(new_logp) * gen_mask).sum() / (gen_mask.sum() + 1e-8)  # Changed 8/11
        self._last_entropy = (-(new_logp)).detach()                          # Changed 8/11

        loss = token_loss * (1.0 / self.accum_steps)

        # 4) optional differentiable KL (stable on-policy estimator)
        kl_mean = 0.0
        if self.cfg.get("kl_beta", 0.0) > 0.0:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
                ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
            ref_logits = ref_logits / self.cfg.get("temperature", 1.0)

            # log-softmax in float32 for stability
            ref_logp_all = F.log_softmax(ref_logits.float(), dim=-1)[..., :-1]
            ref_logp = ref_logp_all.gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[..., -T_g:].view(B, G, T_g)

            # KL(pi||ref) sample estimator = E_pi[logpi - logref]
            kl_tok = ((new_logp - ref_logp) * gen_mask)
            kl_mean = kl_tok.sum() / (gen_mask.sum() + 1e-8)

            loss = loss + self.cfg["kl_beta"] * kl_mean
        else:
            kl_mean = self._kl_metric_only(seq_flat, attn_mask, targets_tok, new_logp, ref_model, gen_mask, T_g)

        # 5) optional ratio logging
        self._maybe_log_ratios(ratios, gen_mask)

        # 6) optimise
        self._backward_and_step(loss, sync_grads)

        # 7) metrics
        metrics: Dict[str, float] = {
            "loss": float(loss.detach().item()),
            "entropy": float(entropy.item()),
            "kl": float(kl_mean if isinstance(kl_mean, float) else kl_mean.item()),
            "r_mean": float(rollouts.reward.mean(dim=(0, 1)).item()),
            "tag_correct": float(rollouts.tag_correct.float().mean(dim=(0, 1)).item()),
            "think_len": float(rollouts.think_len.float().mean(dim=(0, 1)).item()),
        }
        metrics.update(self._ratio_stats(ratios, gen_mask))

        # Added 8/11: placeholder for entropy-formula probes (no-op for now)
        # self._log_entropy_probes(new_logp, adv, gen_mask)  # Added 8/11

        return metrics

    # -- sequence prep ---------------------------------------------------------
    def _build_sequences(self, rollouts: RolloutBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, G, T_g = rollouts.gen_ids.shape
        prompt_rep = rollouts.prompt_ids.unsqueeze(1).expand(-1, G, -1)
        seq_ids = torch.cat((prompt_rep, rollouts.gen_ids), dim=-1)  # (B, G, T_total)

        seq_flat = seq_ids.reshape(B * G, -1)
        attn_mask = (seq_flat != self.pad_id).long()
        targets_tok = seq_flat[:, 1:]  # teacher forcing targets
        gen_mask = (rollouts.gen_ids != self.pad_id).float()  # (B, G, T_g)
        return seq_flat, attn_mask, targets_tok, gen_mask

    # -- forward passes --------------------------------------------------------
    def _policy_logp(
        self,
        seq_flat: torch.Tensor,
        attn_mask: torch.Tensor,
        targets_tok: torch.Tensor,
        B: int,
        G: int,
        T_g: int,
    ) -> torch.Tensor:
        """Return log p(sampled token) for generated slice (BG, T_g)."""
        with torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            logits = self.policy(seq_flat, attention_mask=attn_mask).logits

        logits = logits / self.cfg.get("temperature", 1.0)

        # compute log-softmax in float32 for stability, then gather
        logp_all = F.log_softmax(logits.float(), dim=-1)
        # Gather next-token logprob, then slice the generated region
        new_logp = logp_all[:, :-1].gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g:]
        # clamp and sanitize
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0)
        new_logp = torch.clamp(new_logp, min=-80.0, max=0.0)  # logprob <= 0
        return new_logp

    # -- PPO loss --------------------------------------------------------------
    def _ppo_surrogate(
        self,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (ratios, loss_scalar)."""
        ratios = torch.exp((new_logp - old_logp).clamp(-80, 80)) * gen_mask

        clip_eps_neg = self.cfg["clip_eps"]
        clip_eps_pos = self.cfg.get("clip_+", self.cfg["clip_eps"])

        surr1 = ratios * adv.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1 - clip_eps_neg, 1 + clip_eps_pos) * adv.unsqueeze(-1)
        token_loss = -torch.min(surr1, surr2) * gen_mask  # (B, G, T_g)

        # Dr-GRPO: normalise per-prompt by max gen length
        tokens_per_gen = gen_mask.sum(dim=2)  # (B, G)
        max_lens = tokens_per_gen.max(dim=-1).values  # (B,)
        loss_per_prompt = token_loss.sum(dim=(1, 2)) / (gen_mask.shape[1] * max_lens + 1e-8)
        loss = loss_per_prompt.mean()
        return ratios, loss

    # -- KL helpers ------------------------------------------------------------
    def _kl_metric_only(
        self,
        seq_flat: torch.Tensor,
        attn_mask: torch.Tensor,
        targets_tok: torch.Tensor,
        new_logp: torch.Tensor,
        ref_model,
        gen_mask: torch.Tensor,
        T_g: int,
    ) -> float:
        """KL for logging (no grads): sample estimator E_pi[logpi - logref]."""
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
        ref_logits = ref_logits / self.cfg.get("temperature", 1.0)

        ref_logp_all = F.log_softmax(ref_logits.float(), dim=-1)[..., :-1]
        ref_logp = ref_logp_all.gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[..., -T_g:]
        ref_logp = ref_logp.view_as(new_logp)

        kl_tok = ((new_logp - ref_logp) * gen_mask)
        return float((kl_tok.sum() / (gen_mask.sum() + 1e-8)).item())

    # -- optimisation ----------------------------------------------------------
    def _backward_and_step(self, loss: torch.Tensor, sync_grads: bool) -> None:
        """Handle backward, gradient clipping, optimiser & scheduler."""
        maybe = (self.policy.no_sync if (hasattr(self.policy, "no_sync") and not sync_grads) else nullcontext)
        with maybe():
            loss.backward()

        if sync_grads:
            clip_grad_norm_(self.policy.parameters(), self.cfg["grad_clip"])
            self.opt.step()
            if self.lr_sched is not None:
                self.lr_sched.step()
            self.opt.zero_grad(set_to_none=True)
            self.actual_opt_step += 1

    # -- ratio logging & stats -------------------------------------------------
    def _maybe_log_ratios(self, ratios: torch.Tensor, gen_mask: torch.Tensor) -> None:
        if self._ratio_log_path is None:
            return
        flat = ratios[gen_mask.bool()].detach().cpu().tolist()
        rec = {"step": self.actual_opt_step, "ratios": flat}
        with self._ratio_log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _ratio_stats(self, ratios: torch.Tensor, gen_mask: torch.Tensor) -> Dict[str, float]:
        mask = gen_mask.bool()
        flat_ratios = ratios[mask]
        flat_logr = torch.log(ratios.clamp_min(1e-8))[mask]
        if flat_ratios.numel() == 0:
            return {k: 0.0 for k in ("ratio_mean","ratio_median","ratio_p90","ratio_p99","ratio_max","ratio_clip_frac","logr_std")}
        stats = {
            "ratio_mean": flat_ratios.mean().item(),
            "ratio_median": flat_ratios.median().item(),
            "ratio_p90": flat_ratios.quantile(0.9).item(),
            "ratio_p99": flat_ratios.quantile(0.99).item(),
            "ratio_max": flat_ratios.max().item(),
            "ratio_clip_frac": (flat_logr.abs() > 8.0).float().mean().item(),
            "logr_std": flat_logr.std().item(),
        }
        return stats

    # Placeholder for entropy-kernel study (wired later)  # Added 8/11
    def _log_entropy_probes(self, new_logp: torch.Tensor, adv: torch.Tensor, gen_mask: torch.Tensor) -> None:
        """(Added 8/11) Hook to compute/store S, SA etc. for later kernel analysis."""
        pass
