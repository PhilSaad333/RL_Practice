# rl_training/algs/dr_grpo.py
from __future__ import annotations

import json
import pathlib
from contextlib import nullcontext
from typing import Tuple, Dict

import numpy as np                        # Added 8/11
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

        # === Entropy-probe accumulators (per accumulation window) ===  # Added 8/11
        self._probe_GH1 = None     # dict[param] = sum S * grad S
        self._probe_GA1 = None     # dict[param] = sum SA * grad S
        self._probe_G1  = None     # dict[param] = sum grad S
        self._probe_sumS  = 0.0    # scalar sum S over sequences
        self._probe_sumSA = 0.0    # scalar sum S*A over sequences
        self._probe_N     = 0      # count sequences
        self._probe_S_list = []    # list of per-seq S for saving
        self._probe_A_list = []    # list of per-seq A for saving
        self._probe_tokent_list = []  # mean token entropy per seq

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

        # ── Entropy probe: accumulate per microbatch, then finalize on step ──  # Added 8/11
        probe_cfg = self.cfg.get("entropy_probe", {})
        do_probe = int(probe_cfg.get("every", 0)) > 0
        if do_probe and self._is_main_process():
            self._probe_accumulate_microbatch(rollouts)             # Added 8/11
            if sync_grads:
                # only log/save every N steps
                every = int(probe_cfg.get("every", 0))
                if every > 0 and (self.actual_opt_step % every == 0):
                    try:
                        self._probe_finalize_and_log(probe_cfg)     # Added 8/11
                    except Exception as e:
                        print(f"[entropy_probe] skipped due to error: {e}")
                else:
                    self._probe_reset_window()                      # Added 8/11

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

    # === Entropy-probe helpers (accumulate & finalize) ========================  # Added 8/11
    def _is_main_process(self) -> bool:  # Added 8/11
        import torch.distributed as dist
        return (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _run_dir_from_ratio_path(self) -> pathlib.Path:  # Added 8/11
        if getattr(self, "_ratio_log_path", None) is not None:
            return pathlib.Path(self._ratio_log_path).parent
        return pathlib.Path(".")

    def _probe_reset_window(self) -> None:  # Added 8/11
        self._probe_GH1 = None
        self._probe_GA1 = None
        self._probe_G1  = None
        self._probe_sumS = 0.0
        self._probe_sumSA = 0.0
        self._probe_N = 0
        self._probe_S_list.clear()
        self._probe_A_list.clear()
        self._probe_tokent_list.clear()

    def _trainable_params(self):  # Added 8/11
        return [p for p in self.policy.parameters() if isinstance(p, torch.nn.Parameter) and p.requires_grad]

    def _probe_accumulate_microbatch(self, rollouts: RolloutBatch) -> None:  # Added 8/11
        """
        Re-forward on this microbatch, compute:
          GH1 += Σ S ∇S,  GA1 += Σ SA ∇S,  G1 += Σ ∇S
        and accumulate S, A, token-entropy means for logging.
        """
        # (Re)build sequences
        seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences(rollouts)
        B, G, T_g = rollouts.gen_ids.shape

        # Forward fresh (don’t rely on training graph)
        with torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            logits = self.policy(seq_flat, attention_mask=attn_mask).logits
        logits = logits / self.cfg.get("temperature", 1.0)
        logp_all = F.log_softmax(logits.float(), dim=-1)
        new_logp = logp_all[:, :-1].gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g:]
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
        new_logp = new_logp.view(B, G, T_g)

        # Per-seq S and A
        S_seq = (new_logp * gen_mask).sum(dim=-1)            # (B,G)
        A_seq = self._compute_advantage(rollouts.reward)     # (B,G)
        tok_ent_mean = (-(new_logp) * gen_mask).sum(-1) / (gen_mask.sum(-1) + 1e-8)

        # Flatten for scalar objectives
        S = S_seq.reshape(-1)                                # (N,)
        A = A_seq.reshape(-1)                                # (N,)
        N = S.numel()

        # Scalar objectives whose grads give the sums we need:
        #  GH1 = Σ S ∇S  via grad of 0.5 * Σ S^2
        #  GA1 = Σ SA ∇S via grad of 0.5 * Σ (A * S^2)
        #  G1  = Σ ∇S    via grad of Σ S
        params = self._trainable_params()
        L_GH1 = 0.5 * (S.pow(2).sum())
        L_GA1 = 0.5 * ((A * S.pow(2)).sum())
        L_G1  = S.sum()

        # Changed 8/11: keep graph for first two grad calls
        grads_GH1 = torch.autograd.grad(
            L_GH1, params, retain_graph=True,  create_graph=False, allow_unused=True
        )
        grads_GA1 = torch.autograd.grad(
            L_GA1, params, retain_graph=True,  create_graph=False, allow_unused=True
        )
        grads_G1  = torch.autograd.grad(
            L_G1,  params, retain_graph=False, create_graph=False, allow_unused=True
        )

        # Lazy-init accum dicts
        if self._probe_GH1 is None:
            self._probe_GH1 = {}
            self._probe_GA1 = {}
            self._probe_G1  = {}
            for p in params:
                shape = p.shape
                dev   = p.device
                self._probe_GH1[p] = torch.zeros(shape, device=dev, dtype=torch.float32)
                self._probe_GA1[p] = torch.zeros(shape, device=dev, dtype=torch.float32)
                self._probe_G1[p]  = torch.zeros(shape, device=dev, dtype=torch.float32)

        # Accumulate (cast to fp32 for numeric stability)
        for p, gH1, gA1, g1 in zip(params, grads_GH1, grads_GA1, grads_G1):
            if gH1 is not None:
                self._probe_GH1[p] += gH1.detach().to(torch.float32)
            if gA1 is not None:
                self._probe_GA1[p] += gA1.detach().to(torch.float32)
            if g1 is not None:
                self._probe_G1[p]  += g1.detach().to(torch.float32)

        # Scalar sums and per-seq arrays for logging
        self._probe_sumS  += float(S.sum().item())
        self._probe_sumSA += float((S * A).sum().item())
        self._probe_N     += int(N)
        self._probe_S_list.append(S_seq.detach().cpu())
        self._probe_A_list.append(A_seq.detach().cpu())
        self._probe_tokent_list.append(tok_ent_mean.detach().cpu())

    def _probe_finalize_and_log(self, probe_cfg: dict) -> None:  # Added 8/11
        """On the syncing microbatch, turn accumulators into centered g_H, g_A,
        compute g_H^T P g_A from Adam state, save artifacts, then reset window.
        """
        if self._probe_GH1 is None:  # nothing accumulated
            return

        meanS  = self._probe_sumS  / max(self._probe_N, 1)
        meanSA = self._probe_sumSA / max(self._probe_N, 1)

        dot = 0.0
        normH = 0.0
        normA = 0.0
        lr_vals = []

        # Centered vectors using the three sums:
        # g_H = GH1 - meanS * G1,  g_A = GA1 - meanSA * G1
        for group in self.opt.param_groups:
            eps = float(group.get("eps", 1e-8))
            lr_vals.append(float(group.get("lr", 0.0)))
            for p in group["params"]:
                if p not in self._probe_GH1:
                    continue
                gH = self._probe_GH1[p] - meanS  * self._probe_G1[p]
                gA = self._probe_GA1[p] - meanSA * self._probe_G1[p]

                state = self.opt.state.get(p, {})
                v = state.get("exp_avg_sq", None)   # Adam second moment buffer  # Added 8/11
                if v is None:
                    inv = 1.0 / eps
                    dot   += (gH * gA).sum().item() * inv
                    normH += (gH * gH).sum().item() * inv
                    normA += (gA * gA).sum().item() * inv
                else:
                    denom = v.sqrt().to(gH.dtype) + eps
                    dot   += (gH * gA / denom).sum().item()
                    normH += (gH * gH / denom).sum().item()
                    normA += (gA * gA / denom).sum().item()

        avg_lr = float(np.mean(lr_vals)) if lr_vals else 0.0
        deltaH_pred = -(avg_lr * dot)

        # Save artifacts (only every N steps; we’re already in the gated path)
        run_dir = self._run_dir_from_ratio_path()
        out_root = run_dir / probe_cfg.get("out_dir", "entropy_probes")
        step_dir = out_root / f"step_{self.actual_opt_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save per-seq arrays
        S_all  = torch.cat(self._probe_S_list, dim=0).numpy().astype(np.float32)
        A_all  = torch.cat(self._probe_A_list, dim=0).numpy().astype(np.float32)
        Et_all = torch.cat(self._probe_tokent_list, dim=0).numpy().astype(np.float32)

        np.save(step_dir / "S.npy",  S_all)
        np.save(step_dir / "A.npy",  A_all)
        np.save(step_dir / "entropy_tok_mean.npy", Et_all)

        # Meta and scalars
        meta = {
            "step": int(self.actual_opt_step),
            "N": int(self._probe_N),
            "avg_lr": avg_lr,
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        pred = {
            "gH_P_gA": float(dot),
            "norm_gH_P": float(np.sqrt(max(normH, 0.0))),
            "norm_gA_P": float(np.sqrt(max(normA, 0.0))),
            "deltaH_first_order_adam": float(deltaH_pred),
        }
        (step_dir / "deltaH_pred.json").write_text(json.dumps(pred, indent=2))

        # Reset accumulators for next window
        self._probe_reset_window()

    # Placeholder left here so external calls don’t break older imports  # Added 8/11
    def _log_entropy_probes(self, *args, **kwargs) -> None:
        return  # no-op
