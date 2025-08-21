# rl_training/algs/dr_grpo.py
"""
Dr-GRPO (Distributional Rejection GRPO) Algorithm Implementation

This file contains:
1. Core Dr-GRPO training logic
2. Integrated diagnostic probes (GNS, Entropy)

TODO: After testing, refactor probes into separate modules
"""
from __future__ import annotations

import json
import pathlib
from contextlib import nullcontext
from typing import Tuple, Dict

import numpy as np                        # Added 8/11
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from .base import RLAlgorithm, RolloutBatch
from .gns_probe import GNSProbe

# ============================================================================
# UTILITIES
# ============================================================================

def _unwrap(model):
    """Helper for DDP/unwrapped access"""
    return model.module if hasattr(model, "module") else model


# ============================================================================
# MAIN DR-GRPO ALGORITHM CLASS
# ============================================================================

class DRGRPO(RLAlgorithm):
    """
    Clear, modular PPO/GRPO-style updater with optional differentiable KL.
    """

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------
    
    def __init__(
        self,
        policy,
        cfg: dict,
        *,
        pad_id: int | None = None,
        ratio_log_path: str | pathlib.Path | None = None,
        grad_accum_steps: int | None = None,
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
        self.accum_steps: int = grad_accum_steps if grad_accum_steps is not None else cfg.get("grad_accum_steps", 1)
        self._accum_ctr: int = 0
        self.actual_opt_step: int = 0
        self.device = None

        self._ratio_log_path = pathlib.Path(ratio_log_path) if ratio_log_path else None

        # Always compute sampled entropy (cheap and correct estimator)  # Changed 8/11
        self.compute_entropy = True  # Changed 8/11

        # ====================================================================
        # PROBE INITIALIZATION SECTION
        # ====================================================================
        
        # --- Entropy Probe Fields ---  # Added 8/11
        self._probe_sumWS = 0.0   # Σ w S
        self._probe_sumW  = 0.0   # Σ w
        self._probe_GH1 = None     # dict[param] = sum S * grad S
        self._probe_GA1 = None     # dict[param] = sum A * grad S
        self._probe_G1  = None     # dict[param] = sum grad S
        self._probe_sumS  = 0.0    # scalar sum S over sequences
        self._probe_N     = 0      # count sequences
        self._probe_S_list = []    # list of per-seq S for saving
        self._probe_A_list = []    # list of per-seq A for saving
        self._probe_tokent_list = []  # mean token entropy per seq
        
        # --- GNS (Gradient Noise Scale) Probe ---
        gns_config = cfg.get("gns_probe", {})
        self.gns_probe = GNSProbe(
            window_size=gns_config.get("window_size", 8),
            ema_alpha=gns_config.get("ema_alpha", 0.1),
            enabled=gns_config.get("enabled", False),
            debug=gns_config.get("debug", False)
        )
        self._last_gns_metrics: Dict[str, float] = {}  # Store GNS metrics for logging
        
        # --- Entropy Probes ---
        # Simple entropy probe (lightweight, for regular training)
        from .simple_entropy_probe import SimpleEntropyProbe
        simple_entropy_config = cfg.get("simple_entropy_probe", {})
        self.simple_entropy_probe = SimpleEntropyProbe(
            enabled=simple_entropy_config.get("enabled", False),
            debug=simple_entropy_config.get("debug", False),
            preconditioning_mode=simple_entropy_config.get("preconditioning_mode", "previous_step"),
            log_every=simple_entropy_config.get("log_every", 1)
        )
        
        # Complex entropy probe (expensive, for detailed analysis)
        from .entropy_probe import EntropyProbe
        entropy_config = cfg.get("entropy_probe", {})
        self.entropy_probe = EntropyProbe(
            enabled=entropy_config.get("enabled", False),
            debug=entropy_config.get("debug", False),
            max_sequences=entropy_config.get("max_sequences", 1000),
            store_full_kernel=entropy_config.get("store_full_kernel", True)
        )
        
        self._last_entropy_metrics: Dict[str, float] = {}  # Store entropy metrics for logging
        self._last_simple_entropy_metrics: Dict[str, float] = {}  # Store simple entropy metrics for logging
        
        # Simple entropy probe accumulation
        self._entropy_grad_accumulator = None  # Accumulate entropy gradients across microbatches
        self._last_sync_grads = True  # Track previous sync_grads to detect new training step start

    # ------------------------------------------------------------------------
    # CORE TRAINING METHODS
    # ------------------------------------------------------------------------

    @torch.no_grad()
    def _compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return centred (but *not* normalised) advantage for Dr-GRPO."""
        mean_r = rewards.mean(dim=1, keepdim=True)
        return rewards - mean_r  # (B, G)
        
    @torch.no_grad() 
    def _compute_sequence_log_probs(self, rollouts: RolloutBatch) -> torch.Tensor:
        """Compute sequence-level log probabilities S(t) = log π(t) for entropy probe."""
        # Use the precomputed logprobs from rollouts (per-token) and sum over generation
        B, G, T_gen = rollouts.gen_ids.shape
        gen_mask = (rollouts.gen_ids != self.pad_id).float()  # (B, G, T_gen)
        
        # Sum per-token log probabilities to get sequence log probabilities
        seq_log_probs = (rollouts.logprobs * gen_mask).sum(dim=2)  # (B, G)
        
        return seq_log_probs
    
        
    def call_entropy_probe_on_buffer(self, rollouts: RolloutBatch) -> None:
        """
        Call entropy probe on full buffer of rollouts (called externally by training runner).
        
        Args:
            rollouts: Full buffer rollouts (B=buffer_size, G=num_generations)
        """
        if not self.entropy_probe.enabled or rollouts is None:
            return
            
        # Compute advantages and sequence log probabilities for full buffer
        advantages = self._compute_advantage(rollouts.reward)  # (B, G)
        seq_log_probs = self._compute_sequence_log_probs(rollouts)  # (B, G)
        
        # Get current learning rate
        current_lr = self.opt.param_groups[0]['lr']
        
        try:
            # Store entropy probe data (this computes per-sequence gradients)
            self.entropy_probe.store_step_data(
                rollouts=rollouts,
                advantages=advantages,
                log_probs=seq_log_probs,
                trainable_params=self._trainable_params(),
                optimizer=self.opt,
                learning_rate=current_lr,
                step_idx=self.actual_opt_step,
                policy_model=self.policy,
                pad_id=self.pad_id
            )
            
            # Get computed metrics
            self._last_entropy_metrics = self.entropy_probe.get_metrics()
            
            # Save detailed data periodically
            entropy_config = self.cfg.get("entropy_probe", {})
            save_every = entropy_config.get("save_every", 10)
            if self.actual_opt_step % save_every == 0:
                save_path = f"/tmp/entropy_probe_step_{self.actual_opt_step}.json"
                self.entropy_probe.save_data(save_path)
                if self.entropy_probe.debug:
                    print(f"[EntropyProbe] Saved detailed data to {save_path}")
            
        except Exception as e:
            if self.entropy_probe.debug:
                print(f"[EntropyProbe] Error in buffer step {self.actual_opt_step}: {e}")
            # Clear partial state on error
            self.entropy_probe.reset()

    def step(
        self,
        rollouts: RolloutBatch,
        ref_model,
        *,
        sync_grads: bool = True,
        call_entropy_probe: bool = True,
    ) -> Dict[str, float]:
        """
        One optimisation step (or gradient-accumulation micro-step).
        
        Args:
            call_entropy_probe: If False, skip entropy probe to allow buffer-level accumulation
        """
        # Debug distributed training hanging
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[ALGO DEBUG] Rank {rank} entered step(), sync_grads={sync_grads}")
        
        self.device = rollouts.gen_ids.device
        B, G, T_g = rollouts.gen_ids.shape
        print(f"[ALGO DEBUG] Rank {rank} got batch shape B={B}, G={G}, T_g={T_g}")

        # 1) prepare tensors & advantages
        print(f"[ALGO DEBUG] Rank {rank} building sequences...")
        seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences(rollouts)
        print(f"[ALGO DEBUG] Rank {rank} computing advantages...")
        adv = self._compute_advantage(rollouts.reward)  # (B, G)
        print(f"[ALGO DEBUG] Rank {rank} completed sequence prep and advantage computation")

        # 2) forward & gather log-probs
        print(f"[ALGO DEBUG] Rank {rank} about to call _policy_logp...")
        new_logp = self._policy_logp(seq_flat, attn_mask, targets_tok, B, G, T_g).view(B, G, T_g)
        print(f"[ALGO DEBUG] Rank {rank} completed _policy_logp call")
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
            ref_logp = ref_logp_all.gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[..., -T_g:].view_as(new_logp)


            # KL(pi||ref) sample estimator = E_pi[logpi - logref]
            kl_tok = ((new_logp - ref_logp) * gen_mask)
            kl_mean = kl_tok.sum() / (gen_mask.sum() + 1e-8)

            loss = loss + self.cfg["kl_beta"] * kl_mean
        else:
            kl_mean = self._kl_metric_only(seq_flat, attn_mask, targets_tok, new_logp, ref_model, gen_mask, T_g)

        # 4.5) Simple entropy probe gradient accumulation (per microbatch)
        if self.simple_entropy_probe.enabled and rollouts is not None and call_entropy_probe:
            # Reset accumulator if starting new training step (previous was final microbatch, this is first)
            if self._last_sync_grads and not sync_grads:
                self._entropy_grad_accumulator = None
            
            # Accumulate entropy gradients for this microbatch (use token-level new_logp)
            self._accumulate_entropy_gradients(rollouts, new_logp, gen_mask)
        
        # Track sync_grads state for next call
        self._last_sync_grads = sync_grads

        # 5) optional ratio logging
        self._maybe_log_ratios(ratios, gen_mask)

        # 6) optimise (pass rollouts for ESS computation if GNS enabled or entropy probe enabled)
        pass_rollouts = self.gns_probe.enabled or self.entropy_probe.enabled or self.simple_entropy_probe.enabled
        self._backward_and_step(loss, sync_grads, rollouts if pass_rollouts else None, call_entropy_probe)

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
        
        # Add GNS metrics if available
        if self._last_gns_metrics:
            metrics.update(self._last_gns_metrics)
            
        # Add entropy probe metrics if available
        if self._last_entropy_metrics:
            metrics.update(self._last_entropy_metrics)
            
        # Add simple entropy probe metrics if available
        if self._last_simple_entropy_metrics:
            metrics.update(self._last_simple_entropy_metrics)

        # ====================================================================
        # ENTROPY PROBE SECTION (can be refactored out)
        # ====================================================================
        # Added 8/11
        probe_cfg = self.cfg.get("entropy_probe", {})
        do_probe = int(probe_cfg.get("every", 0)) > 0
        if do_probe and self._is_main_process():
            self._probe_accumulate_microbatch(rollouts)             # Added 8/11
            # Clear gradients after entropy probe to prevent DDP hanging in GNS probe
            self.opt.zero_grad(set_to_none=True)
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

    # ------------------------------------------------------------------------
    # CORE ALGORITHM HELPER METHODS
    # ------------------------------------------------------------------------
    
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
        # Debug distributed training hanging
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[POLICY DEBUG] Rank {rank} entered _policy_logp, seq_flat.shape={seq_flat.shape}")
        
        with torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            print(f"[POLICY DEBUG] Rank {rank} about to call self.policy forward pass...")
            logits = self.policy(seq_flat, attention_mask=attn_mask).logits
            print(f"[POLICY DEBUG] Rank {rank} completed self.policy forward pass, logits.shape={logits.shape}")

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
    def _backward_and_step(self, loss: torch.Tensor, sync_grads: bool, rollouts: RolloutBatch | None = None, call_entropy_probe: bool = False) -> None:
        """Handle backward, gradient clipping, optimiser & scheduler."""
        maybe = (self.policy.no_sync if (hasattr(self.policy, "no_sync") and not sync_grads) else nullcontext)
        with maybe():
            loss.backward()
        
        if sync_grads:
            # ================================================================
            # GNS PROBE: Store gradient snapshot after full step
            # ================================================================
            if self.gns_probe.enabled:
                # Compute Dr-GRPO weight for this step
                dr_grpo_weight = 1.0  # Default weight
                if rollouts is not None:
                    B, G, T_g = rollouts.gen_ids.shape
                    gen_mask = (rollouts.gen_ids != self.pad_id).float()
                    
                    # Compute total Dr-GRPO weight for this step
                    tokens_per_gen = gen_mask.sum(dim=2)  # (B, G)
                    L_max = tokens_per_gen.max(dim=-1).values.clamp_min(1.0)  # (B,)
                    
                    # Sum weights across all samples: Σ_b (G / (G * L_max[b])) = Σ_b (1 / L_max[b])
                    dr_grpo_weight = float((1.0 / L_max).sum().item())
                
                # Store gradient snapshot
                self.gns_probe.store_gradient(self._trainable_params(), dr_grpo_weight)
                
                # Compute metrics if we have enough data
                if self.gns_probe.should_compute():
                    buffer_size = self.cfg.get("buffer_size", 32)
                    self._last_gns_metrics = self.gns_probe.compute_metrics(buffer_size)
                    
            # ================================================================
            # SIMPLE ENTROPY PROBE: Fast δH prediction during regular training
            # ================================================================
            # Simple entropy probe will be called from step() method where new_logp is available
                    
            # ================================================================
            # COMPLEX ENTROPY PROBE: Store per-sequence gradients and compute δH
            # ================================================================
            if self.entropy_probe.enabled and rollouts is not None and call_entropy_probe:
                # Compute advantages and sequence log probabilities
                advantages = self._compute_advantage(rollouts.reward)  # (B, G)
                seq_log_probs = self._compute_sequence_log_probs(rollouts)  # (B, G)
                
                # Get current learning rate
                current_lr = self.opt.param_groups[0]['lr']
                
                try:
                    # Store entropy probe data (this computes per-sequence gradients)
                    self.entropy_probe.store_step_data(
                        rollouts=rollouts,
                        advantages=advantages,
                        log_probs=seq_log_probs,
                        trainable_params=self._trainable_params(),
                        optimizer=self.opt,
                        learning_rate=current_lr,
                        step_idx=self.actual_opt_step,
                        policy_model=self.policy,
                        pad_id=self.pad_id
                    )
                    
                    # Get computed metrics
                    self._last_entropy_metrics = self.entropy_probe.get_metrics()
                    
                    # Save detailed data periodically
                    entropy_config = self.cfg.get("entropy_probe", {})
                    save_every = entropy_config.get("save_every", 10)
                    if self.actual_opt_step % save_every == 0:
                        save_path = f"/tmp/entropy_probe_step_{self.actual_opt_step}.json"
                        self.entropy_probe.save_data(save_path)
                        if self.entropy_probe.debug:
                            print(f"[EntropyProbe] Saved detailed data to {save_path}")
                    
                except Exception as e:
                    if self.entropy_probe.debug:
                        print(f"[EntropyProbe] Error in step {self.actual_opt_step}: {e}")
                    # Clear partial state on error
                    self.entropy_probe.reset()
            
            clip_grad_norm_(self.policy.parameters(), self.cfg["grad_clip"])
            self.opt.step()
            if self.lr_sched is not None:
                self.lr_sched.step()
            
            # Complete simple entropy probe calculation (after optimizer step, with current Adam states)
            if self.simple_entropy_probe.enabled and self._entropy_grad_accumulator is not None:
                try:
                    # Get current learning rate
                    current_lr = self.opt.param_groups[0]['lr']
                    
                    # Complete delta H calculation with accumulated gradients + current Adam states
                    entropy_metrics = self.simple_entropy_probe.complete_delta_h_calculation(
                        accumulated_entropy_grads=self._entropy_grad_accumulator,
                        trainable_params=self._trainable_params(),
                        optimizer=self.opt,
                        learning_rate=current_lr
                    )
                    
                    self._last_simple_entropy_metrics = entropy_metrics
                    
                except Exception as e:
                    if self.simple_entropy_probe.debug:
                        print(f"[SimpleEntropyProbe] Error completing calculation: {e}")
                    self._last_simple_entropy_metrics = {}
            
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

    # ========================================================================
    # ENTROPY PROBE METHODS (TODO: refactor to separate module)
    # ========================================================================
    # Added 8/11
    def _is_main_process(self) -> bool:  # Added 8/11
        return (not dist.is_initialized()) or (dist.get_rank() == 0)

    def _run_dir_from_ratio_path(self) -> pathlib.Path:  # Added 8/11
        if getattr(self, "_ratio_log_path", None) is not None:
            return pathlib.Path(self._ratio_log_path).parent
        return pathlib.Path(".")

    def _probe_reset_window(self) -> None:  # Added 8/11
        self._probe_sumWS = 0.0
        self._probe_sumW  = 0.0
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
        # Use unwrapped model to get parameters without DDP wrapper
        unwrapped_policy = _unwrap(self.policy)
        return [p for p in unwrapped_policy.parameters() if isinstance(p, torch.nn.Parameter) and p.requires_grad]
    
    def _accumulate_entropy_gradients(self, rollouts: RolloutBatch, new_logp: torch.Tensor, gen_mask: torch.Tensor) -> None:
        """
        Accumulate entropy gradients for current microbatch (parallel to training gradient accumulation).
        
        Args:
            rollouts: Current microbatch rollouts
            new_logp: Token-level log probabilities with gradients (B, G, T_g)
            gen_mask: Generation mask (B, G, T_g)
        """
        try:
            # Compute advantages for this microbatch
            advantages = self._compute_advantage(rollouts.reward)  # (B, G)
            
            # Compute entropy gradients for this microbatch using autograd.grad()
            entropy_grads = self.simple_entropy_probe.compute_entropy_gradients_microbatch(
                new_logp, gen_mask, advantages, self._trainable_params(), self.pad_id
            )
            
            # Accumulate across microbatches (same pattern as training gradients)
            if self._entropy_grad_accumulator is None:
                self._entropy_grad_accumulator = entropy_grads
            else:
                self._entropy_grad_accumulator += entropy_grads
                
        except Exception as e:
            if self.simple_entropy_probe.debug:
                print(f"[SimpleEntropyProbe] Error accumulating gradients: {e}")
            # Don't break training if entropy probe fails
            pass
    
    # ========================================================================
    # ENTROPY PROBE IMPLEMENTATION (complex, needs refactoring)
    # ========================================================================
    
    def _probe_accumulate_microbatch(self, rollouts: RolloutBatch) -> None:  # Added 8/11
        """
        Re-forward on this microbatch, compute:
          GH1 += Σ S ∇S,  GA1 += Σ SA ∇S,  G1 += Σ ∇S
        and accumulate S, A, token-entropy means for logging.
        """
        # (Re)build sequences
        seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences(rollouts)
        B, G, T_g = rollouts.gen_ids.shape

        # Forward fresh (don't rely on training graph)
        # Use unwrapped model to bypass DDP for entropy probe
        unwrapped_policy = _unwrap(self.policy)
        with torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            logits = unwrapped_policy(seq_flat, attention_mask=attn_mask).logits
        logits = logits / self.cfg.get("temperature", 1.0)
        logp_all = F.log_softmax(logits.float(), dim=-1)
        new_logp = logp_all[:, :-1].gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g:]
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
        new_logp = new_logp.view(B, G, T_g)

        # Per-seq S and A
        S_seq = (new_logp * gen_mask).sum(dim=-1)            # (B,G), S = sum log p over generated tokens
        A_seq = self._compute_advantage(rollouts.reward)     # (B,G)  (GRPO => centered per-prompt)
        tok_ent_mean = (-(new_logp) * gen_mask).sum(-1) / (gen_mask.sum(-1) + 1e-8)

        # ---- Dr-GRPO weights: w_{p,i} = 1 / (G * Lmax(p)) ----
        tokens_per_gen = gen_mask.sum(dim=2)                 # (B,G)
        Lmax = tokens_per_gen.max(dim=-1).values.clamp_min(1.0)   # (B,)
        w_p = (1.0 / (G * Lmax)).unsqueeze(-1).expand_as(S_seq)   # (B,G)

        # Flatten
        S = S_seq.reshape(-1)                                 # (N,)
        A = A_seq.reshape(-1)                                 # (N,)
        w = w_p.reshape(-1)                                   # (N,)
        N = S.numel()

        # Weighted scalar objectives that produce the exact sums we need:
        #   GH  = Σ w * S ∇S  via grad of 0.5 * Σ w * S^2
        #   GA  = Σ w * A ∇S  via grad of Σ w * A * S
        #   G1  = Σ w     ∇S  via grad of Σ w * S
        params = self._trainable_params()
        L_GH = 0.5 * (w * S.pow(2)).sum()
        L_GA = (w * A * S).sum()
        L_G1 = (w * S).sum()

        # Gradients (avoid DDP sync like GNS probe to prevent distributed hanging)
        from contextlib import nullcontext
        ctx = self.policy.no_sync() if hasattr(self.policy, "no_sync") else nullcontext()
        with ctx:
            grads_GH = torch.autograd.grad(L_GH, params, retain_graph=True,  create_graph=False, allow_unused=True)
            grads_GA = torch.autograd.grad(L_GA, params, retain_graph=True,  create_graph=False, allow_unused=True)
            grads_G1 = torch.autograd.grad(L_G1, params, retain_graph=False, create_graph=False, allow_unused=True)

        # Lazy-init accum dicts (unchanged)
        if self._probe_GH1 is None:
            self._probe_GH1, self._probe_GA1, self._probe_G1 = {}, {}, {}
            for p in params:
                dev, dtype, shape = p.device, torch.float32, p.shape
                self._probe_GH1[p] = torch.zeros(shape, device=dev, dtype=dtype)
                self._probe_GA1[p] = torch.zeros(shape, device=dev, dtype=dtype)
                self._probe_G1[p]  = torch.zeros(shape, device=dev, dtype=dtype)

        for p, gH, gA, g1 in zip(params, grads_GH, grads_GA, grads_G1):
            if gH is not None: self._probe_GH1[p] += gH.detach().to(torch.float32)
            if gA is not None: self._probe_GA1[p] += gA.detach().to(torch.float32)
            if g1 is not None: self._probe_G1[p]  += g1.detach().to(torch.float32)

        # Weighted sums for centering/normalization
        self._probe_sumWS  += float((w * S).sum().item())   # Σ w S
        self._probe_sumW   += float(w.sum().item())         # Σ w
        self._probe_N      += int(N)
        self._probe_S_list.append(S_seq.detach().cpu())
        self._probe_A_list.append(A_seq.detach().cpu())
        self._probe_tokent_list.append(tok_ent_mean.detach().cpu())


    def _probe_finalize_and_log(self, probe_cfg: dict) -> None:  # Added 8/11
        """On the syncing microbatch, turn accumulators into centered g_H, g_A,
        compute g_H^T P g_A from Adam state, save artifacts, then reset window.
        """
        if self._probe_GH1 is None:
            return

        # Weighted mean of S for centering: S̄ = (Σ w S)/(Σ w)
        meanS = (self._probe_sumWS / max(self._probe_sumW, 1e-8))
        sumW  = self._probe_sumW

        dot = normH = normA = 0.0
        lr_vals = []

        for group in self.opt.param_groups:
            eps = float(group.get("eps", 1e-8))
            lr_vals.append(float(group.get("lr", 0.0)))
            for p in group["params"]:
                if p not in self._probe_GH1:
                    continue
                # gH = GH - S̄ * G1  (centered)
                gH = self._probe_GH1[p] - meanS * self._probe_G1[p]
                # gA = GA            (no centering; GRPO A is already baseline-subtracted per prompt)
                gA = self._probe_GA1[p]

                state = self.opt.state.get(p, {})
                v = state.get("exp_avg_sq", None)

                # Note here we use a biased estimator of \sum_i E_t[(S(t)-Sbar) K(t,t_i') A(t_i')]
                # To remove bias we should replace the sum over all t_i and sampled t = t_j with the sum for i != j
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

        # Unnormalized inner product in the Adam metric:
        gH_P_gA_sum = float(dot)

        # Normalized to product of expectations:
        gH_P_gA_mean = float(dot) / max(sumW * sumW, 1e-8)

        deltaH_pred_adam = -(avg_lr * gH_P_gA_mean)


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
        pred = {
            "gH_P_gA_sum": gH_P_gA_sum,
            "gH_P_gA_mean": gH_P_gA_mean,
            "norm_gH_P": float(np.sqrt(max(normH, 0.0))),
            "norm_gA_P": float(np.sqrt(max(normA, 0.0))),
            "deltaH_first_order_adam_mean": float(deltaH_pred_adam),
        }
        (step_dir / "deltaH_pred.json").write_text(json.dumps(pred, indent=2))

        meta = {
            "step": int(self.actual_opt_step),
            "N": int(self._probe_N),
            "avg_lr": avg_lr,
            "sumW": sumW,
        }
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2))


        # Reset accumulators for next window
        self._probe_reset_window()

    # Placeholder left here so external calls don’t break older imports  # Added 8/11
    def _log_entropy_probes(self, *args, **kwargs) -> None:
        return  # no-op


    # === GNS helpers =============================================================
    def _loss_for_batch(self, rollouts: "RolloutBatch", ref_model) -> torch.Tensor:
        """
        Build the same PPO/GRPO loss as in step(), but:
        - no grad accumulation scaling,
        - no optimiser step, just returns the scalar loss.
        """
        print(f"[GNS DEBUG] _loss_for_batch: starting")
        device = rollouts.gen_ids.device
        B, G, T_g = rollouts.gen_ids.shape
        print(f"[GNS DEBUG] _loss_for_batch: batch shape B={B}, G={G}, T_g={T_g}")

        # 1) prep + advantages
        print(f"[GNS DEBUG] _loss_for_batch: calling _build_sequences")
        seq_flat, attn_mask, targets_tok, gen_mask = self._build_sequences(rollouts)
        print(f"[GNS DEBUG] _loss_for_batch: completed _build_sequences, seq_flat.shape={seq_flat.shape}")
        print(f"[GNS DEBUG] _loss_for_batch: calling _compute_advantage")
        adv = self._compute_advantage(rollouts.reward)  # (B, G)
        print(f"[GNS DEBUG] _loss_for_batch: completed _compute_advantage")

        # 2) log-probs
        print(f"[GNS DEBUG] _loss_for_batch: starting forward pass")
        # Use unwrapped model to bypass DDP for GNS probe
        unwrapped_policy = _unwrap(self.policy)
        print(f"[GNS DEBUG] _loss_for_batch: using unwrapped model, type={type(unwrapped_policy)}")
        with torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
            print(f"[GNS DEBUG] _loss_for_batch: calling policy forward pass with seq_flat.shape={seq_flat.shape}")
            logits = unwrapped_policy(seq_flat, attention_mask=attn_mask).logits
            print(f"[GNS DEBUG] _loss_for_batch: completed policy forward pass, logits.shape={logits.shape}")
        logits = logits / self.cfg.get("temperature", 1.0)
        logp_all = F.log_softmax(logits.float(), dim=-1)
        new_logp = logp_all[:, :-1].gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[:, -T_g:]
        new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)
        new_logp = new_logp.view(B, G, T_g)

        old_logp = rollouts.logprobs  # (B, G, T_g)

        # 3) PPO surrogate (same as _ppo_surrogate but returns the mean loss)
        ratios = torch.exp((new_logp - old_logp).clamp(-80, 80)) * gen_mask
        clip_eps_neg = self.cfg["clip_eps"]
        clip_eps_pos = self.cfg.get("clip_+", self.cfg["clip_eps"])
        surr1 = ratios * adv.unsqueeze(-1)
        surr2 = torch.clamp(ratios, 1 - clip_eps_neg, 1 + clip_eps_pos) * adv.unsqueeze(-1)
        token_loss = -torch.min(surr1, surr2) * gen_mask  # (B, G, T_g)
        tokens_per_gen = gen_mask.sum(dim=2)              # (B, G)
        max_lens = tokens_per_gen.max(dim=-1).values      # (B,)
        loss_per_prompt = token_loss.sum(dim=(1, 2)) / (gen_mask.shape[1] * max_lens + 1e-8)
        loss = loss_per_prompt.mean()

        # 4) optional differentiable KL (same beta as training)
        beta = float(self.cfg.get("kl_beta", 0.0))
        if beta > 0.0:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.cfg.get("bf16", True), dtype=torch.bfloat16):
                ref_logits = ref_model(seq_flat, attention_mask=attn_mask).logits
            ref_logits = ref_logits / self.cfg.get("temperature", 1.0)
            ref_logp_all = F.log_softmax(ref_logits.float(), dim=-1)[..., :-1]
            ref_logp = ref_logp_all.gather(-1, targets_tok.unsqueeze(-1)).squeeze(-1)[..., -T_g:].view_as(new_logp)
            kl_tok = ((new_logp - ref_logp) * gen_mask)
            kl_mean = kl_tok.sum() / (gen_mask.sum() + 1e-8)
            loss = loss + beta * kl_mean

        return loss

    @torch.no_grad()
    def _grad_sq_norm_for_batch(self, rollouts: "RolloutBatch", ref_model, *, avoid_ddp_allreduce: bool = True) -> float:
        """
        Compute sum of squared gradients for the mean loss on `rollouts`.
        Does NOT step the optimiser. Optionally suppresses DDP all-reduce.
        """
        # zero grads
        self.opt.zero_grad(set_to_none=True)

        # we need grads, so disable no_grad inside
        for p in self.policy.parameters():
            if p.grad is not None:
                p.grad = None

        ctx = nullcontext()
        if avoid_ddp_allreduce and hasattr(self.policy, "no_sync"):
            ctx = self.policy.no_sync()

        with ctx:
            # enable grads locally
            with torch.enable_grad():
                loss = self._loss_for_batch(rollouts, ref_model)
                loss.backward()

        # grad^2 norm over trainable params (LoRA etc.)
        total = 0.0
        for p in self._trainable_params():
            if p.grad is not None:
                g = p.grad.detach()
                if torch.isfinite(g).all():
                    total += float((g.double() * g.double()).sum().item())
        # clear grads so training step isn't affected
        self.opt.zero_grad(set_to_none=True)
        return total

