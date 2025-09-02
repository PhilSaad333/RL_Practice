"""
DeltaEntropyIS: Importance-sampling based estimator for delta entropy using E/U batches.

Stage 1: Compatibility shim that preserves the existing public API and delegates
to the legacy ImportanceSampler implementation. Subsequent stages will migrate
the internals to use SequenceProcessor and RB entropies per the refactor plan.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
from . import distributed_helpers
from sequence_processing.sequence_processor import BatchedSequences


class DeltaEntropyIS:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        config: Dict[str, Any],
        logger,
        sequence_processor: Optional[object] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.logger = logger
        self.sequence_processor = sequence_processor  # preferred; may be None in Stage 1
        # AMP settings for TF and RL update paths
        self.use_amp = config.get('memory_config', {}).get('amp', False)
        dtype_name = config.get('memory_config', {}).get('dtype', 'bfloat16')
        self.amp_dtype = getattr(torch, dtype_name, torch.bfloat16)

        # Note: Stage 1 delegates to legacy class; mark for removal in later stages
        try:
            from .importance_sampling import ImportanceSampler  # type: ignore
            self._legacy = ImportanceSampler(model=model, config=config, logger=logger)
        except Exception as e:
            self._legacy = None
            self.logger.warning(
                f"DeltaEntropyIS: legacy ImportanceSampler unavailable ({e}); RB refactor pending."
            )

    def entropy_change_two_batch(
        self,
        model: torch.nn.Module,
        E_batch: Dict[str, Any],
        U_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        cfg_importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stage 1: Pass-through to legacy implementation to preserve behavior.
        Later stages will replace this with SequenceProcessor + RB entropies.
        """
        if self._legacy is None:
            raise RuntimeError(
                "DeltaEntropyIS (Stage 1): Legacy ImportanceSampler not available; "
                "cannot perform entropy_change_two_batch yet."
            )
        return self._legacy.entropy_change_two_batch(
            model, E_batch, U_batch, optimizer, cfg_importance
        )



    def _get_generation_lengths(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Get generation lengths for per-token entropy computation."""
        if 'gen_lengths' in batch:
            return batch['gen_lengths']
        
        # Fallback: compute from sequences and attention masks
        sequences = batch['sequences']  # [batch_size, G, max_len]
        attention_masks = batch['attention_masks']  # [batch_size, G, max_len]
        prompt_lens = batch['prompt_lens']  # [batch_size]
        
        batch_size, G, max_len = sequences.shape
        gen_lengths = torch.zeros(batch_size, G, dtype=torch.long, device=sequences.device)
        
        for b in range(batch_size):
            for g in range(G):
                mask = attention_masks[b, g]  # [max_len]
                prompt_len = prompt_lens[b]
                gen_mask = mask[prompt_len:]  # Generation portion
                gen_lengths[b, g] = gen_mask.sum()
        
        return gen_lengths


    def _compute_training_loss_microbatch(self, microbatch: Dict[str, Any], training_loss: str) -> Tuple[torch.Tensor, int]:
        """Compute training loss for a microbatch."""
        if training_loss == "nll":
            # Use negative log-likelihood (teacher forcing) loss
            return self._compute_nll_loss_microbatch(microbatch)
        elif training_loss == "rl":
            # TODO: Implement RL loss when available
            self.logger.warning("RL training loss not implemented, falling back to NLL")
            return self._compute_nll_loss_microbatch(microbatch)
        else:
            raise ValueError(f"Unknown training_loss: {training_loss}")


    def _snapshot_model_optimizer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, snapshot_device: str = 'cpu') -> tuple[dict[str, torch.Tensor], Optional[dict]]:
        """Snapshot model and optimizer states."""
        self.logger.debug("Snapshotting model and optimizer state")
        
        # Save model parameters to CPU
        cpu_snaps = {}
        for name, param in model.named_parameters():
            cpu_snaps[name] = param.detach().to(snapshot_device).clone()
        
        # Save optimizer state (optional but recommended)
        opt_state_snapshot = None
        if hasattr(optimizer, 'state_dict'):
            opt_state_dict = optimizer.state_dict()
            opt_state_snapshot = {}
            for key, value in opt_state_dict.items():
                if isinstance(value, dict):
                    # Handle optimizer state for each parameter
                    opt_state_snapshot[key] = {}
                    for param_key, param_value in value.items():
                        if isinstance(param_value, torch.Tensor):
                            opt_state_snapshot[key][param_key] = param_value.detach().to(snapshot_device).clone()
                        else:
                            opt_state_snapshot[key][param_key] = param_value
                elif isinstance(value, torch.Tensor):
                    opt_state_snapshot[key] = value.detach().to(snapshot_device).clone()
                else:
                    opt_state_snapshot[key] = value
        return cpu_snaps, opt_state_snapshot

    
    def _restore_model_optimizer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cpu_snaps: dict[str, torch.Tensor], opt_state_snapshot: Optional[dict]) -> None:
        """Restore model and optimizer states from snapshots."""
        self.logger.debug("Restoring model and optimizer state")

        # Restore model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(cpu_snaps[name].to(param.device))
        
        # Restore optimizer state
        if opt_state_snapshot is not None:
            try:
                # Reconstruct state dict and load
                restored_state = {}
                for key, value in opt_state_snapshot.items():
                    if isinstance(value, dict):
                        restored_state[key] = {}
                        for param_key, param_value in value.items():
                            if isinstance(param_value, torch.Tensor):
                                # Move back to original device
                                restored_state[key][param_key] = param_value.to(optimizer.param_groups[0]['params'][0].device)
                            else:
                                restored_state[key][param_key] = param_value
                    else:
                        restored_state[key] = value
                        
                optimizer.load_state_dict(restored_state)
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")
        
        # Clear any remaining gradients
        optimizer.zero_grad(set_to_none=True)


    def _compute_snis_two_batch(self, payload_upd: torch.Tensor, logw: torch.Tensor,
                               report_per_token: bool, E_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SNIS estimate for updated entropy using a generic payload.

        - Naive (default): payload_upd = S_upd (log p); H_upd = -E_w[S_upd]
        - RB (if importance.entropy_mode == 'rb'): payload_upd = RB_upd; H_upd = E_w[RB_upd]
        """
        is_dist, rank, world_size = distributed_helpers.get_dist_info()
        # Stabilize weights in log domain
        logw_max = logw.max()
        w_shift = torch.exp(logw - logw_max)  # [B, G]

        # Local sums
        w_sum_local = w_shift.sum()
        wPayload_sum_local = (w_shift * payload_upd).sum()
        w_sq_sum_local = (w_shift ** 2).sum()

        # Global reductions
        if is_dist:
            w_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sum_local)
            wPayload_sum_global = distributed_helpers.all_reduce_scalar_sum(wPayload_sum_local)
            w_sq_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sq_sum_local)
            self.logger.debug(f"Rank {rank}: local w_sum={w_sum_local.item():.6f}, global w_sum={w_sum_global:.6f}")
        else:
            w_sum_global = float(w_sum_local.item())
            wPayload_sum_global = float(wPayload_sum_local.item())
            w_sq_sum_global = float(w_sq_sum_local.item())

        # Entropy estimate
        entropy_mode = (self.config.get('importance', {}) or {}).get('entropy_mode', 'naive')
        if entropy_mode == 'rb':
            H_upd = (wPayload_sum_global / w_sum_global) if w_sum_global != 0 else 0.0
        else:
            H_upd = (-wPayload_sum_global / w_sum_global) if w_sum_global != 0 else 0.0

        # Diagnostics
        ESS = (w_sum_global ** 2) / w_sq_sum_global if w_sq_sum_global != 0 else 0.0
        results = {
            'H_upd': H_upd,
            'diagnostics': {
                'ESS': ESS,
                'w_max': w_shift.max().item(),
                'w_min': w_shift.min().item(),
                'w_sum_global': w_sum_global,
            }
        }

        # Per-token variant
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)  # [B, G]
            wL_sum_local = (w_shift * lengths).sum()
            if is_dist:
                wL_sum_global = distributed_helpers.all_reduce_scalar_sum(wL_sum_local)
            else:
                wL_sum_global = float(wL_sum_local.item())
            if entropy_mode == 'rb':
                H_upd_tok = (wPayload_sum_global / wL_sum_global) if wL_sum_global > 0 else 0.0
            else:
                H_upd_tok = (-wPayload_sum_global / wL_sum_global) if wL_sum_global > 0 else 0.0
            results['H_upd_tok'] = H_upd_tok

        return results

    def _to_batched_sequences_from_probe(self, batch: Dict[str, Any]) -> BatchedSequences:
        """Adapt probe-format batch dict to BatchedSequences for SequenceProcessor."""
        sequences = batch['sequences']            # [B, G, T]
        attention_masks = batch['attention_masks']# [B, G, T]
        prompt_lens = batch['prompt_lens']        # List[B]
        B, G, T = sequences.shape
        # Compute gen_lens per [B][G]
        gen_lens: list[list[int]] = []
        for b in range(B):
            row: list[int] = []
            pl = int(prompt_lens[b])
            for g in range(G):
                L = int(attention_masks[b, g].long().sum().item())
                row.append(max(0, L - pl))
            gen_lens.append(row)
        return BatchedSequences(
            sequences=sequences,
            attention_masks=attention_masks,
            prompt_lens=prompt_lens,
            gen_lens=gen_lens,
            responses_text=None,
        )

    def _eval_S_and_RB_on_E(self, E_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sequence S (logprob sums) and RB (RB entropy sums) on E using SequenceProcessor."""
        if self.sequence_processor is None:
            # Fallback: use local evaluator for S and zero RB (not recommended)
            S = self._eval_logprobs_on_batch(E_batch, self.config.get('importance', {}).get('importance_microbatch_size', 1))
            RB = torch.zeros_like(S)
            return S, RB
        bs = self._to_batched_sequences_from_probe(E_batch)
        logprob_results, _diag = self.sequence_processor.teacher_force_logprobs_with_diagnostics(
            sequences=bs,
            with_grad=False,
            tf_batch_size=self.sequence_processor.config.tf_batch_size if hasattr(self.sequence_processor, 'config') else None,
            compute_rb=True,
        )
        device = next(self.model.parameters()).device
        # Sequence logprobs
        seq_lp_list = []  # [B][G]
        for b_list in logprob_results.sequence_logprobs:
            seq_lp_list.append([float(x) for x in b_list])
        S = torch.tensor(seq_lp_list, device=device, dtype=torch.float32)
        # RB sums
        RB_vals: list[list[float]] = []
        if getattr(logprob_results, 'rb_entropies_torch', None):
            for b in range(len(logprob_results.rb_entropies_torch)):
                row = []
                for g in range(len(logprob_results.rb_entropies_torch[b])):
                    rb_t = logprob_results.rb_entropies_torch[b][g]
                    row.append(float(rb_t.detach().sum().item()) if rb_t is not None and rb_t.numel() > 0 else 0.0)
                RB_vals.append(row)
        else:
            # Use numpy arrays
            for b in range(len(logprob_results.rb_entropies)):
                row = []
                for g in range(len(logprob_results.rb_entropies[b])):
                    rb_np = logprob_results.rb_entropies[b][g]
                    row.append(float(rb_np.sum()) if rb_np is not None and rb_np.size > 0 else 0.0)
                RB_vals.append(row)
        RB = torch.tensor(RB_vals, device=device, dtype=torch.float32)
        return S, RB

    def _rl_update_streaming(self, U_batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
                             rl_grad_accum: int, importance_mb_size: int):
        """Perform RL-aligned update step using GRPO objective with gradient accumulation."""
        self.logger.debug("Taking RL-aligned optimizer step on U batch")
        sequences = U_batch['sequences']  # [B_U, G, max_len]
        prompt_lens = U_batch['prompt_lens']  # [B_U]
        attention_masks = U_batch['attention_masks']  # [B_U, G, max_len]
        advantages = U_batch['advantages']  # [B_U, G]
        max_lengths = U_batch['max_lengths']  # [B_U]
        B_U, G, max_len = sequences.shape
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        mb_size_prompts = importance_mb_size
        total_loss = 0.0
        num_microbatches = 0
        for start_b in range(0, B_U, mb_size_prompts):
            end_b = min(start_b + mb_size_prompts, B_U)
            B_mb = end_b - start_b
            mb_seqs = sequences[start_b:end_b]
            mb_masks = attention_masks[start_b:end_b]
            mb_advantages = advantages[start_b:end_b]
            mb_max_lengths = max_lengths[start_b:end_b]
            mb_prompt_lens = prompt_lens[start_b:end_b]
            flat_seqs = mb_seqs.view(-1, max_len)
            flat_masks = mb_masks.view(-1, max_len)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(flat_seqs, attention_mask=flat_masks).logits
            logits = logits.float()
            logp_all = F.log_softmax(logits, dim=-1)
            targets = flat_seqs[:, 1:].unsqueeze(-1)
            new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)
            new_logp = new_logp.view(B_mb, G, -1)
            mb_loss_terms = []
            for b in range(B_mb):
                prompt_len = mb_prompt_lens[b]
                A_b = mb_advantages[b]
                L_max_b = max(mb_max_lengths[b], 1)
                gen_start = prompt_len - 1
                new_logp_gen = new_logp[b, :, gen_start:]
                gen_mask = mb_masks[b, :, prompt_len:].float()
                min_gen_len = min(new_logp_gen.shape[1], gen_mask.shape[1]) if new_logp_gen.shape[1] > 0 else 0
                if min_gen_len > 0:
                    new_logp_gen = new_logp_gen[:, :min_gen_len]
                    gen_mask = gen_mask[:, :min_gen_len]
                A_expanded = A_b.unsqueeze(1).expand(-1, min_gen_len) if min_gen_len > 0 else A_b.unsqueeze(1)
                weighted_logp = A_expanded * gen_mask * new_logp_gen if min_gen_len > 0 else torch.zeros_like(A_expanded)
                loss_b = -weighted_logp.sum() / (G * L_max_b)
                mb_loss_terms.append(loss_b)
            mb_loss = torch.stack(mb_loss_terms).mean() if mb_loss_terms else torch.tensor(0.0, device=sequences.device)
            scale = B_mb / B_U if B_U > 0 else 1.0
            (mb_loss * scale).backward()
            total_loss += mb_loss.item() * B_mb
            num_microbatches += 1
        optimizer.step()
        avg_loss = total_loss / B_U if B_U > 0 else 0.0
        self.logger.info(f"RL-aligned optimizer step completed: avg_loss = {avg_loss:.6f}, {num_microbatches} microbatches")

    def entropy_change_two_batch_rl(self, model: torch.nn.Module, E_batch: Dict[str, Any],
                                   U_batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
                                   cfg_importance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute ground-truth entropy change using RL-aligned two-batch approach.
        
        This fixes the fundamental misalignment by using the same GRPO RL objective
        as the probe's Y-loss and DR-GRPO training, instead of NLL loss.
        
        Args:
            model: Current model
            E_batch: Evaluation batch for entropy measurement
            U_batch: Update batch for RL-aligned optimizer step
            optimizer: Optimizer to use for update step
            cfg_importance: Configuration dict with RL-specific parameters
                
        Returns:
            Dictionary with aligned ground-truth entropy change results
        """
        start_time = time.time()
        self.logger.info("Starting RL-aligned two-batch ground-truth entropy change computation")
        
        # Extract config parameters
        training_loss = cfg_importance.get('training_loss', 'rl')
        rl_grad_accum = cfg_importance.get('rl_grad_accum', 1)
        importance_mb_size = cfg_importance.get('importance_microbatch_size', 1)
        # Only SNIS is supported (clip option removed)
        report_per_token = cfg_importance.get('report_per_token', False)
        snapshot_device = cfg_importance.get('snapshot_device', 'cpu')
        
        if training_loss != 'rl':
            self.logger.warning(f"RL-aligned method called with training_loss='{training_loss}', forcing to 'rl'")
            training_loss = 'rl'
        
        # ====================================================================
        # STEP A: Snapshot model and optimizer state
        # ====================================================================
        cpu_snaps, opt_state_snapshot = self._snapshot_model_optimizer(model, optimizer, snapshot_device)
        
        # ====================================================================
        # STEP B: Compute original entropy H(θ;E)
        # ====================================================================
        S_orig, RB_orig = self._eval_S_and_RB_on_E(E_batch)
        rb_sum_local = RB_orig.double().sum()
        cnt_local = torch.tensor(RB_orig.numel(), device=rb_sum_local.device, dtype=rb_sum_local.dtype)
        if dist.is_initialized():
            dist.all_reduce(rb_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
        H_orig = (rb_sum_local / cnt_local).item() if cnt_local.item() > 0 else 0.0
        H_orig_tok = None
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch).to(RB_orig.device)
            L_sum_local = lengths.double().sum()
            if dist.is_initialized():
                dist.all_reduce(L_sum_local, op=dist.ReduceOp.SUM)
            H_orig_tok = (rb_sum_local / L_sum_local).item() if L_sum_local.item() > 0 else 0.0
        self.logger.info(f"Original entropy (RB): H(I,;E) = {H_orig:.6f}")
        
        # ====================================================================
        # STEP C: Take RL-aligned optimizer step on U batch
        # ====================================================================
        self._rl_update_streaming(U_batch, optimizer, rl_grad_accum, importance_mb_size)
        
        # ====================================================================
        # STEP D: Compute updated entropy H(θ⁺;E) via importance sampling
        # ====================================================================
        S_upd, RB_upd = self._eval_S_and_RB_on_E(E_batch)
        
        # Compute IS weights: logw = S_upd - S_orig (in log domain)
        logw = S_upd - S_orig  # [B_E, G]
        
        # Apply SNIS estimator with RB payload
        is_results = self._compute_snis_two_batch(RB_upd, logw, report_per_token, E_batch)
        
        H_upd = is_results['H_upd']
        H_upd_tok = is_results.get('H_upd_tok')
        
        self.logger.info(f"Updated entropy (RB, SNIS): H(I,�;E) = {H_upd:.6f}")
        
        # ====================================================================
        # STEP E: Restore model and optimizer state
        # ====================================================================
        self._restore_model_optimizer(model, optimizer, cpu_snaps, opt_state_snapshot)
        
        # ====================================================================
        # STEP F: Compute ground-truth entropy change
        # ====================================================================
        deltaH_true = H_upd - H_orig
        deltaH_true_tok = (H_upd_tok - H_orig_tok) if H_upd_tok is not None and H_orig_tok is not None else None
        
        compute_time = time.time() - start_time
        
        self.logger.info(f"RL-aligned ground-truth entropy change: ΔH_true = {deltaH_true:.10f}")
        if deltaH_true_tok is not None:
            self.logger.info(f"Per-token entropy change: ΔH_true_tok = {deltaH_true_tok:.10f}")
        
        # Return results
        results = {
            'H_orig': H_orig,
            'H_upd': H_upd,
            'deltaH_true': deltaH_true,
            'timing': {
                'total_time': compute_time
            },
            'diagnostics': {
                'is_mode': 'snis_rb',
                'training_loss': training_loss,
                'rl_grad_accum': rl_grad_accum,
                **is_results.get('diagnostics', {})
            }
        }
        
        # Add per-token results if computed
        if deltaH_true_tok is not None:
            results.update({
                'H_orig_tok': H_orig_tok,
                'H_upd_tok': H_upd_tok,
                'deltaH_true_tok': deltaH_true_tok
            })
        
        return results





