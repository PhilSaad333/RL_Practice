"""
DeltaEntropyIS: Importance-sampling based estimator for delta entropy using E/U batches.

This implementation supports:
- RL-aligned update on U (GRPO-style microbatched step)
- SNIS estimator for the updated entropy using a payload:
  - RB mode (recommended): payload = per-sequence RB entropy sums
  - Naive mode (fallback): payload = per-sequence log-prob sums (negative expectation)

Stage 2: Uses SequenceProcessor to evaluate S (log-prob sums) and RB (RB entropy sums)
on the SAME E batch before and after the update, and computes SNIS with DDP-safe reductions.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import distributed_helpers
from sequence_processing.sequence_processor import BatchedSequences


def _global_max_tensor(x: torch.Tensor) -> torch.Tensor:
    """DDP-safe global MAX for a scalar tensor (or any shape treated elementwise).

    On single GPU, returns x. On DDP, returns the all-reduced elementwise max.
    """
    if dist.is_available() and dist.is_initialized():
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.MAX)
        return y
    return x

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
        self.sequence_processor = sequence_processor  # preferred

        self.use_amp = config.get('memory_config', {}).get('amp', False)
        dtype_name = config.get('memory_config', {}).get('dtype', 'bfloat16')
        self.amp_dtype = getattr(torch, dtype_name, torch.bfloat16)

    # Public API (kept compatible with orchestrator)
    def entropy_change_two_batch(
        self,
        model: torch.nn.Module,
        E_batch: Dict[str, Any],
        U_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        cfg_importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.entropy_change_two_batch_rl(model, E_batch, U_batch, optimizer, cfg_importance)

    # ----------------------------
    # Helpers: lengths and snapshots
    # ----------------------------
    def _get_generation_lengths(self, batch: Dict[str, Any]) -> torch.Tensor:
        if 'gen_lengths' in batch:
            return batch['gen_lengths']
        sequences = batch['sequences']
        attention_masks = batch['attention_masks']
        prompt_lens = batch['prompt_lens']
        B, G, _ = sequences.shape
        gen_lengths = torch.zeros(B, G, dtype=torch.long, device=sequences.device)
        for b in range(B):
            pl = int(prompt_lens[b])
            for g in range(G):
                gen_lengths[b, g] = attention_masks[b, g, pl:].long().sum()
        return gen_lengths

    def _snapshot_model_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        snapshot_device: str = 'cpu'
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        self.logger.debug("Snapshotting model and optimizer state")
        cpu_snaps: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            cpu_snaps[name] = param.detach().to(snapshot_device).clone()
        opt_state_snapshot: Optional[Dict] = None
        if hasattr(optimizer, 'state_dict'):
            opt_state_snapshot = {}
            opt_state_dict = optimizer.state_dict()
            for key, value in opt_state_dict.items():
                if isinstance(value, dict):
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

    def _restore_model_optimizer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cpu_snaps: Dict[str, torch.Tensor],
        opt_state_snapshot: Optional[Dict]
    ) -> None:
        self.logger.debug("Restoring model and optimizer state")
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(cpu_snaps[name].to(param.device))
        if opt_state_snapshot is not None:
            try:
                restored_state: Dict[str, Any] = {}
                for key, value in opt_state_snapshot.items():
                    if isinstance(value, dict):
                        restored_state[key] = {}
                        for param_key, param_value in value.items():
                            if isinstance(param_value, torch.Tensor):
                                restored_state[key][param_key] = param_value.to(optimizer.param_groups[0]['params'][0].device)
                            else:
                                restored_state[key][param_key] = param_value
                    else:
                        restored_state[key] = value
                optimizer.load_state_dict(restored_state)
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")
        optimizer.zero_grad(set_to_none=True)

    # ----------------------------
    # SNIS estimator with generic payload (RB or log-probs)
    # ----------------------------
    def _compute_snis_two_batch(
        self,
        payload_upd: torch.Tensor,
        logw: torch.Tensor,
        report_per_token: bool,
        E_batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        is_dist, rank, world_size = distributed_helpers.get_dist_info()
        
        # Global max coordination for numerical stability (DDP-safe)
        logw_max = _global_max_tensor(logw.max())
        
        # STAGE 1 FIX: Float64 precision for weight arithmetic
        logw64 = (logw - logw_max).to(torch.float64)
        w_shift = torch.exp(logw64)
        
        # STAGE 1 FIX: All weight computations in float64
        w_sum_local = w_shift.sum()
        w_payload_sum_local = (w_shift * payload_upd.to(torch.float64)).sum()
        w_sq_sum_local = (w_shift * w_shift).sum()
        
        if is_dist:
            w_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sum_local)
            w_payload_sum_global = distributed_helpers.all_reduce_scalar_sum(w_payload_sum_local)
            w_sq_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sq_sum_local)
            self.logger.debug(f"Rank {rank}: local w_sum={w_sum_local.item():.6f}, global w_sum={w_sum_global:.6f}")
        else:
            w_sum_global = float(w_sum_local.item())
            w_payload_sum_global = float(w_payload_sum_local.item())
            w_sq_sum_global = float(w_sq_sum_local.item())
        entropy_mode = (self.config.get('importance', {}) or {}).get('entropy_mode', 'rb')
        if entropy_mode == 'rb':
            H_upd = (w_payload_sum_global / w_sum_global) if w_sum_global != 0 else 0.0
        else:
            H_upd = (-w_payload_sum_global / w_sum_global) if w_sum_global != 0 else 0.0

        # STAGE 1 FIX: Enhanced diagnostics for numerical health monitoring
        ESS = (w_sum_global ** 2) / w_sq_sum_global if w_sq_sum_global != 0 else 0.0
        N_total = logw.numel()
        ESS_fraction = ESS / max(float(N_total), 1.0)
        
        # Log warning if ESS is critically low
        if ESS_fraction < 0.05:
            self.logger.warning(f"⚠️ Critical: ESS fraction = {ESS_fraction:.2%} < 5%. "
                              f"Importance sampling may be unreliable. "
                              f"Consider reducing step size or using use_is=False.")
        elif ESS_fraction < 0.10:
            self.logger.warning(f"⚠️ Low ESS fraction = {ESS_fraction:.2%}. "
                              f"Results may have high variance.")
        
        results: Dict[str, Any] = {
            'H_upd': H_upd,
            'diagnostics': {
                'ESS': ESS,
                'ESS_fraction': ESS_fraction,
                'N_total': N_total,
                'logw_max_global': float(logw_max.item()),
                'logw_mean': float(logw.mean().item()),
                'logw_std': float(logw.std().item()) if logw.numel() > 1 else 0.0,
                'w_max': w_shift.max().item(),
                'w_min': w_shift.min().item(),
                'w_sum_global': w_sum_global,
            }
        }
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch).to(torch.float64)
            wL_sum_local = (w_shift * lengths).sum()
            if is_dist:
                wL_sum_global = distributed_helpers.all_reduce_scalar_sum(wL_sum_local)
            else:
                wL_sum_global = float(wL_sum_local.item())
            if entropy_mode == 'rb':
                H_upd_tok = (w_payload_sum_global / wL_sum_global) if wL_sum_global > 0 else 0.0
            else:
                H_upd_tok = (-w_payload_sum_global / wL_sum_global) if wL_sum_global > 0 else 0.0
            results['H_upd_tok'] = H_upd_tok
        return results

    # ----------------------------
    # SequenceProcessor adapters/evals
    # ----------------------------
    def _to_batched_sequences_from_probe(self, batch: Dict[str, Any]) -> BatchedSequences:
        sequences = batch['sequences']
        attention_masks = batch['attention_masks']
        prompt_lens = batch['prompt_lens']
        B, G, T = sequences.shape
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

    def _eval_S_and_RB_on_E(self, E_batch: Dict[str, Any], use_q_measure: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run in eval mode to disable dropout/batchnorm noise during TF/RB evaluation
        was_training = self.model.training
        self.model.eval()
        try:
            if self.sequence_processor is None:
                S = self._eval_logprobs_on_batch(
                    E_batch, self.config.get('importance', {}).get('importance_microbatch_size', 1)
                )
                RB = torch.zeros_like(S)
                return S, RB
            bs = self._to_batched_sequences_from_probe(E_batch)
            logprob_results, diagnostics_results = self.sequence_processor.teacher_force_logprobs_with_diagnostics(
                sequences=bs,
                with_grad=False,
                tf_batch_size=getattr(self.sequence_processor.config, 'tf_batch_size', None),
                compute_rb=True,
            )
        finally:
            # Restore original training/eval state
            self.model.train(was_training)
        device = next(self.model.parameters()).device
        
        # Stage 2: Choose between p and q measures
        if use_q_measure and logprob_results.sequence_logqs is not None:
            seq_lp_list: list[list[float]] = []
            for b_list in logprob_results.sequence_logqs:
                seq_lp_list.append([float(x) for x in b_list])
            self.logger.debug("Using q (sampling) measure for importance weights")
        else:
            seq_lp_list: list[list[float]] = []
            for b_list in logprob_results.sequence_logprobs:
                seq_lp_list.append([float(x) for x in b_list])
            if use_q_measure:
                self.logger.warning("Requested q measure but sequence_logqs not available, falling back to p")
        S = torch.tensor(seq_lp_list, device=device, dtype=torch.float32)
        RB_vals: list[list[float]] = []
        have_torch_rb = bool(getattr(logprob_results, 'rb_entropies_torch', None))
        have_np_rb = bool(getattr(logprob_results, 'rb_entropies', None))
        if have_torch_rb:
            for b in range(len(logprob_results.rb_entropies_torch)):
                row = []
                for g in range(len(logprob_results.rb_entropies_torch[b])):
                    rb_t = logprob_results.rb_entropies_torch[b][g]
                    row.append(float(rb_t.detach().sum().item()) if rb_t is not None and rb_t.numel() > 0 else 0.0)
                RB_vals.append(row)
        elif have_np_rb:
            for b in range(len(logprob_results.rb_entropies)):
                row = []
                for g in range(len(logprob_results.rb_entropies[b])):
                    rb_np = logprob_results.rb_entropies[b][g]
                    row.append(float(rb_np.sum()) if rb_np is not None else 0.0)
                RB_vals.append(row)
        else:
            # Fallback to diagnostics packs (SequenceDiagnostics.rb_entropy_sum)
            diag = getattr(diagnostics_results, 'diagnostics', None)
            if diag is not None:
                for b in range(len(diag)):
                    row = []
                    for g in range(len(diag[b])):
                        seq_diag = getattr(diag[b][g], 'seq', None)
                        rb_sum = float(getattr(seq_diag, 'rb_entropy_sum', 0.0)) if seq_diag is not None else 0.0
                        row.append(rb_sum)
                    RB_vals.append(row)
            else:
                # As a last resort, zero RB (should not happen if compute_rb=True)
                RB_vals = [[0.0 for _ in range(S.shape[1])] for _ in range(S.shape[0])]
        RB = torch.tensor(RB_vals, device=device, dtype=torch.float32)
        # Debug summary to ensure payloads are non-zero
        try:
            lengths = self._get_generation_lengths(E_batch).to(device)
            self.logger.info(
                f"[RB-DEBUG] S_sum={S.sum().item():.4f}, S_mean={S.mean().item():.4f}, "
                f"RB_sum={RB.sum().item():.4f}, RB_mean={RB.mean().item():.4f}, "
                f"len_mean={lengths.float().mean().item():.2f}"
            )
        except Exception:
            pass
        return S, RB

    # ----------------------------
    # RL-aligned update (GRPO)
    # ----------------------------

    def _rl_update_streaming(
        self,
        U_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        rl_grad_accum: int,
        importance_mb_size: int,
        *,
        ref_model=None,                  # <- optional, for KL parity
    ) -> None:
        self.logger.debug("Taking RL-aligned optimizer step on U batch")

        sequences = U_batch['sequences']          # [B_U, G, max_len]
        prompt_lens = U_batch['prompt_lens']      # [B_U]
        attention_masks = U_batch['attention_masks']  # [B_U, G, max_len]
        advantages = U_batch['advantages']        # [B_U, G]
        max_lengths = U_batch['max_lengths']      # [B_U]
        B_U, G, max_len = sequences.shape

        # --- training-mode and grads ---
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        # --- shared config knobs used in training ---
        temp = float(self.config.get("generation", {}).get("temperature", 1.0))  # parity with training
        use_clipping = "clip_eps" in self.config
        clip_eps_neg = float(self.config.get("clip_eps", 0.0))
        clip_eps_pos = float(self.config.get("clip_+", self.config.get("clip_eps", 0.0)))
        kl_beta = float(self.config.get("kl_beta", 0.0))
        do_kl = (kl_beta > 0.0 and ref_model is not None)

        # === 1) Compute old_logp ONCE at the snapshot (pre-update), PPO-style ===
        old_logp_full = None
        if use_clipping:
            with torch.no_grad():
                all_new_logp = []
                for start_b in range(0, B_U, importance_mb_size):
                    end_b = min(start_b + importance_mb_size, B_U)
                    mb_seqs = sequences[start_b:end_b]       # [B_mb, G, L]
                    mb_masks = attention_masks[start_b:end_b] # [B_mb, G, L]

                    flat_seqs = mb_seqs.view(-1, max_len)
                    flat_masks = mb_masks.view(-1, max_len)
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        logits = self.model(flat_seqs, attention_mask=flat_masks).logits
                    # temperature + float32 log-softmax
                    logits = (logits / temp).float()
                    logp_all = F.log_softmax(logits, dim=-1)
                    targets = flat_seqs[:, 1:].unsqueeze(-1)
                    logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)
                    logp = logp.view(end_b - start_b, G, -1)
                    all_new_logp.append(logp)
                old_logp_full = torch.cat(all_new_logp, dim=0)  # [B_U, G, L-1]
                # sanitize & clamp like training
                old_logp_full = torch.nan_to_num(old_logp_full, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)

        # === 2) Microbatch loop (identical structure) ===
        total_loss = 0.0
        num_microbatches = 0
        for start_b in range(0, B_U, importance_mb_size):
            end_b = min(start_b + importance_mb_size, B_U)
            B_mb = end_b - start_b

            mb_seqs = sequences[start_b:end_b]
            mb_masks = attention_masks[start_b:end_b]
            mb_adv = advantages[start_b:end_b]
            mb_Lmax = max_lengths[start_b:end_b]
            mb_prompt = prompt_lens[start_b:end_b]

            flat_seqs = mb_seqs.view(-1, max_len)
            flat_masks = mb_masks.view(-1, max_len)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(flat_seqs, attention_mask=flat_masks).logits

            # parity with training: temperature then float32 log-softmax
            logits = (logits / max(temp, 1e-8)).float()
            logp_all = F.log_softmax(logits, dim=-1)
            targets = flat_seqs[:, 1:].unsqueeze(-1)
            new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)
            new_logp = new_logp.view(B_mb, G, -1)
            # sanitize/clamp
            new_logp = torch.nan_to_num(new_logp, neginf=-80.0, posinf=0.0).clamp(min=-80.0, max=0.0)

            mb_loss_terms = []
            for b in range(B_mb):
                prompt_len = int(mb_prompt[b])
                L_max_b = max(int(mb_Lmax[b]), 1)
                gen_start = prompt_len - 1
                lp_gen = new_logp[b, :, gen_start:]                       # (G, Tg_b)
                mask_gen = mb_masks[b, :, prompt_len:].float()            # (G, Tg_b)
                if lp_gen.numel() == 0 or mask_gen.numel() == 0:
                    continue
                Tg = min(lp_gen.shape[1], mask_gen.shape[1])
                lp_gen = lp_gen[:, :Tg]
                mask_gen = mask_gen[:, :Tg]

                if use_clipping:
                    # Build old_logp for same slice
                    old_lp_gen = old_logp_full[start_b + b, :, gen_start:gen_start+Tg]  # (G, Tg)
                    ratios = torch.exp((lp_gen - old_lp_gen).clamp(-80, 80)) * mask_gen
                    adv_b = mb_adv[b].unsqueeze(-1)                                   # (G,1)
                    surr1 = ratios * adv_b
                    surr2 = torch.clamp(ratios, 1 - clip_eps_neg, 1 + clip_eps_pos) * adv_b
                    token_loss = -torch.min(surr1, surr2) * mask_gen                  # (G, Tg)
                    loss_b = token_loss.sum() / (G * L_max_b + 1e-8)
                    # Optional differentiable KL, same estimator as training
                    if do_kl:
                        with torch.no_grad():
                            # build the same flat view (B_mb*G, L) for ref forward if needed; or reuse above logits with ref model
                            pass  # (left as extension if you wire in ref_model and its tokenizer)
                else:
                    adv_exp = mb_adv[b].unsqueeze(1).expand(-1, Tg)                   # (G, Tg)
                    weighted_logp = adv_exp * mask_gen * lp_gen
                    loss_b = -weighted_logp.sum() / (G * L_max_b + 1e-8)

                mb_loss_terms.append(loss_b)

            mb_loss = torch.stack(mb_loss_terms).mean() if mb_loss_terms else torch.tensor(0.0, device=sequences.device)
            scale = (B_mb / B_U) if B_U > 0 else 1.0
            (mb_loss * scale).backward()
            total_loss += mb_loss.item() * B_mb
            num_microbatches += 1

        # --- gradient clipping parity with training ---
        max_norm = float(self.config.get("max_grad_norm", 0.0))
        if max_norm and max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm)

        optimizer.step()
        avg_loss = total_loss / B_U if B_U > 0 else 0.0
        self.logger.info(f"RL-aligned optimizer step completed: avg_loss = {avg_loss:.6f}, {num_microbatches} microbatches")


    # ----------------------------
    # Local fallback evaluator (naive S only)
    # ----------------------------
    def _eval_logprobs_on_batch(self, batch: Dict[str, Any], mb_size: int) -> torch.Tensor:
        sequences = batch['sequences']
        attention_masks = batch['attention_masks']
        prompt_lens = batch['prompt_lens']
        B, G, max_len = sequences.shape
        S_list: list[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for b in range(B):
                seqs_b = sequences[b]
                masks_b = attention_masks[b]
                prompt_len = int(prompt_lens[b])
                chunks: list[torch.Tensor] = []
                for g_start in range(0, G, mb_size):
                    g_end = min(g_start + mb_size, G)
                    micro_seqs = seqs_b[g_start:g_end]
                    micro_masks = masks_b[g_start:g_end]
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                            logits = self.model(micro_seqs, attention_mask=micro_masks).logits
                        logits = logits.float()
                        logp = F.log_softmax(logits, dim=-1)
                        targets = micro_seqs[:, 1:].unsqueeze(-1)
                        token_logp = logp[:, :-1].gather(-1, targets).squeeze(-1)
                        gen_start = max(prompt_len - 1, 0)
                        gen_token_logp = token_logp[:, gen_start:]
                        gen_mask = micro_masks[:, prompt_len:].float()
                        min_len = min(gen_token_logp.shape[1], gen_mask.shape[1]) if gen_token_logp.shape[1] > 0 else 0
                        if min_len > 0:
                            gen_token_logp = gen_token_logp[:, :min_len]
                            gen_mask = gen_mask[:, :min_len]
                            seq_logp = (gen_token_logp * gen_mask).sum(dim=1)
                        else:
                            seq_logp = torch.zeros(gen_token_logp.shape[0], device=logits.device)
                        chunks.append(seq_logp)
                S_b = torch.cat(chunks, dim=0) if chunks else torch.zeros(G, device=sequences.device)
                S_list.append(S_b)
        finally:
            self.model.train(was_training)
        return torch.stack(S_list, dim=0)

    # ----------------------------
    # RL + SNIS (RB payload)
    # ----------------------------
    def entropy_change_two_batch_rl(
        self,
        model: torch.nn.Module,
        E_batch: Dict[str, Any],
        U_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        cfg_importance: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.info("Starting RL-aligned two-batch ground-truth entropy change computation")
        # Config (RL + SNIS only)
        training_loss = 'rl'
        rl_grad_accum = cfg_importance.get('rl_grad_accum', 1)
        importance_mb_size = cfg_importance.get('importance_microbatch_size', 1)
        report_per_token = cfg_importance.get('report_per_token', False)
        snapshot_device = cfg_importance.get('snapshot_device', 'cpu')

        # Stage 2: Choose measure for weights (p vs q)
        # - If explicitly configured, honor it.
        # - Otherwise, auto-select q if either (top_p < 1) OR (temperature != 1).
        use_q = False
        if 'measure' in cfg_importance:
            measure = cfg_importance.get('measure', 'p')
            use_q = (measure == 'q')
        elif hasattr(self, 'sequence_processor') and self.sequence_processor is not None and hasattr(self.sequence_processor, 'config'):
            sp_cfg = self.sequence_processor.config
            if getattr(sp_cfg, 'top_p', 1.0) < 1.0:
                use_q = True
                self.logger.info(f"Auto-selecting q measure due to top_p={sp_cfg.top_p}")
            else:
                temp_val = float(getattr(sp_cfg, 'temperature', 1.0) or 1.0)
                if abs(temp_val - 1.0) > 1e-8:
                    use_q = True
                    self.logger.info(f"Auto-selecting q measure due to temperature={temp_val}")
        
        # A) Snapshot
        cpu_snaps, opt_state_snapshot = self._snapshot_model_optimizer(model, optimizer, snapshot_device)

        # B) Original entropy on E (RB)
        S_orig, RB_orig = self._eval_S_and_RB_on_E(E_batch, use_q_measure=use_q)
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
        self.logger.info(f"Original entropy (RB): H(I;E) = {H_orig:.6f}")

        # C) RL-aligned optimizer step on U
        self._rl_update_streaming(U_batch, optimizer, rl_grad_accum, importance_mb_size)

        # D) Updated entropy via SNIS with RB payload
        S_upd, RB_upd = self._eval_S_and_RB_on_E(E_batch, use_q_measure=use_q)
        logw = S_upd - S_orig
        is_results = self._compute_snis_two_batch(RB_upd, logw, report_per_token, E_batch)
        H_upd = is_results['H_upd']
        H_upd_tok = is_results.get('H_upd_tok')
        self.logger.info(f"Updated entropy (RB, SNIS): H(I_updated;E) = {H_upd:.6f}")
        
        # STAGE 1 FIX: Log enhanced diagnostics for monitoring numerical health
        diags = is_results.get('diagnostics', {})
        if 'ESS_fraction' in diags:
            self.logger.info(f"[SNIS Diagnostics] ESS = {diags['ESS']:.1f}/{diags['N_total']} "
                           f"({diags['ESS_fraction']:.1%}), "
                           f"logw_max = {diags['logw_max_global']:.3f}, "
                           f"logw_mean = {diags['logw_mean']:.3f}")
        
        # Store intermediate results for detailed logging if needed
        if hasattr(self, '_store_importance_details') and self._store_importance_details:
            self._importance_details = {
                'S_orig': S_orig,
                'S_upd': S_upd, 
                'RB_orig': RB_orig,
                'RB_upd': RB_upd,
                'logw': logw
            }

        # E) Restore
        self._restore_model_optimizer(model, optimizer, cpu_snaps, opt_state_snapshot)

        # F) Delta
        deltaH_true = H_upd - H_orig
        deltaH_true_tok = (H_upd_tok - H_orig_tok) if (H_upd_tok is not None and H_orig_tok is not None) else None
        compute_time = time.time() - start_time
        self.logger.info(f"RL-aligned ground-truth delta entropy: deltaH_true = {deltaH_true:.10f}")

        results: Dict[str, Any] = {
            'H_orig': H_orig,
            'H_upd': H_upd,
            'deltaH_true': deltaH_true,
            'timing': {
                'total_time': compute_time,
            },
            'diagnostics': {
                'is_mode': 'snis_rb',
                'training_loss': training_loss,
                'rl_grad_accum': rl_grad_accum,
                **is_results.get('diagnostics', {}),
            },
        }
        if deltaH_true_tok is not None:
            results.update({
                'H_orig_tok': H_orig_tok,
                'H_upd_tok': H_upd_tok,
                'deltaH_true_tok': deltaH_true_tok,
            })
        return results
