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
        # AMP settings for TF and RL update paths
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
        logw_max = logw.max()
        w_shift = torch.exp(logw - logw_max)
        w_sum_local = w_shift.sum()
        w_payload_sum_local = (w_shift * payload_upd).sum()
        w_sq_sum_local = (w_shift ** 2).sum()
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
        ESS = (w_sum_global ** 2) / w_sq_sum_global if w_sq_sum_global != 0 else 0.0
        results: Dict[str, Any] = {
            'H_upd': H_upd,
            'diagnostics': {
                'ESS': ESS,
                'w_max': w_shift.max().item(),
                'w_min': w_shift.min().item(),
                'w_sum_global': w_sum_global,
            }
        }
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)
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

    def _eval_S_and_RB_on_E(self, E_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sequence_processor is None:
            S = self._eval_logprobs_on_batch(E_batch, self.config.get('importance', {}).get('importance_microbatch_size', 1))
            RB = torch.zeros_like(S)
            return S, RB
        bs = self._to_batched_sequences_from_probe(E_batch)
        logprob_results, _diag = self.sequence_processor.teacher_force_logprobs_with_diagnostics(
            sequences=bs,
            with_grad=False,
            tf_batch_size=getattr(self.sequence_processor.config, 'tf_batch_size', None),
            compute_rb=True,
        )
        device = next(self.model.parameters()).device
        seq_lp_list: list[list[float]] = []
        for b_list in logprob_results.sequence_logprobs:
            seq_lp_list.append([float(x) for x in b_list])
        S = torch.tensor(seq_lp_list, device=device, dtype=torch.float32)
        RB_vals: list[list[float]] = []
        if getattr(logprob_results, 'rb_entropies_torch', None):
            for b in range(len(logprob_results.rb_entropies_torch)):
                row = []
                for g in range(len(logprob_results.rb_entropies_torch[b])):
                    rb_t = logprob_results.rb_entropies_torch[b][g]
                    row.append(float(rb_t.detach().sum().item()) if rb_t is not None and rb_t.numel() > 0 else 0.0)
                RB_vals.append(row)
        else:
            for b in range(len(logprob_results.rb_entropies)):
                row = []
                for g in range(len(logprob_results.rb_entropies[b])):
                    rb_np = logprob_results.rb_entropies[b][g]
                    row.append(float(rb_np.sum()) if rb_np is not None else 0.0)
                RB_vals.append(row)
        RB = torch.tensor(RB_vals, device=device, dtype=torch.float32)
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
    ) -> None:
        self.logger.debug("Taking RL-aligned optimizer step on U batch")
        sequences = U_batch['sequences']
        prompt_lens = U_batch['prompt_lens']
        attention_masks = U_batch['attention_masks']
        advantages = U_batch['advantages']
        max_lengths = U_batch['max_lengths']
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
                prompt_len = int(mb_prompt_lens[b])
                A_b = mb_advantages[b]
                L_max_b = max(int(mb_max_lengths[b]), 1)
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
            scale = (B_mb / B_U) if B_U > 0 else 1.0
            (mb_loss * scale).backward()
            total_loss += mb_loss.item() * B_mb
            num_microbatches += 1
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

        # A) Snapshot
        cpu_snaps, opt_state_snapshot = self._snapshot_model_optimizer(model, optimizer, snapshot_device)

        # B) Original entropy on E (RB)
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
        self.logger.info(f"Original entropy (RB): H(I;E) = {H_orig:.6f}")

        # C) RL-aligned optimizer step on U
        self._rl_update_streaming(U_batch, optimizer, rl_grad_accum, importance_mb_size)

        # D) Updated entropy via SNIS with RB payload
        S_upd, RB_upd = self._eval_S_and_RB_on_E(E_batch)
        logw = S_upd - S_orig
        is_results = self._compute_snis_two_batch(RB_upd, logw, report_per_token, E_batch)
        H_upd = is_results['H_upd']
        H_upd_tok = is_results.get('H_upd_tok')
        self.logger.info(f"Updated entropy (RB, SNIS): H(I_updated;E) = {H_upd:.6f}")

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

