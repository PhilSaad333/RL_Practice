"""
Somewhat poorly named file.

The idea is that we want to compute the actual entropy change ΔH via importance sampling, which \delta H_1 is
supposed to approximate. 

To do a proper comparison, we need to:
    1) estimate the entropy of the original model H(π_θ) on the evaluation batch E
    2) take a real optimizer step on the update batch U
    3) estimate the entropy of the updated model H(π_{θ+δθ}) on the SAME evaluation batch E
    4) compute ΔH = H(π_{θ+δθ}) - H(π_θ)

The third step requires using importance sampling if we want to use the same batch E.

I'm not 100% sure if this it theoretically the right approach, but it seems like the most reasonable way to do it.
If we used a different batch E' for the updated model, then we would be measuring something different - we would get noise
from the variance of the entropy estimates on different batches, even as the learning rate goes to zero. The quantity we are 
measuring here should in principle behave linearly in the learning rate for small learning rates.



"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import math
from . import distributed_helpers


class ImportanceSampler:
    """
    Computes actual entropy change via self-normalized importance sampling.
    
    This measures ΔH = H(π_{θ+δθ}) - H(π_θ) by using importance sampling to estimate
    the entropy of the updated model without requiring new response generation.
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = next(model.parameters()).device
        
        # Importance sampling configuration
        self.use_snis = config['importance_sampling']['use_snis']
        self.use_psis = config['importance_sampling']['use_psis'] 
        self.ess_threshold = config['importance_sampling']['ess_threshold']
        self.resample_on_low_ess = config['importance_sampling']['resample_on_low_ess']
        
        # AMP settings
        self.use_amp = config['memory_config']['amp']
        self.amp_dtype = getattr(torch, config['memory_config']['dtype'])
        
        self.logger.info(f"ImportanceSampler initialized: use_snis={self.use_snis}, ess_threshold={self.ess_threshold}")
        
    def compute_entropy_change(self, batch_data: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """
        Compute actual entropy change ΔH via importance sampling.
        
        UPDATED (P4): Takes optimizer to perform real step on same sequences.
        
        Args:
            batch_data: Batch from ProbeComponents.sample_batch() 
            optimizer: Optimizer to use for taking a real step
            
        Returns:
            Dictionary containing:
            - deltaH_snis: Estimated entropy change
            - ESS: Effective sample size
            - psis_k: Pareto-smoothed IS diagnostic (if enabled)
            - timing: Computation timing
            - diagnostics: Additional diagnostic information
        """
        start_time = time.time()
        
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B]
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        B, G, max_len = sequences.shape
        
        self.logger.info(f"Computing entropy change via importance sampling for {B}x{G} responses")
        
        # Step 1: Compute original model log probabilities π_θ once
        logprobs_original = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
        
        # Step 2: Take optimizer step and compute updated model log probabilities π_{θ+δθ}
        # Pass cached S_orig to avoid redundant recomputation
        logprobs_updated, _ = self._compute_updated_logprobs(batch_data, optimizer, cached_S_orig=logprobs_original)
        
        # Step 3: Compute importance weights
        log_weights = logprobs_updated - logprobs_original  # [B, G]
        
        # 🔍 IMPORTANCE SAMPLING DEBUG  
        self.logger.info(f"🔍 [IS] logprobs_original: mean={logprobs_original.mean().item():.3f}, min={logprobs_original.min().item():.3f}, max={logprobs_original.max().item():.3f}")
        self.logger.info(f"🔍 [IS] logprobs_updated: mean={logprobs_updated.mean().item():.3f}, min={logprobs_updated.min().item():.3f}, max={logprobs_updated.max().item():.3f}")
        self.logger.info(f"🔍 [IS] log_weights: mean={log_weights.mean().item():.3f}, min={log_weights.min().item():.3f}, max={log_weights.max().item():.3f}")
        self.logger.info(f"🔍 [IS] Expected: logprobs should be NEGATIVE (log probabilities), updated should be LARGER (better model)")
        
        # Step 3: Apply importance sampling
        if self.use_snis:
            snis_results = self._compute_snis_entropy(logprobs_updated, log_weights)
            entropy_results = snis_results
        else:
            # Standard importance sampling (less stable)
            is_results = self._compute_is_entropy(logprobs_updated, log_weights)
            entropy_results = is_results
            
        # Step 4: Compute original model entropy
        original_entropy = self._compute_original_entropy(logprobs_original)
        
        # Step 5: Compute entropy change
        updated_entropy = entropy_results['entropy_estimate']
        delta_h = updated_entropy - original_entropy
        
        # 🔍 ENTROPY CHANGE DEBUG
        self.logger.info(f"🔍 [ENTROPY] original_entropy={original_entropy:.3f}")
        self.logger.info(f"🔍 [ENTROPY] updated_entropy={updated_entropy:.3f}")  
        self.logger.info(f"🔍 [ENTROPY] delta_h={delta_h:.3f}")
        self.logger.info(f"🔍 [ENTROPY] Expected: Entropy should DECREASE (negative delta_h) during training")
        
        # 🔍 MAGNITUDE CHECK
        seq_length_est = sequences.shape[2] - torch.tensor(prompt_lens).float().mean()
        expected_entropy_magnitude = seq_length_est * 0.25  # ~0.25 per token
        self.logger.info(f"🔍 [MAGNITUDE] Avg sequence length ≈ {seq_length_est:.1f}, expected total entropy ≈ {expected_entropy_magnitude:.1f}")
        self.logger.info(f"🔍 [MAGNITUDE] Observed entropy magnitudes: orig={abs(original_entropy):.1f}, updated={abs(updated_entropy):.1f}")
        if abs(delta_h) > expected_entropy_magnitude:
            self.logger.warning(f"🔍 [MAGNITUDE] WARNING: |delta_h|={abs(delta_h):.1f} >> expected magnitude {expected_entropy_magnitude:.1f}")
        
        # Step 6: Apply PSIS if requested
        psis_results = {}
        if self.use_psis:
            psis_results = self._apply_psis_smoothing(log_weights)
            
        # Step 7: Check ESS and resample if needed
        ess = entropy_results.get('ESS', 0.0)
        resample_flag = False
        if self.resample_on_low_ess and ess < self.ess_threshold * (B * G):
            self.logger.warning(f"Low ESS ({ess:.2f} < {self.ess_threshold * B * G:.2f}) - flagging for resample")
            resample_flag = True
            
        return {
            "deltaH_snis": delta_h,
            "ESS": ess,
            "psis_k": psis_results.get('k_hat', None),
            "original_entropy": original_entropy,
            "updated_entropy": updated_entropy,
            "timing": {
                "importance_sampling_time": time.time() - start_time
            },
            "diagnostics": {
                "B": B,
                "G": G,
                "total_samples": B * G,
                "method": "snis" if self.use_snis else "is",
                "resample_needed": resample_flag,
                "log_weight_mean": log_weights.mean().item(),
                "log_weight_std": log_weights.std().item(),
                "log_weight_min": log_weights.min().item(), 
                "log_weight_max": log_weights.max().item(),
                **entropy_results.get('diagnostics', {}),
                **psis_results
            }
        }
        
    def _compute_updated_logprobs(self, batch_data: Dict[str, Any], optimizer: torch.optim.Optimizer, cached_S_orig: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probabilities under the updated model π_{θ+δθ} after a real optimizer step.
        
        CRITICAL FIX (P4): Implement actual importance sampling on same sequences.
        This makes a copy of the model, applies one optimizer step, then recomputes
        log-probs on the SAME sequences for SNIS.
        
        Args:
            batch_data: Batch data containing sequences
            optimizer: Optimizer to take a step with
            cached_S_orig: Optional pre-computed original log probabilities to avoid recomputation
            
        Returns:
            Tuple of (updated log probabilities S⁺, original log probabilities S_orig)
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B] 
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        
        B, G, max_len = sequences.shape
        
        # STEP 1: Use cached S_orig if provided, otherwise compute original log-probs
        if cached_S_orig is not None:
            original_logprobs = cached_S_orig
            self.logger.debug("Using cached original log-probs to avoid redundant computation")
        else:
            original_logprobs = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
            self.logger.debug("Computing original log-probs (no cache provided)")
        
        # STEP 2: Take training step using streaming backward (avoids massive computation graph)
        # Snapshot parameters to CPU for proper restore after AdamW step  
        cpu_snapshots = []
        for param in self.model.parameters():
            if param.requires_grad:
                cpu_snapshots.append(param.detach().to('cpu').clone())
        
        # Snapshot optimizer state to prevent drift in training state
        opt_state_cpu = optimizer.state_dict()  # Shallow copy of Python objects
        # Deep-copy tensors to CPU to be safe
        for state in opt_state_cpu.get('state', {}).values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.detach().cpu().clone()
        
        self.logger.debug("Taking optimizer step for importance sampling with streaming backward (params & optimizer state saved)")
        
        # STEP 3: Accumulate gradients using streaming backward (no memory buildup)
        optimizer.zero_grad(set_to_none=True)
        total_tokens = self._accumulate_grads_for_importance(batch_data)
        
        # Scale gradients by 1/total_tokens to match mean loss semantics
        if total_tokens > 0:
            scale = 1.0 / total_tokens
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale)
        
        # Take optimizer step (θ → θ⁺)
        optimizer.step()
        
        # STEP 4: Recompute log-probs S⁺ on SAME sequences under updated model θ⁺
        updated_logprobs = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
        
        # STEP 5: Restore original parameters from CPU snapshots  
        # This properly handles AdamW's complex parameter updates
        with torch.no_grad():
            param_idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data.copy_(cpu_snapshots[param_idx].to(param.device))
                    param_idx += 1
        
        # Restore optimizer state to prevent training state drift
        optimizer.load_state_dict(opt_state_cpu)
        # Move any tensors in optimizer state back to correct device
        device = next(self.model.parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        optimizer.zero_grad(set_to_none=True)
                    
        self.logger.debug("Restored original model parameters and optimizer state after importance sampling step")
        
        return updated_logprobs, original_logprobs
        
    def _compute_logprobs_for_sequences(self, sequences: torch.Tensor, 
                                      prompt_lens: List[int], 
                                      attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log probabilities for given tokenized sequences with microbatching.
        
        Args:
            sequences: [B, G, max_len] tokenized sequences
            prompt_lens: [B] prompt lengths
            attention_masks: [B, G, max_len] attention masks
            
        Returns:
            [B, G] sequence log probabilities
        """
        B, G, max_len = sequences.shape
        all_logprobs = []
        
        # Use importance-specific microbatch size to avoid OOM during importance sampling
        importance_microbatch_size = self.config.get('memory_config', {}).get('importance_microbatch_size', 2)
        self.logger.debug(f"Using importance_microbatch_size={importance_microbatch_size} for memory efficiency")
        
        for b in range(B):
            batch_seqs = sequences[b]  # [G, max_len]
            prompt_len = prompt_lens[b]
            batch_masks = attention_masks[b]  # [G, max_len]
            
            # Microbatch the G sequences for this prompt to avoid OOM
            prompt_logprobs = []
            for g_start in range(0, G, importance_microbatch_size):
                g_end = min(g_start + importance_microbatch_size, G)
                micro_seqs = batch_seqs[g_start:g_end]  # [micro_size, max_len]
                micro_masks = batch_masks[g_start:g_end]  # [micro_size, max_len]
                
                # Teacher forcing pass for log probability computation (no gradients needed)
                was_training = self.model.training
                self.model.eval()  # Use eval mode for inference
                
                try:
                    with torch.no_grad():  # CRITICAL: No gradients needed - saves ~50% memory
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                            logits = self.model(micro_seqs, attention_mask=micro_masks).logits  # [micro_size, max_len, vocab_size]
                    
                    # Convert to log probabilities in float32 for stability (like dr_grpo.py)
                    log_probs = F.log_softmax(logits.float(), dim=-1)  # [micro_size, max_len, vocab_size]
                    
                    # Extract log probs of actual tokens (causal shift)
                    target_ids = micro_seqs[:, 1:].unsqueeze(-1)  # [micro_size, max_len-1, 1]
                    token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [micro_size, max_len-1]
                    
                    # Sum over generation tokens only (excluding prompt)
                    gen_start = prompt_len - 1  # -1 for causal shift
                    gen_token_log_probs = token_log_probs[:, gen_start:]  # [micro_size, gen_len]
                    
                    # Create mask for real generation tokens
                    gen_mask = micro_masks[:, prompt_len:].float()  # [micro_size, gen_len]
                    if gen_mask.shape[1] > gen_token_log_probs.shape[1]:
                        gen_mask = gen_mask[:, :gen_token_log_probs.shape[1]]
                    elif gen_mask.shape[1] < gen_token_log_probs.shape[1]:
                        gen_token_log_probs = gen_token_log_probs[:, :gen_mask.shape[1]]
                    
                    # Compute sequence log probabilities
                    seq_log_probs = (gen_token_log_probs * gen_mask).sum(dim=1)  # [micro_size]
                    prompt_logprobs.append(seq_log_probs)
                    
                finally:
                    self.model.train(was_training)
            
            # Concatenate microbatched results for this prompt
            all_seq_logprobs = torch.cat(prompt_logprobs, dim=0)  # [G]
            all_logprobs.append(all_seq_logprobs)
        
        return torch.stack(all_logprobs, dim=0)  # [B, G]
        
    def _accumulate_grads_for_importance(self, batch_data: Dict[str, Any]) -> int:
        """
        Accumulate gradients for importance sampling using streaming backward.
        
        This replaces _build_training_loss to avoid building massive computation graphs.
        Each microbatch calls backward() immediately to free memory.
        
        Returns:
            total_tokens: Number of generation tokens for gradient scaling
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B]
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        
        B, G, max_len = sequences.shape
        total_tokens = 0
        
        # Use microbatching to process sequences in small chunks
        importance_microbatch_size = self.config.get('memory_config', {}).get('importance_microbatch_size', 1)
        
        # Ensure model is in training mode for gradients  
        self.model.train()
        
        # Disable gradient checkpointing for entropy probe - needed for clean autograd graphs
        # if hasattr(self.model, 'gradient_checkpointing_enable'):
        #     self.model.gradient_checkpointing_enable()
        
        for b in range(B):
            batch_seqs = sequences[b]  # [G, max_len]
            prompt_len = prompt_lens[b]
            batch_masks = attention_masks[b]  # [G, max_len]
            
            # Microbatch the G sequences to avoid OOM during gradient computation
            for g_start in range(0, G, importance_microbatch_size):
                g_end = min(g_start + importance_microbatch_size, G)
                micro_seqs = batch_seqs[g_start:g_end].to(self.device)  # [micro_size, max_len]
                micro_masks = batch_masks[g_start:g_end].to(self.device)  # [micro_size, max_len]
                
                # Forward pass with gradients
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(micro_seqs, attention_mask=micro_masks).logits
                
                temp = self.config['generation'].get('temperature', 1.0)
                if temp != 1.0:
                    logits = logits / temp

                # Convert to float32 for numerical stability (like dr_grpo.py)
                logits = logits.float()
                
                # Compute cross-entropy loss on generation tokens only
                for micro_g in range(logits.shape[0]):
                    g = g_start + micro_g  # Global sequence index
                    seq = micro_seqs[micro_g]  # [max_len]
                    seq_logits = logits[micro_g]  # [max_len, vocab_size]
                    seq_mask = micro_masks[micro_g]  # [max_len]
                    
                    # Focus on generation tokens
                    gen_targets = seq[prompt_len:]  # [gen_len]
                    gen_logits = seq_logits[prompt_len-1:-1]  # [gen_len, vocab_size] (causal shift)
                    gen_mask = seq_mask[prompt_len:].float()  # [gen_len]
                    
                    if gen_targets.shape[0] > 0 and gen_logits.shape[0] > 0:
                        min_len = min(gen_targets.shape[0], gen_logits.shape[0])
                        gen_targets = gen_targets[:min_len]
                        gen_logits = gen_logits[:min_len]
                        gen_mask = gen_mask[:min_len]
                        
                        # Compute loss (logits already in float32 for numerical stability)
                        losses = F.cross_entropy(gen_logits, gen_targets, reduction='none')
                        masked_loss = (losses * gen_mask).sum()
                        
                        # Count tokens for final scaling
                        tokens_in_seq = gen_mask.sum()
                        total_tokens += int(tokens_in_seq.item())
                        
                        # 🔥 CRITICAL: Call backward() IMMEDIATELY to free computation graph
                        if tokens_in_seq > 0:
                            masked_loss.backward()
                
                # Free memory immediately after each microbatch
                del logits, micro_seqs, micro_masks
                torch.cuda.empty_cache()
        
        return total_tokens
        
    def _compute_snis_entropy(self, logprobs: torch.Tensor, log_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Compute entropy via self-normalized importance sampling.
        
        SNIS estimator: Ĥ = -Σ_i w_i * log(π(x_i)) / Σ_i w_i
        where w_i = exp(log_weight_i)
        """
        # Convert to weights (stable)
        log_weights_shifted = log_weights - log_weights.max()  # Numerical stability
        weights = torch.exp(log_weights_shifted)  # [B, G]
        
        # Flatten for computation
        logprobs_flat = logprobs.flatten()  # [B*G]
        weights_flat = weights.flatten()  # [B*G]
        
        # SNIS entropy estimate
        weighted_logprobs = weights_flat * logprobs_flat
        entropy_estimate = -weighted_logprobs.sum() / weights_flat.sum()
        
        # Effective sample size
        sum_weights = weights_flat.sum()
        sum_weights_sq = (weights_flat ** 2).sum()
        ess = sum_weights ** 2 / sum_weights_sq
        
        return {
            "entropy_estimate": entropy_estimate.item(),
            "ESS": ess.item(),
            "diagnostics": {
                "sum_weights": sum_weights.item(),
                "max_weight": weights_flat.max().item(),
                "min_weight": weights_flat.min().item(),
                "weight_variance": weights_flat.var().item()
            }
        }
        
    def _compute_is_entropy(self, logprobs: torch.Tensor, log_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Compute entropy via standard importance sampling (less stable than SNIS).
        
        IS estimator: Ĥ = -Σ_i w_i * log(π(x_i)) / N
        """
        log_weights_shifted = log_weights - log_weights.max()
        weights = torch.exp(log_weights_shifted)
        
        logprobs_flat = logprobs.flatten()
        weights_flat = weights.flatten()
        
        weighted_logprobs = weights_flat * logprobs_flat
        entropy_estimate = -weighted_logprobs.mean()
        
        # ESS computation  
        ess = len(weights_flat) / (1 + weights_flat.var() / weights_flat.mean() ** 2)
        
        return {
            "entropy_estimate": entropy_estimate.item(), 
            "ESS": ess.item(),
            "diagnostics": {
                "method": "standard_is"
            }
        }
        
    def _compute_original_entropy(self, logprobs: torch.Tensor) -> float:
        """Compute entropy of original model π_θ with proper DDP all-reduce."""
        # For the original model, entropy is just -E[log π(x)]
        # Use global mean in multi-GPU settings
        
        # Local tensors
        neg_sum_local = (-logprobs).sum()
        cnt_local = torch.tensor(logprobs.numel(), device=logprobs.device, dtype=neg_sum_local.dtype)
        
        # All-reduce across ranks if DDP is active
        if dist.is_initialized():
            dist.all_reduce(neg_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
            
        # Global mean
        original_entropy = (neg_sum_local / cnt_local).item()
        return original_entropy
        
    def _apply_psis_smoothing(self, log_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Apply Pareto-smoothed importance sampling.
        
        This is a simplified version - full PSIS requires fitting Pareto distribution
        to the tail of the importance weights.
        """
        # Flatten weights
        log_weights_flat = log_weights.flatten()
        weights_flat = torch.exp(log_weights_flat - log_weights_flat.max())
        
        # Sort weights descending
        sorted_weights, _ = torch.sort(weights_flat, descending=True)
        
        # Estimate Pareto parameter k from top 20% of weights
        M = len(sorted_weights)
        top_20_pct = max(1, M // 5)
        top_weights = sorted_weights[:top_20_pct]
        
        if len(top_weights) > 1:
            # Rough estimate of k parameter
            log_top = torch.log(top_weights)
            k_hat = -1.0 / (log_top.mean() - log_top.max())
        else:
            k_hat = 0.0
            
        # k > 0.7 indicates problematic importance sampling
        is_problematic = k_hat > 0.7
        
        return {
            "k_hat": k_hat.item() if isinstance(k_hat, torch.Tensor) else k_hat,
            "psis_problematic": is_problematic,
            "top_weight_fraction": top_20_pct / M
        }
        
    def validate_importance_sampling(self, batch_data: Dict[str, Any]) -> bool:
        """
        Validate that importance sampling setup is working correctly.
        
        Returns:
            True if validation passes
        """
        try:
            # Test computation without errors
            results = self.compute_entropy_change(batch_data)
            
            # Check for NaN/inf results
            if math.isnan(results['deltaH_snis']) or math.isinf(results['deltaH_snis']):
                self.logger.error("Importance sampling produced NaN/inf entropy change")
                return False
                
            # Check ESS is reasonable
            ess = results['ESS']
            total_samples = results['diagnostics']['total_samples']
            if ess <= 0 or ess > total_samples:
                self.logger.error(f"Invalid ESS: {ess} (should be in (0, {total_samples}])")
                return False
                
            self.logger.info("Importance sampling validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Importance sampling validation failed: {e}")
            return False
    
    # ========================================================================
    # STAGE 2: Two-Batch Ground-Truth Entropy Change
    # ========================================================================
    
    def entropy_change_two_batch(self, model: torch.nn.Module, E_batch: Dict[str, Any], 
                                U_batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
                                cfg_importance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute ground-truth entropy change using two-batch approach.
        
        Process:
        1. Snapshot model state
        2. Compute H(θ;E) with original model
        3. Take one optimizer step on U batch  
        4. Compute H(θ⁺;E) with updated model using importance sampling
        5. Restore original model state
        6. Return ΔH_true = H(θ⁺;E) - H(θ;E)
        
        Args:
            model: Current model  
            E_batch: Evaluation batch for entropy measurement
            U_batch: Update batch for optimizer step
            optimizer: Optimizer to use for update step
            cfg_importance: Configuration dict with keys:
                - training_loss: "nll" or "rl" 
                - importance_microbatch_size: microbatch size for processing
                - is_mode: "snis" or "clip" 
                - clip_c: clipping constant for IS (if using clipped IS)
                - report_per_token: whether to compute per-token metrics
                - snapshot_device: "cpu" or "gpu" for model snapshots
                
        Returns:
            Dictionary with ground-truth entropy change results
        """
        # DISPATCH: Route to RL-aligned implementation when training_loss='rl'
        training_loss = cfg_importance.get('training_loss', 'nll')
        if training_loss == 'rl':
            self.logger.info("Routing to RL-aligned two-batch entropy change computation")
            return self.entropy_change_two_batch_rl(model, E_batch, U_batch, optimizer, cfg_importance)
        
        start_time = time.time()
        self.logger.info("Starting two-batch ground-truth entropy change computation (NLL mode)")
        
        # Extract config parameters
        importance_mb_size = cfg_importance.get('importance_microbatch_size', 1)
        is_mode = cfg_importance.get('is_mode', 'snis')
        clip_c = cfg_importance.get('clip_c', 10.0)
        report_per_token = cfg_importance.get('report_per_token', False)
        snapshot_device = cfg_importance.get('snapshot_device', 'cpu')
        
        # ====================================================================
        # STEP A: Snapshot model and optimizer state
        # ====================================================================
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
        
        # ====================================================================
        # STEP B: Compute original entropy H(θ;E) 
        # ====================================================================
        self.logger.debug("Computing original entropy H(θ;E)")
        
        # Compute log probabilities for E batch with original model
        with torch.no_grad():
            model.eval()
            S_orig = self._compute_logprobs_microbatched(E_batch, importance_mb_size)
        
        # Compute sequence-level entropy with proper DDP all-reduce
        neg_sum_local = (-S_orig).sum()
        cnt_local = torch.tensor(S_orig.numel(), device=S_orig.device, dtype=neg_sum_local.dtype)
        
        if dist.is_initialized():
            dist.all_reduce(neg_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
            
        H_orig = (neg_sum_local / cnt_local).item()
        
        # Compute per-token entropy if requested
        H_orig_tok = None
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)
            total_tokens = lengths.sum().item() 
            H_orig_tok = -S_orig.sum().item() / total_tokens if total_tokens > 0 else 0.0
        
        self.logger.info(f"Original entropy: H(θ;E) = {H_orig:.6f}")
        
        # ====================================================================
        # STEP C: Take one optimizer step on U batch
        # ====================================================================
        self.logger.debug("Taking optimizer step on U batch")
        
        # Accumulate gradients from U batch using microbatched training loss
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        total_tokens = 0
        
        # Process U batch in microbatches
        for microbatch in self._iter_microbatches(U_batch, importance_mb_size):
            mb_loss, mb_tokens = self._compute_training_loss_microbatch(microbatch, training_loss)
            
            # Backward pass per microbatch
            mb_loss.backward()
            
            total_loss += mb_loss.item() * mb_tokens
            total_tokens += mb_tokens
        
        # Normalize gradients by total tokens for mean loss semantics
        if total_tokens > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(total_tokens)
        
        # Take optimizer step
        optimizer.step()
        
        self.logger.info(f"Optimizer step completed: avg_loss = {total_loss/total_tokens:.6f}")
        
        # ====================================================================
        # STEP D: Compute updated entropy H(θ⁺;E) via importance sampling
        # ====================================================================
        self.logger.debug("Computing updated entropy H(θ⁺;E) via importance sampling")
        
        # Compute log probabilities for E batch with updated model
        with torch.no_grad():
            model.eval()
            S_upd = self._compute_logprobs_microbatched(E_batch, importance_mb_size)
        
        # Compute IS weights: logw = S_upd - S_orig (in log domain)
        logw = S_upd - S_orig  # [batch_size, G]
        
        # Apply importance sampling estimator
        if is_mode == "snis":
            # Self-normalized importance sampling
            is_results = self._compute_snis_two_batch(S_upd, logw, report_per_token, E_batch)
        elif is_mode == "clip":
            # Clipped importance sampling  
            is_results = self._compute_clip_is_two_batch(S_upd, logw, clip_c, report_per_token, E_batch)
        else:
            raise ValueError(f"Unknown is_mode: {is_mode}")
        
        H_upd = is_results['H_upd']
        H_upd_tok = is_results.get('H_upd_tok')
        
        self.logger.info(f"Updated entropy: H(θ⁺;E) = {H_upd:.6f}")
        
        # ====================================================================
        # STEP F: Restore model and optimizer state
        # ====================================================================
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
        
        # ====================================================================
        # STEP G: Compute ground-truth entropy change
        # ====================================================================
        deltaH_true = H_upd - H_orig
        deltaH_true_tok = (H_upd_tok - H_orig_tok) if H_upd_tok is not None and H_orig_tok is not None else None
        
        compute_time = time.time() - start_time
        
        self.logger.info(f"Ground-truth entropy change: ΔH_true = {deltaH_true:.10f}")
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
                'is_mode': is_mode,
                'training_loss': training_loss,
                'total_tokens': total_tokens,
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
    
    def _compute_logprobs_microbatched(self, batch: Dict[str, Any], mb_size: int) -> torch.Tensor:
        """Compute log probabilities using microbatched forward passes."""
        sequences = batch['sequences']  # [batch_size, G, max_len]
        prompt_lens = batch['prompt_lens']  # [batch_size]
        attention_masks = batch['attention_masks']  # [batch_size, G, max_len]
        batch_size = len(sequences)
        
        all_logprobs = []
        
        # Process each prompt individually to bound memory  
        for b in range(batch_size):
            seq_b = sequences[b:b+1]  # [1, G, max_len] - keep batch dimension
            mask_b = attention_masks[b:b+1]  # [1, G, max_len]
            prompt_len = [prompt_lens[b]]  # [1] - make it a list
            
            # Compute logprobs for this single-prompt batch
            logprobs_b = self._compute_logprobs_for_sequences(seq_b, prompt_len, mask_b)
            all_logprobs.append(logprobs_b.squeeze(0))  # Remove batch dimension
        
        return torch.stack(all_logprobs, dim=0)  # [batch_size, G]
    
    def _compute_snis_two_batch(self, S_upd: torch.Tensor, logw: torch.Tensor,
                               report_per_token: bool, E_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Compute SNIS entropy estimate for two-batch method with multi-GPU support."""
        is_dist, rank, world_size = distributed_helpers.get_dist_info()
        
        # Stabilize weights in log domain
        logw_max = logw.max()
        w_shift = torch.exp(logw - logw_max)  # [batch_size, G]
        
        # Compute local sums
        w_sum_local = w_shift.sum()
        wS_sum_local = (w_shift * S_upd).sum()
        w_sq_sum_local = (w_shift**2).sum()  # For ESS computation
        
        # All-reduce sums across ranks for multi-GPU
        if is_dist:
            w_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sum_local)
            wS_sum_global = distributed_helpers.all_reduce_scalar_sum(wS_sum_local)
            w_sq_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sq_sum_local)
            
            self.logger.debug(f"Rank {rank}: local w_sum={w_sum_local.item():.6f}, global w_sum={w_sum_global:.6f}")
        else:
            w_sum_global = w_sum_local.item()
            wS_sum_global = wS_sum_local.item()
            w_sq_sum_global = w_sq_sum_local.item()
        
        # Compute SNIS entropy estimate from global sums
        H_upd = -wS_sum_global / w_sum_global if w_sum_global != 0 else 0.0
        
        # Compute diagnostics
        ESS = (w_sum_global**2) / w_sq_sum_global if w_sq_sum_global != 0 else 0.0
        
        results = {
            'H_upd': H_upd,
            'diagnostics': {
                'ESS': ESS,
                'w_max': w_shift.max().item(),
                'w_min': w_shift.min().item(),
                'w_sum_global': w_sum_global
            }
        }
        
        # Per-token estimate if requested
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)  # [batch_size, G]
            wL_sum_local = (w_shift * lengths).sum()
            
            if is_dist:
                wL_sum_global = distributed_helpers.all_reduce_scalar_sum(wL_sum_local)
            else:
                wL_sum_global = wL_sum_local.item()
                
            H_upd_tok = -wS_sum_global / wL_sum_global if wL_sum_global > 0 else 0.0
            results['H_upd_tok'] = H_upd_tok
        
        return results
    
    def _compute_clip_is_two_batch(self, S_upd: torch.Tensor, logw: torch.Tensor,
                                  clip_c: float, report_per_token: bool, E_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Compute clipped IS entropy estimate for two-batch method with multi-GPU support.""" 
        is_dist, rank, world_size = distributed_helpers.get_dist_info()
        
        # Convert to linear weights and clip
        w = torch.exp(logw)  # [batch_size, G]
        w_clipped = torch.clamp(w, max=clip_c)
        
        # Compute local sums
        w_sum_local = w_clipped.sum()
        wS_sum_local = (w_clipped * S_upd).sum()
        
        # All-reduce sums across ranks for multi-GPU
        if is_dist:
            w_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sum_local)
            wS_sum_global = distributed_helpers.all_reduce_scalar_sum(wS_sum_local)
        else:
            w_sum_global = w_sum_local.item()
            wS_sum_global = wS_sum_local.item()
        
        # Compute clipped IS entropy estimate from global sums
        H_upd = -wS_sum_global / w_sum_global if w_sum_global != 0 else 0.0
        
        # Compute diagnostics (using local values for metrics that don't need global reduction)
        w_sq_sum_local = (w_clipped**2).sum()
        if is_dist:
            w_sq_sum_global = distributed_helpers.all_reduce_scalar_sum(w_sq_sum_local)
            ESS = (w_sum_global**2) / w_sq_sum_global if w_sq_sum_global != 0 else 0.0
        else:
            ESS = (w_sum_global**2) / w_sq_sum_local.item() if w_sq_sum_local.item() != 0 else 0.0
        
        results = {
            'H_upd': H_upd,
            'diagnostics': {
                'ESS': ESS,
                'clipped_fraction': (w > clip_c).float().mean().item(),
                'w_max': w_clipped.max().item(),
                'w_sum_global': w_sum_global
            }
        }
        
        # Per-token estimate if requested
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)  # [batch_size, G]
            wL_sum_local = (w_clipped * lengths).sum()
            
            if is_dist:
                wL_sum_global = distributed_helpers.all_reduce_scalar_sum(wL_sum_local)
            else:
                wL_sum_global = wL_sum_local.item()
                
            H_upd_tok = -wS_sum_global / wL_sum_global if wL_sum_global > 0 else 0.0
            results['H_upd_tok'] = H_upd_tok
        
        return results
    
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
    
    def _compute_nll_loss_microbatch(self, microbatch: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """Compute NLL loss for a microbatch."""
        sequences = microbatch['sequences']  # [mb_size, G, max_len]  
        attention_masks = microbatch['attention_masks']  # [mb_size, G, max_len]
        prompt_lens = microbatch['prompt_lens']  # [mb_size]
        
        mb_size, G, max_len = sequences.shape
        total_loss = 0.0
        total_tokens = 0
        
        # Process each prompt in microbatch
        for b in range(mb_size):
            for g in range(G):
                seq = sequences[b, g]  # [max_len]
                mask = attention_masks[b, g]  # [max_len]
                prompt_len = prompt_lens[b]
                
                # Forward pass
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(seq.unsqueeze(0), attention_mask=mask.unsqueeze(0)).logits  # [1, max_len, vocab_size]
                
                # Compute NLL loss for generation tokens only
                gen_tokens = seq[prompt_len:]  # Generation tokens [gen_len]
                gen_mask = mask[prompt_len:]  # Mask for generation tokens [gen_len]
                
                # Only compute loss on actual generation tokens (not padding)
                valid_gen_tokens = gen_tokens[gen_mask.bool()]
                
                if len(valid_gen_tokens) > 0:
                    # Logits that predict generation tokens
                    logits_for_gen = logits[0, prompt_len-1:prompt_len-1+len(gen_tokens)]  # [gen_len, vocab_size]
                    # Only use logits corresponding to valid tokens
                    valid_logits = logits_for_gen[gen_mask.bool()]  # [valid_len, vocab_size]
                    
                    loss = F.cross_entropy(valid_logits, valid_gen_tokens, reduction='sum')
                    total_loss += loss
                    total_tokens += len(valid_gen_tokens)
        
        # Return mean loss for this microbatch and token count
        return total_loss / max(total_tokens, 1), total_tokens
    
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
    
    # ========================================================================
    # STAGE 3: RL-ALIGNED IMPORTANCE SAMPLING IMPLEMENTATION
    # ========================================================================
    
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
        is_mode = cfg_importance.get('is_mode', 'snis')
        clip_c = cfg_importance.get('clip_c', 10.0)
        report_per_token = cfg_importance.get('report_per_token', False)
        snapshot_device = cfg_importance.get('snapshot_device', 'cpu')
        
        if training_loss != 'rl':
            self.logger.warning(f"RL-aligned method called with training_loss='{training_loss}', forcing to 'rl'")
            training_loss = 'rl'
        
        # ====================================================================
        # STEP A: Snapshot model and optimizer state
        # ====================================================================
        model_snapshot = self._snapshot_state(model, optimizer, snapshot_device)
        
        # ====================================================================
        # STEP B: Compute original entropy H(θ;E)
        # ====================================================================
        S_orig = self._eval_logprobs_on_batch(E_batch, importance_mb_size)
        # Compute sequence-level entropy with proper DDP all-reduce
        neg_sum_local = (-S_orig).sum()
        cnt_local = torch.tensor(S_orig.numel(), device=S_orig.device, dtype=neg_sum_local.dtype)
        
        if dist.is_initialized():
            dist.all_reduce(neg_sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_local, op=dist.ReduceOp.SUM)
            
        H_orig = (neg_sum_local / cnt_local).item()
        
        # Per-token entropy if requested
        H_orig_tok = None
        if report_per_token:
            lengths = self._get_generation_lengths(E_batch)
            total_tokens = lengths.sum().item()
            H_orig_tok = -S_orig.sum().item() / total_tokens if total_tokens > 0 else 0.0
            
        self.logger.info(f"Original entropy: H(θ;E) = {H_orig:.6f}")
        
        # ====================================================================
        # STEP C: Take RL-aligned optimizer step on U batch
        # ====================================================================
        self._rl_update_streaming(U_batch, optimizer, rl_grad_accum, importance_mb_size)
        
        # ====================================================================
        # STEP D: Compute updated entropy H(θ⁺;E) via importance sampling
        # ====================================================================
        S_upd = self._eval_logprobs_on_batch(E_batch, importance_mb_size)
        
        # Compute IS weights: logw = S_upd - S_orig (in log domain)
        logw = S_upd - S_orig  # [B_E, G]
        
        # Apply importance sampling estimator
        if is_mode == "snis":
            is_results = self._compute_snis_two_batch(S_upd, logw, report_per_token, E_batch)
        elif is_mode == "clip":
            is_results = self._compute_clip_is_two_batch(S_upd, logw, clip_c, report_per_token, E_batch)
        else:
            raise ValueError(f"Unknown is_mode: {is_mode}")
        
        H_upd = is_results['H_upd']
        H_upd_tok = is_results.get('H_upd_tok')
        
        self.logger.info(f"Updated entropy: H(θ⁺;E) = {H_upd:.6f}")
        
        # ====================================================================
        # STEP E: Restore model and optimizer state
        # ====================================================================
        self._restore_state(model, optimizer, model_snapshot)
        
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
                'is_mode': is_mode,
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
    
    def _eval_logprobs_on_batch(self, batch: Dict[str, Any], mb_size: int) -> torch.Tensor:
        """
        Compute log probabilities for batch with eval mode, attention_mask, and AMP bf16.
        
        This replaces the existing _compute_logprobs_for_sequences with proper attention_mask
        support and alignment with DR-GRPO forward passes.
        
        Args:
            batch: Batch data with sequences, attention_masks, prompt_lens
            mb_size: Microbatch size for memory efficiency
            
        Returns:
            [B, G] sequence log probabilities (sum over generated tokens)
        """
        sequences = batch['sequences']  # [B, G, max_len]
        prompt_lens = batch['prompt_lens']  # [B]
        attention_masks = batch['attention_masks']  # [B, G, max_len]
        
        B, G, max_len = sequences.shape
        all_logprobs = []
        
        # Set eval mode and use no_grad for inference
        was_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                for b in range(B):
                    batch_seqs = sequences[b]  # [G, max_len]
                    prompt_len = prompt_lens[b]
                    batch_masks = attention_masks[b]  # [G, max_len]
                    
                    # Microbatch the G sequences for memory efficiency
                    prompt_logprobs = []
                    for g_start in range(0, G, mb_size):
                        g_end = min(g_start + mb_size, G)
                        micro_seqs = batch_seqs[g_start:g_end]  # [micro_size, max_len]
                        micro_masks = batch_masks[g_start:g_end]  # [micro_size, max_len]
                        
                        # Forward pass with attention_mask (aligned with DR-GRPO)
                        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                            logits = self.model(micro_seqs, attention_mask=micro_masks).logits
                        
                        # Convert to float32 for numerical stability (like DR-GRPO)
                        logits = logits.float()
                        log_probs = F.log_softmax(logits, dim=-1)  # [micro_size, max_len, vocab_size]
                        
                        # Extract log probs of actual tokens (causal shift)
                        target_ids = micro_seqs[:, 1:].unsqueeze(-1)  # [micro_size, max_len-1, 1]
                        token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [micro_size, max_len-1]
                        
                        # Sum over generation tokens only (align with DR-GRPO and probe)
                        gen_start = prompt_len - 1  # -1 for causal shift
                        gen_token_log_probs = token_log_probs[:, gen_start:]  # [micro_size, gen_len]
                        
                        # Create mask for generation tokens (exclude padding)
                        gen_mask = micro_masks[:, prompt_len:].float()  # [micro_size, gen_len]
                        if gen_mask.shape[1] > gen_token_log_probs.shape[1]:
                            gen_mask = gen_mask[:, :gen_token_log_probs.shape[1]]
                        elif gen_mask.shape[1] < gen_token_log_probs.shape[1]:
                            gen_token_log_probs = gen_token_log_probs[:, :gen_mask.shape[1]]
                        
                        # Compute sequence log probabilities (sum over generation tokens)
                        seq_log_probs = (gen_token_log_probs * gen_mask).sum(dim=1)  # [micro_size]
                        prompt_logprobs.append(seq_log_probs)
                    
                    # Concatenate microbatched results for this prompt
                    all_seq_logprobs = torch.cat(prompt_logprobs, dim=0)  # [G]
                    all_logprobs.append(all_seq_logprobs)
        finally:
            self.model.train(was_training)
        
        return torch.stack(all_logprobs, dim=0)  # [B, G]
    
    def _snapshot_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                       snapshot_device: str) -> Dict[str, Any]:
        """
        Snapshot model parameters and optimizer state to specified device.
        
        Args:
            model: Model to snapshot
            optimizer: Optimizer to snapshot 
            snapshot_device: Device to store snapshots ('cpu' or 'gpu')
            
        Returns:
            Dictionary containing model and optimizer snapshots
        """
        self.logger.debug(f"Snapshotting model and optimizer state to {snapshot_device}")
        
        # Snapshot model parameters
        model_snapshot = {}
        for name, param in model.named_parameters():
            model_snapshot[name] = param.detach().to(snapshot_device).clone()
        
        # Snapshot optimizer state (deep copy to handle exp_avg, exp_avg_sq, etc.)
        optimizer_snapshot = None
        try:
            import copy
            optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
            
            # Move tensors in optimizer state to snapshot device
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().to(snapshot_device).clone()
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(item) for item in obj]
                else:
                    return obj
            
            optimizer_snapshot = move_to_device(optimizer_snapshot)
            
        except Exception as e:
            self.logger.warning(f"Failed to snapshot optimizer state: {e}")
            optimizer_snapshot = None
        
        return {
            'model': model_snapshot,
            'optimizer': optimizer_snapshot,
            'snapshot_device': snapshot_device
        }
    
    def _restore_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      snapshot: Dict[str, Any]):
        """
        Restore model parameters and optimizer state from snapshot.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            snapshot: Snapshot dictionary from _snapshot_state
        """
        self.logger.debug("Restoring model and optimizer state from snapshot")
        
        # Restore model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in snapshot['model']:
                    param.data.copy_(snapshot['model'][name].to(param.device))
        
        # Restore optimizer state if available
        if snapshot['optimizer'] is not None:
            try:
                # Move optimizer state back to original device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [move_to_device(item, target_device) for item in obj]
                    else:
                        return obj
                
                # Get target device from first model parameter
                target_device = next(model.parameters()).device
                restored_opt_state = move_to_device(snapshot['optimizer'], target_device)
                optimizer.load_state_dict(restored_opt_state)
                
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")
        
        # Clear gradients for safety
        optimizer.zero_grad(set_to_none=True)
    
    def _rl_update_streaming(self, U_batch: Dict[str, Any], optimizer: torch.optim.Optimizer,
                           rl_grad_accum: int, importance_mb_size: int):
        """
        Perform RL-aligned update step using GRPO objective with gradient accumulation.
        
        This implements the same RL objective as probe Y-loss and DR-GRPO:
        Loss_mb = -mean_over_prompts[sum_{g,t}(A_b,g * gen_mask_b,g,t * new_logp_b,g,t) / (G * L_max_b)]
        
        Args:
            U_batch: Update batch with sequences, advantages, max_lengths
            optimizer: Optimizer for taking the step
            rl_grad_accum: Number of microbatches to accumulate before step
            importance_mb_size: Sequences per forward pass
        """
        self.logger.debug("Taking RL-aligned optimizer step on U batch")
        
        sequences = U_batch['sequences']  # [B_U, G, max_len]
        prompt_lens = U_batch['prompt_lens']  # [B_U]
        attention_masks = U_batch['attention_masks']  # [B_U, G, max_len]
        advantages = U_batch['advantages']  # [B_U, G]
        max_lengths = U_batch['max_lengths']  # [B_U]
        
        B_U, G, max_len = sequences.shape
        
        # Set training mode and clear gradients
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Process U batch in microbatches for memory efficiency
        mb_size_prompts = importance_mb_size
        total_loss = 0.0
        num_microbatches = 0
        
        for start_b in range(0, B_U, mb_size_prompts):
            end_b = min(start_b + mb_size_prompts, B_U)
            B_mb = end_b - start_b
            
            # Extract microbatch
            mb_seqs = sequences[start_b:end_b]  # [B_mb, G, max_len]
            mb_masks = attention_masks[start_b:end_b]  # [B_mb, G, max_len]
            mb_advantages = advantages[start_b:end_b]  # [B_mb, G]
            mb_max_lengths = max_lengths[start_b:end_b]  # [B_mb]
            mb_prompt_lens = prompt_lens[start_b:end_b]  # [B_mb]
            
            # Flatten sequences for model forward: [B_mb * G, max_len]
            flat_seqs = mb_seqs.view(-1, max_len)
            flat_masks = mb_masks.view(-1, max_len)
            
            # Forward pass with autocast and attention_mask (aligned with DR-GRPO)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(flat_seqs, attention_mask=flat_masks).logits  # [B_mb*G, max_len, vocab_size]
            
            # Convert to float32 for numerical stability (like DR-GRPO)
            logits = logits.float()
            logp_all = F.log_softmax(logits, dim=-1)  # [B_mb*G, max_len, vocab_size]
            
            # Gather next-token log probabilities
            targets = flat_seqs[:, 1:].unsqueeze(-1)  # [B_mb*G, max_len-1, 1]
            new_logp = logp_all[:, :-1].gather(-1, targets).squeeze(-1)  # [B_mb*G, max_len-1]
            
            # Reshape back to [B_mb, G, max_len-1]
            new_logp = new_logp.view(B_mb, G, -1)
            
            # Compute RL loss per prompt (token-level, GRPO normalization)
            mb_loss_terms = []
            for b in range(B_mb):
                prompt_len = mb_prompt_lens[b]
                A_b = mb_advantages[b]  # [G] - advantages for this prompt
                L_max_b = max(mb_max_lengths[b], 1)  # Avoid division by zero
                
                # Extract generation region: last T_g tokens
                gen_start = prompt_len - 1  # -1 for causal shift
                new_logp_gen = new_logp[b, :, gen_start:]  # [G, T_g]
                gen_mask = mb_masks[b, :, prompt_len:].float()  # [G, T_g]
                
                # Align tensor shapes
                min_gen_len = min(new_logp_gen.shape[1], gen_mask.shape[1])
                new_logp_gen = new_logp_gen[:, :min_gen_len]  # [G, min_gen_len]
                gen_mask = gen_mask[:, :min_gen_len]  # [G, min_gen_len]
                
                # Expand advantages to token level: [G, 1] -> [G, min_gen_len]
                A_expanded = A_b.unsqueeze(1).expand(-1, min_gen_len)  # [G, min_gen_len]
                
                # Compute loss for this prompt: -sum_{g,t}(A_b,g * gen_mask * new_logp) / (G * L_max_b)
                weighted_logp = A_expanded * gen_mask * new_logp_gen  # [G, min_gen_len]
                loss_b = -weighted_logp.sum() / (G * L_max_b)
                mb_loss_terms.append(loss_b)
            
            # Mean loss over prompts in microbatch
            mb_loss = torch.stack(mb_loss_terms).mean()
            
            # Scale for gradient accumulation (to match global mean)
            scale = B_mb / B_U  # This ensures proper gradient scaling
            scaled_loss = mb_loss * scale
            
            # Backward pass per microbatch
            scaled_loss.backward()
            
            total_loss += mb_loss.item() * B_mb  # For logging
            num_microbatches += 1
        
        # Take optimizer step after accumulating gradients
        optimizer.step()
        
        avg_loss = total_loss / B_U if B_U > 0 else 0.0
        self.logger.info(f"RL-aligned optimizer step completed: avg_loss = {avg_loss:.6f}, {num_microbatches} microbatches")
    
    def _iter_microbatches(self, batch: Dict[str, Any], mb_size: int):
        """Iterate over microbatches by slicing along prompt dimension."""
        batch_size = len(batch['sequences'])
        
        for start_idx in range(0, batch_size, mb_size):
            end_idx = min(start_idx + mb_size, batch_size)
            microbatch = {}
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    microbatch[key] = value[start_idx:end_idx]
                elif isinstance(value, (list, tuple)):
                    microbatch[key] = value[start_idx:end_idx]
                else:
                    microbatch[key] = value
            
            yield microbatch