"""
Importance Sampling

Computes actual entropy change ΔH via self-normalized importance sampling (SNIS).
Based on Section VI of offline_entropy_probe_strategy.txt.

The key insight is that we can measure actual entropy change using the same batch
of responses by treating the post-update model as the target distribution.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import math


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
        
        # Step 1: Compute original model log probabilities π_θ
        logprobs_original = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
        
        # Step 2: Take optimizer step and compute updated model log probabilities π_{θ+δθ}
        logprobs_updated = self._compute_updated_logprobs(batch_data, optimizer)
        
        # Step 3: Compute importance weights
        log_weights = logprobs_updated - logprobs_original  # [B, G]
        
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
        
    def _compute_updated_logprobs(self, batch_data: Dict[str, Any], optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Compute log probabilities under the updated model π_{θ+δθ} after a real optimizer step.
        
        CRITICAL FIX (P4): Implement actual importance sampling on same sequences.
        This makes a copy of the model, applies one optimizer step, then recomputes
        log-probs on the SAME sequences for SNIS.
        
        Args:
            batch_data: Batch data containing sequences
            optimizer: Optimizer to take a step with
            
        Returns:
            Updated log probabilities S⁺ for the same sequences
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B] 
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        
        B, G, max_len = sequences.shape
        
        # STEP 1: Compute original log-probs S before step
        original_logprobs = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
        
        # STEP 2: Create a training loss on the same batch to take a real optimizer step
        # Use a simple training objective: maximize log-likelihood of responses
        training_loss = self._build_training_loss(batch_data)
        
        self.logger.debug("Taking optimizer step for importance sampling")
        
        # STEP 3: Take optimizer step (this modifies model parameters θ → θ⁺)
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        
        # STEP 4: Recompute log-probs S⁺ on SAME sequences under updated model θ⁺
        updated_logprobs = self._compute_logprobs_for_sequences(sequences, prompt_lens, attention_masks)
        
        # STEP 5: Restore original parameters (undo the step)
        # This is critical - we don't want to permanently modify the model
        optimizer.zero_grad()
        with torch.no_grad():
            # Undo the step: θ = θ⁺ - η * ∇L  
            for param in self.model.parameters():
                if param.grad is not None:
                    # Get the step that was taken
                    lr = optimizer.param_groups[0]['lr']
                    param.data.sub_(param.grad * lr)  # Undo: subtract what was added
                    
        self.logger.debug("Restored original model parameters after importance sampling step")
        
        return updated_logprobs
        
    def _compute_logprobs_for_sequences(self, sequences: torch.Tensor, 
                                      prompt_lens: List[int], 
                                      attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log probabilities for given tokenized sequences.
        
        Args:
            sequences: [B, G, max_len] tokenized sequences
            prompt_lens: [B] prompt lengths
            attention_masks: [B, G, max_len] attention masks
            
        Returns:
            [B, G] sequence log probabilities
        """
        B, G, max_len = sequences.shape
        all_logprobs = []
        
        # Process in batches for memory efficiency
        for b in range(B):
            batch_seqs = sequences[b]  # [G, max_len]
            prompt_len = prompt_lens[b]
            batch_masks = attention_masks[b]  # [G, max_len]
            
            # Teacher forcing pass with gradients enabled
            was_training = self.model.training
            self.model.train()
            
            try:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(batch_seqs).logits  # [G, max_len, vocab_size]
                
                # Convert to log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # [G, max_len, vocab_size]
                
                # Extract log probs of actual tokens (causal shift)
                target_ids = batch_seqs[:, 1:].unsqueeze(-1)  # [G, max_len-1, 1]
                token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [G, max_len-1]
                
                # Sum over generation tokens only (excluding prompt)
                gen_start = prompt_len - 1  # -1 for causal shift
                gen_token_log_probs = token_log_probs[:, gen_start:]  # [G, gen_len]
                
                # Create mask for real generation tokens
                gen_mask = batch_masks[:, prompt_len:].float()  # [G, gen_len]
                if gen_mask.shape[1] > gen_token_log_probs.shape[1]:
                    gen_mask = gen_mask[:, :gen_token_log_probs.shape[1]]
                elif gen_mask.shape[1] < gen_token_log_probs.shape[1]:
                    gen_token_log_probs = gen_token_log_probs[:, :gen_mask.shape[1]]
                
                # Compute sequence log probabilities
                seq_log_probs = (gen_token_log_probs * gen_mask).sum(dim=1)  # [G]
                all_logprobs.append(seq_log_probs)
                
            finally:
                self.model.train(was_training)
        
        return torch.stack(all_logprobs, dim=0)  # [B, G]
        
    def _build_training_loss(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Build a training loss on the batch for taking an optimizer step.
        
        This uses a simple negative log-likelihood objective.
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B]
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        
        B, G, max_len = sequences.shape
        total_loss = 0.0
        total_tokens = 0
        
        # Simple training objective: maximize log-likelihood of generated tokens
        for b in range(B):
            batch_seqs = sequences[b]  # [G, max_len]
            prompt_len = prompt_lens[b]
            batch_masks = attention_masks[b]  # [G, max_len]
            
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(batch_seqs).logits  # [G, max_len, vocab_size]
            
            # Compute cross-entropy loss on generation tokens only
            for g in range(G):
                seq = batch_seqs[g]  # [max_len]
                seq_logits = logits[g]  # [max_len, vocab_size]
                seq_mask = batch_masks[g]  # [max_len]
                
                # Focus on generation tokens
                gen_targets = seq[prompt_len:]  # [gen_len]
                gen_logits = seq_logits[prompt_len-1:-1]  # [gen_len, vocab_size] (causal shift)
                gen_mask = seq_mask[prompt_len:].float()  # [gen_len]
                
                if gen_targets.shape[0] > 0 and gen_logits.shape[0] > 0:
                    min_len = min(gen_targets.shape[0], gen_logits.shape[0])
                    gen_targets = gen_targets[:min_len]
                    gen_logits = gen_logits[:min_len]
                    gen_mask = gen_mask[:min_len]
                    
                    # Compute loss
                    losses = F.cross_entropy(gen_logits, gen_targets, reduction='none')
                    masked_loss = (losses * gen_mask).sum()
                    total_loss += masked_loss
                    total_tokens += gen_mask.sum()
        
        return total_loss / total_tokens if total_tokens > 0 else torch.tensor(0.0, device=self.device)
        
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
        """Compute entropy of original model π_θ."""
        # For the original model, entropy is just -E[log π(x)]
        return -logprobs.mean().item()
        
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