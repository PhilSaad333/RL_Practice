# entropy_probe.py - First-order entropy change analysis for RL training
import torch
import torch.distributed as dist
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SequenceData:
    """Metadata and metrics for a single sequence."""
    prompt_idx: int              # Which prompt (0 to B-1)
    gen_idx: int                 # Which generation (0 to G-1) 
    global_idx: int              # Global sequence index (0 to B*G-1)
    log_prob: float              # S(t) = log π(t)
    advantage: float             # A(t) = advantage
    length: int                  # Sequence length (excluding padding)
    step_idx: int                # Training step when collected
    device_rank: int             # Which device this came from


class EntropyProbe:
    """
    First-order entropy change analysis probe.
    
    Implements the theoretical framework from the research paper:
    δH = -η * (1/(B*G)) * Σ_{i=1}^{B*G} * (1/L_max(p_i)) * 
         E_t[(S(t) - S̄) * K_1(t,t_i') * A(t_i')]
    
    Where:
    - K_1(t,t') = Σ_α ∂_α log π(t) ∂_α log π(t') P_α (Fisher kernel with Adam conditioning)
    - S(t) = log π(t) (log probabilities)  
    - A(t) = advantages
    - P_α = Adam conditioning factors (momentum/squared gradient)
    
    This implementation:
    - Stores gradients ∂_α log π(t) for all sequences in a step
    - Computes K_1(t,t') Fisher kernel matrix between all sequence pairs
    - Stores metadata for prompt grouping and sequence lookup
    - Single-device only (cross-device terms require gradient sharing)
    - Enables analysis of same-prompt vs different-prompt K_1 structure
    """
    
    def __init__(
        self,
        enabled: bool = True,
        debug: bool = False,
        max_sequences: int = 2000,  # Memory safety limit
        store_full_kernel: bool = True  # Whether to store full K_1 matrix
    ):
        """
        Args:
            enabled: Whether the probe is active
            debug: Whether to print debug information
            max_sequences: Maximum sequences to store (memory safety)
            store_full_kernel: Whether to store full K_1(t,t') matrix
        """
        self.enabled = enabled
        self.debug = debug
        self.max_sequences = max_sequences
        self.store_full_kernel = store_full_kernel
        
        # Storage for current step
        self.current_step_gradients: List[torch.Tensor] = []  # ∂_α log π(t) for each sequence
        self.current_step_sequences: List[SequenceData] = []  # Metadata for each sequence
        self.current_step_adam_conditioning: Optional[torch.Tensor] = None  # P_α factors
        
        # Storage for analysis
        self.fisher_kernel: Optional[torch.Tensor] = None  # K_1(t,t') matrix
        self.sequence_metadata: List[SequenceData] = []
        self.step_counter = 0
        
        # Computed metrics
        self.delta_h_estimate: Optional[float] = None
        self.mean_log_prob: Optional[float] = None
        
    def store_step_data(
        self,
        rollouts: Any,  # RolloutBatch 
        advantages: torch.Tensor,  # (B, G)
        log_probs: torch.Tensor,   # (B, G, T_gen) sequence log probabilities
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_idx: int,
        policy_model: torch.nn.Module,  # Need model for forward passes
        pad_id: int = 0
    ) -> None:
        """
        Store gradient and sequence data for one training step.
        
        Args:
            rollouts: RolloutBatch with sequence data
            advantages: Computed advantages (B, G)
            log_probs: Per-token log probabilities (B, G, T_gen)
            trainable_params: Model parameters with gradients
            optimizer: Adam optimizer (for conditioning factors)
            learning_rate: Current learning rate η
            step_idx: Current training step
        """
        if not self.enabled:
            return
            
        device_rank = dist.get_rank() if dist.is_initialized() else 0
        B, G, T_gen = rollouts.gen_ids.shape
        
        # Check memory safety
        total_sequences = B * G
        if len(self.current_step_sequences) + total_sequences > self.max_sequences:
            if self.debug:
                print(f"[EntropyProbe] Skipping step {step_idx}: would exceed max_sequences={self.max_sequences}")
            return
            
        # Collect per-sequence gradients (expensive but accurate)
        self.current_step_gradients = self._compute_per_sequence_gradients(
            rollouts, advantages, log_probs, trainable_params, policy_model, pad_id
        )
        
        # Collect sequence metadata
        self._collect_sequence_metadata(
            rollouts, advantages, log_probs, step_idx, device_rank, pad_id
        )
        
        # Store Adam conditioning factors
        self._collect_adam_conditioning(optimizer, trainable_params)
        
        # Compute Fisher kernel if requested
        if self.store_full_kernel and len(self.current_step_gradients) > 1:
            self._compute_fisher_kernel()
            
        # Estimate δH for this step
        self._estimate_delta_h(learning_rate, rollouts)
        
        self.step_counter += 1
        
        if self.debug:
            print(f"[EntropyProbe] Stored {total_sequences} sequences from step {step_idx}")
            
    def _compute_per_sequence_gradients(
        self,
        rollouts: Any,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        trainable_params: List[torch.nn.Parameter],
        policy_model: torch.nn.Module,
        pad_id: int
    ) -> List[torch.Tensor]:
        """
        Compute per-sequence gradients ∂_α log π(t) for each sequence.
        
        This is the expensive but accurate approach: separate forward/backward
        pass for each sequence to get exact per-sequence gradients.
        """
        B, G, T_gen = rollouts.gen_ids.shape
        gen_mask = (rollouts.gen_ids != pad_id).float()
        sequence_gradients = []
        
        # Save original optimizer state
        original_state = {param: param.grad.clone() if param.grad is not None else None 
                         for param in trainable_params}
        
        for b in range(B):
            for g in range(G):
                # Skip empty sequences
                if gen_mask[b, g].sum() == 0:
                    sequence_gradients.append(torch.zeros(0))
                    continue
                
                # Create single-sequence batch
                single_rollout = self._extract_single_sequence(rollouts, b, g)
                single_advantage = advantages[b:b+1, g:g+1]  # (1, 1)
                
                # Clear gradients
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.zero_()
                
                # Compute loss for this single sequence
                single_loss = self._compute_single_sequence_loss(
                    single_rollout, single_advantage, policy_model, pad_id
                )
                
                # Backward pass to get gradients
                single_loss.backward(retain_graph=True)
                
                # Collect gradients for this sequence
                grad_chunks = []
                for param in trainable_params:
                    if param.grad is not None:
                        grad_chunks.append(param.grad.detach().flatten().cpu())
                    else:
                        grad_chunks.append(torch.zeros(param.numel(), device='cpu'))
                
                sequence_grad = torch.cat(grad_chunks) if grad_chunks else torch.zeros(0)
                sequence_gradients.append(sequence_grad)
        
        # Restore original gradients
        for param, orig_grad in original_state.items():
            param.grad = orig_grad
            
        return sequence_gradients
        
    def _extract_single_sequence(self, rollouts: Any, b: int, g: int) -> Any:
        """Extract a single sequence (b,g) as a batch of size 1."""
        from .base import RolloutBatch  # Import here to avoid circular imports
        
        return RolloutBatch(
            prompt_ids=rollouts.prompt_ids[b:b+1],      # (1, T_p)
            gen_ids=rollouts.gen_ids[b:b+1, g:g+1],     # (1, 1, T_gen)
            reward=rollouts.reward[b:b+1, g:g+1],       # (1, 1)
            logprobs=rollouts.logprobs[b:b+1, g:g+1],   # (1, 1, T_gen)
            tag_correct=rollouts.tag_correct[b:b+1, g:g+1],  # (1, 1)
            think_len=rollouts.think_len[b:b+1, g:g+1]  # (1, 1)
        )
        
    def _compute_single_sequence_loss(
        self, 
        single_rollout: Any, 
        single_advantage: torch.Tensor,
        policy_model: torch.nn.Module,
        pad_id: int
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a single sequence.
        
        Implements: L = -A(t) * log π(t) where log π(t) is computed from the model.
        """
        B, G, T_g = single_rollout.gen_ids.shape  # Should be (1, 1, T_gen)
        T_p = single_rollout.prompt_ids.shape[1]
        
        # Build full sequence: prompt + generation
        prompt_rep = single_rollout.prompt_ids.unsqueeze(1).expand(-1, G, -1)  # (1, 1, T_p)
        full_seq = torch.cat((prompt_rep, single_rollout.gen_ids), dim=-1)  # (1, 1, T_total)
        
        # Prepare for model forward pass
        seq_ids = full_seq.view(1, -1)  # (1, T_total)
        device = seq_ids.device
        
        # Forward pass through model to get logits
        with torch.enable_grad():  # Ensure gradients are computed
            outputs = policy_model(seq_ids)
            logits = outputs.logits  # (1, T_total, vocab_size)
        
        # Compute log probabilities for generated tokens only
        # We only want gradients w.r.t. the generation part, not the prompt
        gen_start = T_p
        gen_end = T_p + T_g
        
        # Extract generation tokens and corresponding logits
        gen_tokens = seq_ids[:, gen_start:gen_end]  # (1, T_g)
        gen_logits = logits[:, gen_start-1:gen_end-1]  # (1, T_g, vocab_size) - shifted for causal modeling
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)  # (1, T_g, vocab_size)
        
        # Gather log probabilities for actual generated tokens
        token_log_probs = log_probs.gather(2, gen_tokens.unsqueeze(-1)).squeeze(-1)  # (1, T_g)
        
        # Mask out padding tokens
        gen_mask = (gen_tokens != pad_id).float()  # (1, T_g)
        masked_log_probs = token_log_probs * gen_mask  # (1, T_g)
        
        # Sum log probabilities for the sequence
        sequence_log_prob = masked_log_probs.sum()  # scalar
        
        # Apply Dr-GRPO normalization (1/L_max where L_max is length of this sequence)
        sequence_length = gen_mask.sum().clamp_min(1.0)
        normalized_log_prob = sequence_log_prob / sequence_length
        
        # GRPO loss: -A(t) * log π(t)
        loss = -single_advantage.squeeze() * normalized_log_prob
        
        return loss
        
    def _collect_sequence_metadata(
        self,
        rollouts: Any,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        step_idx: int,
        device_rank: int,
        pad_id: int
    ) -> None:
        """Collect metadata for each sequence."""
        B, G, T_gen = rollouts.gen_ids.shape
        gen_mask = (rollouts.gen_ids != pad_id).float()
        
        # Compute sequence-level log probabilities S(t)
        seq_log_probs = (log_probs * gen_mask).sum(dim=2)  # (B, G)
        
        self.current_step_sequences.clear()
        
        for b in range(B):
            for g in range(G):
                # Skip if sequence is empty (all padding)
                if gen_mask[b, g].sum() == 0:
                    continue
                    
                global_idx = len(self.current_step_sequences)
                
                # Store sequence metadata
                seq_data = SequenceData(
                    prompt_idx=b,
                    gen_idx=g,
                    global_idx=global_idx,
                    log_prob=float(seq_log_probs[b, g].item()),
                    advantage=float(advantages[b, g].item()),
                    length=int(gen_mask[b, g].sum().item()),
                    step_idx=step_idx,
                    device_rank=device_rank
                )
                
                self.current_step_sequences.append(seq_data)
        
    def _collect_adam_conditioning(
        self,
        optimizer: torch.optim.Optimizer,
        trainable_params: List[torch.nn.Parameter]
    ) -> None:
        """Collect Adam conditioning factors P_α."""
        # Extract momentum and squared gradient terms from Adam
        conditioning_factors = []
        
        for param in trainable_params:
            if param in optimizer.state:
                state = optimizer.state[param]
                
                # Adam conditioning: P_α involves momentum and squared gradients
                # Simplified version: use 1/(sqrt(v) + eps) where v is squared gradient
                if 'exp_avg_sq' in state:
                    v = state['exp_avg_sq']
                    eps = optimizer.param_groups[0].get('eps', 1e-8)
                    conditioning = 1.0 / (torch.sqrt(v) + eps)
                    conditioning_factors.append(conditioning.detach().flatten().cpu())
                else:
                    # Fallback: no conditioning (equivalent to SGD)
                    conditioning_factors.append(torch.ones_like(param.grad.detach().flatten().cpu()))
            else:
                # No optimizer state yet
                conditioning_factors.append(torch.ones_like(param.grad.detach().flatten().cpu()))
                
        self.current_step_adam_conditioning = torch.cat(conditioning_factors)
        
    def _compute_fisher_kernel(self) -> None:
        """Compute K_1(t,t') = Σ_α ∂_α log π(t) ∂_α log π(t') P_α."""
        if len(self.current_step_gradients) < 2:
            return
            
        n_sequences = len(self.current_step_gradients)
        
        # Stack gradients: (n_sequences, n_params)
        gradients = torch.stack(self.current_step_gradients)
        
        # Apply Adam conditioning element-wise
        if self.current_step_adam_conditioning is not None:
            conditioned_gradients = gradients * self.current_step_adam_conditioning.unsqueeze(0)
        else:
            conditioned_gradients = gradients
            
        # Compute Fisher kernel: K_1(t,t') = g_t^T P g_t'
        self.fisher_kernel = torch.mm(conditioned_gradients, conditioned_gradients.t())
        
        if self.debug:
            print(f"[EntropyProbe] Computed Fisher kernel: {self.fisher_kernel.shape}")
            print(f"[EntropyProbe] Diagonal elements (first 5): {self.fisher_kernel.diag()[:5]}")
            
    def _estimate_delta_h(self, learning_rate: float, rollouts: Any) -> None:
        """Estimate δH using the first-order formula with Dr-GRPO weighting."""
        if len(self.current_step_sequences) < 2 or self.fisher_kernel is None:
            return
            
        # Extract data for computation
        log_probs = [seq.log_prob for seq in self.current_step_sequences]
        advantages = [seq.advantage for seq in self.current_step_sequences]
        B, G, _ = rollouts.gen_ids.shape
        
        # Compute L_max for each prompt group
        prompt_max_lengths = {}
        for seq in self.current_step_sequences:
            prompt_idx = seq.prompt_idx
            if prompt_idx not in prompt_max_lengths:
                prompt_max_lengths[prompt_idx] = 0
            prompt_max_lengths[prompt_idx] = max(prompt_max_lengths[prompt_idx], seq.length)
        
        # Compute mean log probability S̄
        mean_log_prob = np.mean(log_probs)
        self.mean_log_prob = mean_log_prob
        
        # Compute δH estimate using the Dr-GRPO formula:
        # δH ≈ -η * (1/(B*G)) * Σ_i Σ_j (S(t_i) - S̄) * K_1(t_i,t_j) * A(t_j) * (1/L_max(p_j))
        delta_h = 0.0
        n_sequences = len(self.current_step_sequences)
        
        for i in range(n_sequences):
            for j in range(n_sequences):
                seq_j = self.current_step_sequences[j]
                l_max_j = prompt_max_lengths[seq_j.prompt_idx]
                
                s_centered = log_probs[i] - mean_log_prob
                kernel_val = float(self.fisher_kernel[i, j].item())
                advantage_j = advantages[j]
                
                # Apply Dr-GRPO weighting: advantages weighted by 1/L_max for their prompt
                weighted_advantage = advantage_j / l_max_j
                
                delta_h += s_centered * kernel_val * weighted_advantage
                
        delta_h *= -learning_rate / (B * G)
        self.delta_h_estimate = delta_h
        
        if self.debug:
            print(f"[EntropyProbe] Estimated δH = {delta_h:.6f} (Dr-GRPO weighted)")
            print(f"[EntropyProbe] Prompt max lengths: {prompt_max_lengths}")
            
    def get_metrics(self) -> Dict[str, float]:
        """Return computed metrics for logging."""
        metrics = {
            "entropy_probe_n_sequences": len(self.current_step_sequences),
            "entropy_probe_step_counter": self.step_counter,
        }
        
        if self.delta_h_estimate is not None:
            metrics["entropy_probe_delta_h"] = self.delta_h_estimate
            
        if self.mean_log_prob is not None:
            metrics["entropy_probe_mean_log_prob"] = self.mean_log_prob
            
        if self.fisher_kernel is not None:
            # Add Fisher kernel statistics
            diag_mean = float(self.fisher_kernel.diag().mean().item())
            off_diag = self.fisher_kernel - torch.diag(self.fisher_kernel.diag())
            off_diag_mean = float(off_diag.mean().item())
            
            metrics.update({
                "entropy_probe_fisher_diag_mean": diag_mean,
                "entropy_probe_fisher_off_diag_mean": off_diag_mean,
            })
            
        return metrics
        
    def analyze_prompt_structure(self) -> Dict[str, Any]:
        """Analyze K_1 structure: same-prompt vs different-prompt correlations."""
        if self.fisher_kernel is None or len(self.current_step_sequences) < 2:
            return {}
            
        same_prompt_vals = []
        diff_prompt_vals = []
        
        n = len(self.current_step_sequences)
        for i in range(n):
            for j in range(i+1, n):  # Upper triangle only
                seq_i = self.current_step_sequences[i]
                seq_j = self.current_step_sequences[j]
                kernel_val = float(self.fisher_kernel[i, j].item())
                
                if seq_i.prompt_idx == seq_j.prompt_idx:
                    same_prompt_vals.append(kernel_val)
                else:
                    diff_prompt_vals.append(kernel_val)
                    
        analysis = {}
        if same_prompt_vals:
            analysis.update({
                "same_prompt_mean": np.mean(same_prompt_vals),
                "same_prompt_std": np.std(same_prompt_vals),
                "same_prompt_count": len(same_prompt_vals)
            })
            
        if diff_prompt_vals:
            analysis.update({
                "diff_prompt_mean": np.mean(diff_prompt_vals),
                "diff_prompt_std": np.std(diff_prompt_vals), 
                "diff_prompt_count": len(diff_prompt_vals)
            })
            
        return analysis
        
    def reset(self) -> None:
        """Reset probe state for next step."""
        self.current_step_gradients.clear()
        self.current_step_sequences.clear()
        self.current_step_adam_conditioning = None
        self.fisher_kernel = None
        self.delta_h_estimate = None
        self.mean_log_prob = None
        
    def save_data(self, filepath: str) -> None:
        """Save collected data for detailed analysis."""
        if not self.enabled:
            return
            
        data = {
            "step_counter": self.step_counter,
            "sequences": [
                {
                    "prompt_idx": seq.prompt_idx,
                    "gen_idx": seq.gen_idx,
                    "global_idx": seq.global_idx,
                    "log_prob": seq.log_prob,
                    "advantage": seq.advantage,
                    "length": seq.length,
                    "step_idx": seq.step_idx,
                    "device_rank": seq.device_rank
                }
                for seq in self.current_step_sequences
            ],
            "fisher_kernel": self.fisher_kernel.tolist() if self.fisher_kernel is not None else None,
            "delta_h_estimate": self.delta_h_estimate,
            "mean_log_prob": self.mean_log_prob,
            "prompt_structure_analysis": self.analyze_prompt_structure()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.debug:
            print(f"[EntropyProbe] Saved data to {filepath}")