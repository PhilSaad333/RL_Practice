# simple_entropy_probe.py - Lightweight entropy change estimation for regular training
import torch
import torch.distributed as dist
from typing import Optional, Dict, List, Any
import numpy as np
from dataclasses import dataclass


class SimpleEntropyProbe:
    """
    Lightweight entropy change estimation during regular RL training.
    
    Computes δH ≈ Σ_α (∂_α H) × (δθ_α) × P_α without storing Fisher kernel.
    This provides fast entropy monitoring with minimal computational overhead.
    
    Key features:
    - O(n_params) memory usage (not O(n_sequences²))
    - ~5% computational overhead
    - Real-time entropy change prediction
    - Configurable Adam preconditioning
    """
    
    def __init__(
        self,
        enabled: bool = True,
        debug: bool = False,
        preconditioning_mode: str = "previous_step",  # "none", "previous_step"
        log_every: int = 1,
    ):
        """
        Args:
            enabled: Whether the probe is active
            debug: Whether to print debug information
            preconditioning_mode: How to handle Adam conditioning factors
            log_every: Log metrics every N steps
        """
        self.enabled = enabled
        self.debug = debug
        self.preconditioning_mode = preconditioning_mode
        self.log_every = log_every
        
        # Metrics storage
        self.step_counter = 0
        self.last_delta_h = None
        self.last_entropy_gradient_norm = None
        self.last_param_update_norm = None
        
        # Running statistics
        self.delta_h_history = []
        self.entropy_history = []
        
    def compute_entropy_gradients_microbatch(
        self,
        token_log_probs: torch.Tensor,  # (B, G, T_g) token-level log probabilities with gradients
        gen_mask: torch.Tensor,         # (B, G, T_g) generation mask
        advantages: torch.Tensor,       # (B, G) advantages
        trainable_params: List[torch.nn.Parameter],
        pad_id: int = 0
    ) -> torch.Tensor:
        """
        Compute entropy gradients for a single microbatch using torch.autograd.grad().
        
        This works directly with token-level log probabilities to match the exact 
        computation path used by the optimizer, ensuring gradient flow to LoRA parameters.
        
        Args:
            token_log_probs: Token-level log probabilities with gradients (B, G, T_g)
            gen_mask: Generation mask (B, G, T_g)
            advantages: Advantages for this microbatch (B, G)  
            trainable_params: Trainable parameters
            pad_id: Padding token ID
            
        Returns:
            Flattened entropy gradients for this microbatch
        """
        if not self.enabled:
            return torch.zeros(sum(p.numel() for p in trainable_params), device=token_log_probs.device)
            
        B, G, T_g = token_log_probs.shape
        device = token_log_probs.device
        
        # Compute sequence log probabilities from token-level (same as optimizer path)
        seq_log_probs = (token_log_probs * gen_mask).sum(dim=-1)  # (B, G) with gradients
        
        # Compute mean log probability for baseline (distributed)
        local_mean = seq_log_probs.mean()
        if dist.is_initialized():
            # Average across all GPUs for global baseline
            global_mean = local_mean.clone()
            dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
            mean_log_prob = global_mean
        else:
            mean_log_prob = local_mean
        
        # Centered log probabilities: S(t) - S̄_global (detached to remove unwanted gradients)
        centered_log_probs = (seq_log_probs - mean_log_prob).detach()  # (B, G)
        
        # Create entropy loss: sum((S-S̄) * S) where only S has gradients
        # This gives us ∂/∂α [sum((S-S̄) * S)] = sum((S-S̄) * ∂S/∂α) = the entropy gradient we want
        entropy_loss = torch.sum(centered_log_probs * seq_log_probs)
        
        # Use torch.autograd.grad() instead of .backward() to avoid interfering with .grad
        try:
            entropy_grads = torch.autograd.grad(
                outputs=entropy_loss,
                inputs=trainable_params,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
                allow_unused=True  # Some LoRA parameters may not be used in this computation
            )
        except RuntimeError as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] autograd.grad failed: {e}")
            return torch.zeros(sum(p.numel() for p in trainable_params), device=device)
        
        # Get global batch size for proper normalization
        if dist.is_initialized():
            world_size = dist.get_world_size()
            global_batch_size = B * G * world_size
        else:
            global_batch_size = B * G
            
        # Process gradients: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N_global) Σ (S-S̄) ∂_α S
        entropy_grad_chunks = []
        non_zero_grads = 0
        none_grads = 0
        for param, grad in zip(trainable_params, entropy_grads):
            if grad is not None:
                entropy_grad = -grad.detach().flatten() / global_batch_size
                entropy_grad_chunks.append(entropy_grad)
                if torch.any(grad != 0):
                    non_zero_grads += 1
            else:
                none_grads += 1
                entropy_grad_chunks.append(torch.zeros(param.numel(), device=device))
        
        result = torch.cat(entropy_grad_chunks)
        
        if self.debug:
            print(f"[SimpleEntropyProbe] Microbatch entropy computation:")
            print(f"  Entropy loss: {entropy_loss.item():.6f}, device: {entropy_loss.device}")
            print(f"  Token logp shape: {token_log_probs.shape}, device: {token_log_probs.device}")
            print(f"  Seq logp grad_fn: {seq_log_probs.grad_fn}")
            print(f"  Trainable params: {len(trainable_params)}")
            print(f"  First param device: {trainable_params[0].device if trainable_params else 'None'}")
            print(f"  Param requires_grad: {trainable_params[0].requires_grad if trainable_params else 'None'}")
            print(f"  Non-zero gradients: {non_zero_grads}/{len(trainable_params)}")
            print(f"  None gradients: {none_grads}/{len(trainable_params)}")
            print(f"  Result norm: {torch.norm(result).item():.6f}")
        
        return result
    
    def complete_delta_h_calculation(
        self,
        accumulated_entropy_grads: torch.Tensor,  # Accumulated entropy gradients across microbatches
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float
    ) -> Dict[str, float]:
        """
        Complete the delta H calculation using accumulated entropy gradients and current Adam states.
        
        Args:
            accumulated_entropy_grads: Entropy gradients accumulated across all microbatches
            trainable_params: Trainable parameters
            optimizer: Current optimizer with up-to-date Adam states
            learning_rate: Current learning rate
            
        Returns:
            Dictionary with entropy change prediction and diagnostic metrics
        """
        if not self.enabled:
            return {}
            
        try:
            # All-reduce accumulated entropy gradients across ranks (like training gradients)
            if dist.is_initialized():
                dist.all_reduce(accumulated_entropy_grads, op=dist.ReduceOp.SUM)
            
            # Extract parameter updates: δθ_α (what optimizer just applied)
            param_updates = self._extract_parameter_updates(trainable_params, learning_rate)
            
            # Apply Adam preconditioning: P_α (using current Adam states)
            conditioned_updates = self._apply_preconditioning(param_updates, optimizer, trainable_params)
            
            # Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(accumulated_entropy_grads * conditioned_updates).item()
            
            if self.debug:
                print(f"[SimpleEntropyProbe] Final calculation:")
                print(f"  Accumulated entropy grad norm: {torch.norm(accumulated_entropy_grads).item():.6f}")
                print(f"  Param updates norm: {torch.norm(param_updates).item():.6f}")
                print(f"  Conditioned updates norm: {torch.norm(conditioned_updates).item():.6f}")
                print(f"  Delta H: {delta_h:.6f}")
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(accumulated_entropy_grads).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            self.step_counter += 1
            
            # Debug output
            if self.debug and (self.step_counter % self.log_every == 0):
                print(f"[SimpleEntropyProbe] Step {self.step_counter}:")
                print(f"  Predicted δH: {delta_h:.6f}")
                print(f"  Entropy grad norm: {self.last_entropy_gradient_norm:.6f}")
                print(f"  Param update norm: {self.last_param_update_norm:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in delta H calculation: {e}")
            return {}

    def compute_delta_h_prediction(
        self,
        rollouts: Any,  # RolloutBatch
        advantages: torch.Tensor,  # (B, G)
        log_probs: torch.Tensor,   # (B, G) sequence-level log probabilities
        trainable_params: List[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        pad_id: int = 0
    ) -> Dict[str, float]:
        """
        Predict entropy change using simple δH = Σ_α (∂_α H) × (δθ_α) × P_α formula.
        
        Returns:
            Dictionary with entropy change prediction and diagnostic metrics
        """
        if not self.enabled:
            return {}
            
        try:
            # 1. Compute entropy gradients: ∂_α H = -E[(S-S̄) ∂_α S]
            entropy_gradients = self._compute_entropy_gradients(
                rollouts, log_probs, trainable_params, pad_id
            )
            
            # 2. Get parameter updates: δθ_α (what optimizer will apply)
            param_updates = self._extract_parameter_updates(trainable_params, learning_rate)
            
            # 3. Apply Adam preconditioning: P_α
            conditioned_updates = self._apply_preconditioning(param_updates, optimizer, trainable_params)
            
            # 4. Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
            delta_h = torch.sum(entropy_gradients * conditioned_updates).item()
            
            # Store metrics
            self.last_delta_h = delta_h
            self.last_entropy_gradient_norm = torch.norm(entropy_gradients).item()
            self.last_param_update_norm = torch.norm(conditioned_updates).item()
            
            # Update history
            self.delta_h_history.append(delta_h)
            if len(self.delta_h_history) > 100:  # Keep last 100 steps
                self.delta_h_history.pop(0)
                
            # Compute current entropy for reference
            current_entropy = self._compute_current_entropy(log_probs, pad_id)
            self.entropy_history.append(current_entropy)
            if len(self.entropy_history) > 100:
                self.entropy_history.pop(0)
            
            self.step_counter += 1
            
            # Debug output
            if self.debug and (self.step_counter % self.log_every == 0):
                print(f"[SimpleEntropyProbe] Step {self.step_counter}:")
                print(f"  Predicted δH: {delta_h:.6f}")
                print(f"  Current entropy: {current_entropy:.6f}")
                print(f"  Entropy grad norm: {self.last_entropy_gradient_norm:.6f}")
                print(f"  Param update norm: {self.last_param_update_norm:.6f}")
            
            return {
                "simple_entropy_delta_h_predicted": delta_h,
                "simple_entropy_current": current_entropy,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_step_count": self.step_counter,
            }
            
        except Exception as e:
            if self.debug:
                print(f"[SimpleEntropyProbe] Error in step {self.step_counter}: {e}")
            return {}
    
    def _compute_entropy_gradients(
        self,
        rollouts: Any,
        log_probs: torch.Tensor,  # (B, G)
        trainable_params: List[torch.nn.Parameter],
        pad_id: int
    ) -> torch.Tensor:
        """
        Compute aggregated entropy gradients: ∂_α H = -E[(S-S̄) ∂_α S].
        
        This is much cheaper than computing per-sequence gradients.
        """
        B, G = log_probs.shape
        device = log_probs.device
        
        # Compute mean log probability for baseline (distributed)
        local_mean = log_probs.mean()
        if dist.is_initialized():
            # Average across all GPUs for global baseline
            global_mean = local_mean.clone()
            dist.all_reduce(global_mean, op=dist.ReduceOp.AVG)
            mean_log_prob = global_mean
        else:
            mean_log_prob = local_mean
        
        # Centered log probabilities: S(t) - S̄_global
        centered_log_probs = log_probs - mean_log_prob  # (B, G)
        
        # We need to compute: E[(S-S̄) ∂_α S] = E[(S-S̄) ∂_α log π(t)]
        # This requires computing gradients of log probabilities w.r.t. parameters
        
        # Clear existing gradients
        for param in trainable_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute weighted sum of log probabilities for backprop
        # This gives us ∂_α Σ_t (S(t) - S̄) S(t) = Σ_t (S(t) - S̄) ∂_α S(t)
        weighted_log_prob_sum = torch.sum(centered_log_probs * log_probs)
        
        # Backward pass to get gradients
        weighted_log_prob_sum.backward(retain_graph=True)
        
        # Collect gradients and normalize by global batch size
        entropy_grad_chunks = []
        
        # Get global batch size for proper normalization
        if dist.is_initialized():
            world_size = dist.get_world_size()
            global_batch_size = B * G * world_size
        else:
            global_batch_size = B * G
            
        for param in trainable_params:
            if param.grad is not None:
                # Entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N_global) Σ (S-S̄) ∂_α S
                entropy_grad = -param.grad.detach().flatten() / global_batch_size
                entropy_grad_chunks.append(entropy_grad)
            else:
                entropy_grad_chunks.append(torch.zeros(param.numel(), device=device))
        
        local_entropy_grad = torch.cat(entropy_grad_chunks)
        
        # Sum gradients across GPUs (they're already normalized by global batch size)
        if dist.is_initialized():
            global_entropy_grad = local_entropy_grad.clone()
            dist.all_reduce(global_entropy_grad, op=dist.ReduceOp.SUM)
            return global_entropy_grad
        else:
            return local_entropy_grad
    
    def _extract_parameter_updates(
        self,
        trainable_params: List[torch.nn.Parameter],
        learning_rate: float
    ) -> torch.Tensor:
        """Extract parameter updates δθ_α that optimizer will apply."""
        param_update_chunks = []
        
        for param in trainable_params:
            if param.grad is not None:
                # Basic SGD update: δθ = -η * ∇L
                param_update = -learning_rate * param.grad.detach().flatten()
                param_update_chunks.append(param_update)
            else:
                param_update_chunks.append(torch.zeros(param.numel(), device=param.device))
        
        return torch.cat(param_update_chunks)
    
    def _apply_preconditioning(
        self,
        param_updates: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        trainable_params: List[torch.nn.Parameter]
    ) -> torch.Tensor:
        """Apply Adam preconditioning factors P_α to parameter updates."""
        
        if self.preconditioning_mode == "none":
            return param_updates
        
        if self.preconditioning_mode == "previous_step":
            # Use Adam conditioning from previous step
            conditioning_chunks = []
            param_idx = 0
            
            for param in trainable_params:
                param_size = param.numel()
                
                if param in optimizer.state and 'exp_avg_sq' in optimizer.state[param]:
                    # Adam conditioning: 1 / (sqrt(v) + eps)
                    state = optimizer.state[param]
                    v = state['exp_avg_sq']
                    eps = optimizer.param_groups[0].get('eps', 1e-8)
                    conditioning = 1.0 / (torch.sqrt(v) + eps)
                    conditioning_chunks.append(conditioning.flatten())
                else:
                    # No conditioning available (early in training)
                    conditioning_chunks.append(torch.ones(param_size, device=param.device))
                
                param_idx += param_size
            
            conditioning_factors = torch.cat(conditioning_chunks)
            return param_updates * conditioning_factors
        
        return param_updates
    
    def _compute_current_entropy(self, log_probs: torch.Tensor, pad_id: int) -> float:
        """Compute current entropy H = -E[log π(t)] for reference."""
        return -log_probs.mean().item()
    
    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics for logging."""
        metrics = {}
        
        if self.last_delta_h is not None:
            metrics.update({
                "simple_entropy_delta_h": self.last_delta_h,
                "simple_entropy_grad_norm": self.last_entropy_gradient_norm,
                "simple_entropy_param_norm": self.last_param_update_norm,
                "simple_entropy_steps": self.step_counter,
            })
        
        # Add rolling statistics
        if len(self.delta_h_history) > 1:
            metrics.update({
                "simple_entropy_delta_h_mean": np.mean(self.delta_h_history[-10:]),  # Last 10 steps
                "simple_entropy_delta_h_std": np.std(self.delta_h_history[-10:]),
            })
        
        if len(self.entropy_history) > 1:
            metrics.update({
                "simple_entropy_current_mean": np.mean(self.entropy_history[-10:]),
                "simple_entropy_trend": self.entropy_history[-1] - self.entropy_history[0] if len(self.entropy_history) >= 2 else 0.0,
            })
        
        return metrics
    
    def reset(self) -> None:
        """Reset probe state."""
        self.last_delta_h = None
        self.last_entropy_gradient_norm = None
        self.last_param_update_norm = None
        
    def save_history(self, filepath: str) -> None:
        """Save entropy history for analysis."""
        if not self.enabled:
            return
            
        data = {
            "delta_h_history": self.delta_h_history,
            "entropy_history": self.entropy_history,
            "step_counter": self.step_counter,
            "config": {
                "preconditioning_mode": self.preconditioning_mode,
                "log_every": self.log_every,
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.debug:
            print(f"[SimpleEntropyProbe] Saved history to {filepath}")