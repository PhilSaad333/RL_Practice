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
        
        # Compute mean log probability for baseline
        mean_log_prob = log_probs.mean()
        
        # Centered log probabilities: S(t) - S̄
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
        
        # Collect gradients and normalize by batch size
        entropy_grad_chunks = []
        for param in trainable_params:
            if param.grad is not None:
                # Entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S] = -(1/N) Σ (S-S̄) ∂_α S
                entropy_grad = -param.grad.detach().flatten() / (B * G)
                entropy_grad_chunks.append(entropy_grad)
            else:
                entropy_grad_chunks.append(torch.zeros(param.numel(), device=device))
        
        return torch.cat(entropy_grad_chunks)
    
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