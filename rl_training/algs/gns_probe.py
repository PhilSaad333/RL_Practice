# gns_probe.py - Gradient Noise Scale probe for monitoring batch size efficiency
import torch
import torch.distributed as dist
from typing import Optional, Dict, List, Tuple


class GNSProbe:
    """
    Gradient Noise Scale (GNS) probe for estimating critical batch size.
    
    The GNS measures the ratio of gradient variance to the squared gradient norm,
    helping determine if batch size is too small (high noise) or larger than needed.
    
    This implementation:
    - Stores gradient snapshots over a window of training steps
    - Computes variance across these snapshots
    - Properly handles Dr-GRPO weighted losses
    - Stores gradients on CPU to save GPU memory
    - Tracks EMA of GNS metrics for stability
    """
    
    def __init__(
        self,
        window_size: int = 8,
        ema_alpha: float = 0.1,
        enabled: bool = True,
        debug: bool = False
    ):
        """
        Args:
            window_size: Number of steps to store for variance computation
            ema_alpha: EMA smoothing factor for GNS metrics (0 = no smoothing, 1 = no memory)
            enabled: Whether the probe is active
            debug: Whether to print debug information
        """
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.enabled = enabled
        self.debug = debug
        
        # Storage for gradient snapshots and their weights
        # Each entry is (gradient_tensor_cpu, dr_grpo_weight_sum)
        self.gradient_history: List[Tuple[torch.Tensor, float]] = []
        
        # EMA tracking for smooth metrics
        self.ema_gns_mu: Optional[float] = None
        self.ema_tr_sigma: Optional[float] = None
        self.ema_critical_batch: Optional[float] = None
        
        # Counters
        self.steps_accumulated = 0
        self.total_steps_seen = 0
        
    def store_gradient(
        self,
        trainable_params: List[torch.nn.Parameter],
        dr_grpo_weight: float = 1.0
    ) -> None:
        """
        Store a gradient snapshot after a training step.
        
        Args:
            trainable_params: List of model parameters with gradients
            dr_grpo_weight: Total weight for this step from Dr-GRPO weighting
        """
        if not self.enabled:
            return
            
        # Collect gradients and move to CPU
        grad_chunks = []
        total_params = 0
        
        for p in trainable_params:
            if p.grad is None:
                continue
            # Move to CPU and flatten
            grad_chunks.append(p.grad.detach().float().flatten().cpu())
            total_params += p.grad.numel()
        
        if not grad_chunks:
            return
            
        # Concatenate into single vector on CPU
        grad_vector = torch.cat(grad_chunks)
        
        # Store with weight
        self.gradient_history.append((grad_vector, dr_grpo_weight))
        
        # Maintain window size
        if len(self.gradient_history) > self.window_size:
            self.gradient_history.pop(0)
        
        self.steps_accumulated += 1
        self.total_steps_seen += 1
        
        if self.debug and dist.get_rank() == 0:
            print(f"[GNS] Stored gradient {self.steps_accumulated}/{self.window_size}, "
                  f"||g||^2 = {float((grad_vector * grad_vector).sum()):.6e}, "
                  f"weight = {dr_grpo_weight:.4f}")
    
    def compute_metrics(self, effective_batch_size: int) -> Dict[str, float]:
        """
        Compute GNS metrics from stored gradients.
        
        Args:
            effective_batch_size: Total effective batch size (for critical batch computation)
                                 Should be buffer_size for Dr-GRPO
        
        Returns:
            Dictionary with GNS metrics
        """
        if not self.enabled or len(self.gradient_history) < 2:
            return {
                "gns_mu": 0.0,
                "gns_tr_sigma": 0.0,
                "gns_critical_batch": 0.0,
                "gns_ema_mu": self.ema_gns_mu or 0.0,
                "gns_ema_tr_sigma": self.ema_tr_sigma or 0.0,
                "gns_ema_critical_batch": self.ema_critical_batch or 0.0,
                "gns_window_size": len(self.gradient_history),
            }
        
        # Extract gradients and weights
        gradients = [g for g, _ in self.gradient_history]
        weights = [w for _, w in self.gradient_history]
        
        # Normalize weights
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        # Compute weighted mean gradient
        mean_grad = torch.zeros_like(gradients[0])
        for grad, w in zip(gradients, norm_weights):
            mean_grad += grad * w
        
        # Compute weighted variance
        variance = 0.0
        for grad, w in zip(gradients, norm_weights):
            diff = grad - mean_grad
            variance += w * float((diff * diff).sum())
        
        # Compute gradient norm squared
        grad_norm_sq = float((mean_grad * mean_grad).sum())
        
        # Compute GNS metrics
        tr_sigma = variance  # This is the trace of the covariance
        gns_mu = tr_sigma / max(grad_norm_sq, 1e-12)
        critical_batch = gns_mu * effective_batch_size
        
        # Update EMAs
        if self.ema_gns_mu is None:
            # Initialize EMAs on first computation
            self.ema_gns_mu = gns_mu
            self.ema_tr_sigma = tr_sigma
            self.ema_critical_batch = critical_batch
        else:
            # Update EMAs
            self.ema_gns_mu = (1 - self.ema_alpha) * self.ema_gns_mu + self.ema_alpha * gns_mu
            self.ema_tr_sigma = (1 - self.ema_alpha) * self.ema_tr_sigma + self.ema_alpha * tr_sigma
            self.ema_critical_batch = (1 - self.ema_alpha) * self.ema_critical_batch + self.ema_alpha * critical_batch
        
        if self.debug and dist.get_rank() == 0:
            print(f"[GNS] Computed from {len(self.gradient_history)} steps: "
                  f"mu={gns_mu:.6e}, tr_sigma={tr_sigma:.6e}, "
                  f"critical_batch={critical_batch:.1f}")
        
        return {
            "gns_mu": gns_mu,
            "gns_tr_sigma": tr_sigma,
            "gns_critical_batch": critical_batch,
            "gns_ema_mu": self.ema_gns_mu,
            "gns_ema_tr_sigma": self.ema_tr_sigma,
            "gns_ema_critical_batch": self.ema_critical_batch,
            "gns_window_size": len(self.gradient_history),
        }
    
    def reset(self) -> None:
        """Reset the probe state."""
        self.gradient_history.clear()
        self.steps_accumulated = 0
        # Keep EMAs and total_steps_seen for continuity
    
    def should_compute(self) -> bool:
        """Check if we have enough data to compute metrics."""
        return self.enabled and self.steps_accumulated >= self.window_size