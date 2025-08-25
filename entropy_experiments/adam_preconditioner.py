"""
Adam Preconditioner

Extracts and applies Adam optimizer's second moment preconditioning P^{1/2}.
Based on Section V of offline_entropy_probe_strategy.txt.

The Adam preconditioner transforms gradients according to:
    P^{1/2} g = (v^{1/2} + ε)^{-1} ⊙ g

where v is the second moment estimate from Adam's optimizer state.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Any
import logging


class AdamPreconditioner:
    """
    Extracts second moment estimates from Adam optimizer and applies preconditioning.
    
    This implements the P^{1/2} transformation to match the optimizer's geometry,
    ensuring that our entropy probe predictions align with actual training dynamics.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: Dict[str, Any], logger: logging.Logger):
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        
        # Validate optimizer type
        if not isinstance(optimizer, (optim.Adam, optim.AdamW)):
            self.logger.warning(f"Optimizer {type(optimizer)} is not Adam/AdamW - preconditioner may not work correctly")
            
        # Extract Adam hyperparameters
        self.eps = optimizer.param_groups[0].get('eps', 1e-8)
        self.beta2 = optimizer.param_groups[0].get('betas', (0.9, 0.999))[1]
        
        # Cache second moment states
        self._v_states = {}
        self._extract_second_moments()
        
        self.logger.info(f"AdamPreconditioner initialized with eps={self.eps}, beta2={self.beta2}")
        
    def _extract_second_moments(self) -> None:
        """
        Extract second moment estimates v from Adam optimizer state.
        
        CRITICAL FIX (P3): Extract for ALL parameters, not gated on p.grad existence.
        Use parameter object ID for mapping to avoid ordering issues.
        """
        self._v_states = {}
        self._param_to_group = {}  # Map param id to (group_idx, beta2, eps, step)
        
        for group_idx, group in enumerate(self.optimizer.param_groups):
            beta2 = group.get('betas', (0.9, 0.999))[1]
            eps = group.get('eps', 1e-8)
            
            for param in group['params']:
                param_id = id(param)
                state = self.optimizer.state.get(param, {})
                
                # Extract v (exp_avg_sq) and step count
                v = state.get('exp_avg_sq', None)
                step = state.get('step', 0)
                
                if v is None:
                    # Initialize to zeros if no state exists yet
                    self.logger.debug(f"Parameter {param_id} has no exp_avg_sq - initializing to zeros")
                    v = torch.zeros_like(param.data)
                else:
                    v = v.clone()  # Make a copy
                
                # Store v and metadata by parameter ID
                self._v_states[param_id] = v
                self._param_to_group[param_id] = {
                    'group_idx': group_idx,
                    'beta2': beta2, 
                    'eps': eps,
                    'step': step
                }
                        
        self.logger.debug(f"Extracted second moments for {len(self._v_states)} parameters")
        
    def apply_preconditioner(self, gradient: torch.Tensor, param: torch.nn.Parameter) -> torch.Tensor:
        """
        Apply Adam preconditioning: P^{1/2} g = (v_hat^{1/2} + ε)^{-1} ⊙ g
        
        CRITICAL FIX (P3): Use parameter object directly, apply bias correction.
        
        Args:
            gradient: Gradient tensor to precondition
            param: The parameter object this gradient belongs to
            
        Returns:
            Preconditioned gradient
        """
        param_id = id(param)
        
        if param_id not in self._v_states:
            self.logger.warning(f"No v state found for parameter {param_id} - returning unpreconditioned gradient")
            return gradient
        
        v = self._v_states[param_id]
        param_info = self._param_to_group[param_id]
        
        # Apply bias correction (if step > 0)
        step = param_info['step']
        if step > 0:
            beta2 = param_info['beta2']
            bias_correction2 = 1 - beta2 ** step
            v_hat = v / bias_correction2
        else:
            v_hat = v
            
        eps = param_info['eps']
        
        # Apply preconditioning: (v_hat^{1/2} + ε)^{-1} ⊙ g
        preconditioned = gradient / (torch.sqrt(v_hat) + eps)
        
        return preconditioned
        
    def apply_preconditioner_to_params(self, param_gradients: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        """
        Apply preconditioning to a list of parameter gradients.
        
        UPDATED: Use parameter objects instead of indices.
        
        Args:
            param_gradients: List of (param, gradient) tuples where gradient may be None
            
        Returns:
            List of preconditioned gradients
        """
        preconditioned = []
        
        for param, grad in param_gradients:
            if grad is not None:
                preconditioned_grad = self.apply_preconditioner(grad, param)
                preconditioned.append(preconditioned_grad)
            else:
                preconditioned.append(None)
                
        return preconditioned
        
        
    def get_preconditioner_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics about the preconditioner."""
        if not self._v_states:
            return {"num_params": 0}
            
        v_values = [v.detach().cpu() for v in self._v_states.values()]
        all_v = torch.cat([v.flatten() for v in v_values])
        
        sqrt_v_plus_eps = torch.sqrt(all_v) + self.eps
        preconditioner_values = 1.0 / sqrt_v_plus_eps
        
        return {
            "num_params": len(self._v_states),
            "v_mean": all_v.mean().item(),
            "v_std": all_v.std().item(),
            "v_min": all_v.min().item(), 
            "v_max": all_v.max().item(),
            "preconditioner_mean": preconditioner_values.mean().item(),
            "preconditioner_std": preconditioner_values.std().item(),
            "preconditioner_min": preconditioner_values.min().item(),
            "preconditioner_max": preconditioner_values.max().item(),
            "eps": self.eps,
            "beta2": self.beta2
        }
        
    def update_second_moments(self) -> None:
        """Re-extract second moments from optimizer (call if optimizer state changes)."""
        self._extract_second_moments()
        self.logger.debug("Updated cached second moment estimates")
        
    def validate_preconditioner(self) -> bool:
        """
        Validate that preconditioner is working correctly.
        
        UPDATED FIX (P3): Proper validation with parameter objects and bias correction.
        
        Returns:
            True if validation passes
        """
        try:
            # Check that we have v states for optimizer parameters
            if not self._v_states:
                self.logger.error("No second moment states found")
                return False
                
            # Check that all optimizer parameters have v states
            missing_params = []
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    param_id = id(param)
                    if param_id not in self._v_states:
                        missing_params.append(param_id)
                        
            if missing_params:
                self.logger.error(f"Missing v states for {len(missing_params)} parameters: {missing_params[:5]}...")
                return False
                
            # Check for NaN/inf in v states
            for param_id, v in self._v_states.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    self.logger.error(f"Invalid values in v state for param {param_id}")
                    return False
                    
            # Test preconditioning with actual parameters
            test_passed = 0
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    # Create test gradient with same shape as parameter
                    test_grad = torch.randn_like(param.data) * 0.01
                    
                    try:
                        preconditioned = self.apply_preconditioner(test_grad, param)
                        
                        if torch.isnan(preconditioned).any() or torch.isinf(preconditioned).any():
                            self.logger.error(f"Preconditioning produces invalid values for param {id(param)}")
                            return False
                            
                        # Test that preconditioning actually changes the gradient (unless v is all zeros)
                        v = self._v_states[id(param)]
                        if v.sum() > 0 and torch.allclose(preconditioned, test_grad):
                            self.logger.warning(f"Preconditioning had no effect for param {id(param)}")
                            
                        test_passed += 1
                        if test_passed >= 3:  # Test first few parameters
                            break
                    except Exception as e:
                        self.logger.error(f"Preconditioning failed for param {id(param)}: {e}")
                        return False
                        
                if test_passed >= 3:
                    break
                    
            self.logger.info(f"Preconditioner validation passed (tested {test_passed} parameters)")
            return True
            
        except Exception as e:
            self.logger.error(f"Preconditioner validation failed: {e}")
            return False