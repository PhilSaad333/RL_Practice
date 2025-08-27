"""
Debug Probe Components

Enhanced version of probe_components.py with comprehensive debug logging
to track gradient magnitudes, buffer norms, and scaling operations.

This version adds detailed logging to identify the source of the 40,000x
unexpected scaling in δH₁ values.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import nullcontext
import logging
from collections import defaultdict
import time
import math

class DebugProbeComponents:
    """
    Debug version of ProbeComponents with extensive logging.
    
    Tracks gradient magnitudes, buffer norms, and scaling operations
    at all critical points to identify normalization issues.
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], logger: logging.Logger):
        self.model = model
        self.config = config  
        self.logger = logger
        self.device = next(model.parameters()).device
        
        # AMP settings to match training
        self.use_amp = config['memory_config']['amp']
        self.amp_dtype = getattr(torch, config['memory_config']['dtype'])
        
        # Sampling parameters
        self.B = config['batch_config']['B']  # prompts per batch
        self.G = config['batch_config']['G']  # responses per prompt
        self.microbatch_size = config['memory_config']['microbatch_size']
        
        # Mode configuration
        probe_config = config.get('probe_config', {})
        self.mode = probe_config.get('mode', 'exact')
        self.M = probe_config.get('M', None)
        
        # Initialize tokenizer if needed
        self._tokenizer = None
        
        # Debug settings
        self.debug_detailed = True  # Enable detailed per-microbatch logging
        self.debug_norm_threshold = 1e-10  # Log if norms are below this threshold
        
    # =================================================================
    # DEBUG HELPER FUNCTIONS
    # =================================================================
    
    def compute_total_grad_norm(self) -> float:
        """Compute L2 norm of all parameter gradients."""
        total_norm_sq = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param_norm_sq = param.grad.detach().norm(dtype=torch.float64).item() ** 2
                total_norm_sq += param_norm_sq
                param_count += 1
                
        total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0
        self.logger.debug(f"    compute_total_grad_norm: {param_count} params, norm={total_norm:.6e}")
        return total_norm
    
    def compute_buffer_norm(self, buffer: Dict[int, torch.Tensor]) -> float:
        """Compute L2 norm of parameter buffer."""
        total_norm_sq = 0.0
        param_count = len(buffer)
        
        for param_id, tensor in buffer.items():
            tensor_norm_sq = tensor.norm(dtype=torch.float64).item() ** 2
            total_norm_sq += tensor_norm_sq
            
        total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0
        self.logger.debug(f"    compute_buffer_norm: {param_count} tensors, norm={total_norm:.6e}")
        return total_norm
        
    def log_buffer_stats(self, buffer: Dict[int, torch.Tensor], name: str):
        """Log detailed statistics about a parameter buffer."""
        if not buffer:
            self.logger.debug(f"    {name}: EMPTY BUFFER")
            return
            
        norm = self.compute_buffer_norm(buffer)
        tensor_count = len(buffer)
        
        # Compute min/max values across all tensors
        all_values = []
        for tensor in buffer.values():
            all_values.extend(tensor.flatten().tolist())
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            mean_val = sum(all_values) / len(all_values)
        else:
            min_val = max_val = mean_val = 0.0
            
        self.logger.debug(f"    {name}: norm={norm:.6e}, tensors={tensor_count}, "
                         f"min={min_val:.6e}, max={max_val:.6e}, mean={mean_val:.6e}")
        
        # Flag potential issues
        if norm < self.debug_norm_threshold:
            self.logger.warning(f"    ⚠️  {name} has very small norm: {norm:.6e}")
    
    def log_scaling_operation(self, operation: str, scale_factor: float, 
                             norm_before: float, norm_after: float):
        """Log scaling operation with before/after norms."""
        expected_ratio = abs(scale_factor) if scale_factor != 0 else 0.0
        actual_ratio = norm_after / norm_before if norm_before > 0 else float('inf')
        
        self.logger.info(f"🔍 SCALING: {operation}")
        self.logger.info(f"    scale_factor={scale_factor:.6e}")
        self.logger.info(f"    norm_before={norm_before:.6e}")  
        self.logger.info(f"    norm_after={norm_after:.6e}")
        self.logger.info(f"    expected_ratio={expected_ratio:.6e}")
        self.logger.info(f"    actual_ratio={actual_ratio:.6e}")
        
        if expected_ratio > 0 and abs(actual_ratio - expected_ratio) > 0.01 * expected_ratio:
            self.logger.warning(f"    ⚠️  Ratio mismatch! Expected {expected_ratio:.6e}, got {actual_ratio:.6e}")

    # =================================================================
    # EXISTING METHODS WITH DEBUG LOGGING ADDED
    # =================================================================
    
    def scale_param_gradients(self, scale_factor: float):
        """Scale all parameter gradients by a factor."""
        norm_before = self.compute_total_grad_norm()
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data *= scale_factor
                
        norm_after = self.compute_total_grad_norm()
        self.log_scaling_operation("scale_param_gradients", scale_factor, norm_before, norm_after)

    def add_into_param_buffer(self, target_buffer: Dict[int, torch.Tensor]):
        """Add current gradients into target buffer with debug logging."""
        buffer_norm_before = self.compute_buffer_norm(target_buffer)
        grad_norm = self.compute_total_grad_norm()
        
        self.logger.debug(f"🔍 ACCUMULATION: add_into_param_buffer")
        self.logger.debug(f"    buffer_norm_before={buffer_norm_before:.6e}")
        self.logger.debug(f"    grad_norm_to_add={grad_norm:.6e}")
        
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param_id = id(param)
                if param_id not in target_buffer:
                    target_buffer[param_id] = torch.zeros_like(param.grad, dtype=torch.float32, device='cpu')
                
                # Convert gradient to CPU fp32 and accumulate
                grad_cpu = param.grad.detach().to('cpu', dtype=torch.float32)
                target_buffer[param_id] += grad_cpu
        
        buffer_norm_after = self.compute_buffer_norm(target_buffer)
        self.logger.debug(f"    buffer_norm_after={buffer_norm_after:.6e}")
        
        if buffer_norm_before > 0:
            norm_increase = buffer_norm_after - buffer_norm_before
            self.logger.debug(f"    norm_increase={norm_increase:.6e}")
    
    def zeros_like_params(self, dtype=torch.float32, device='cpu') -> Dict[int, torch.Tensor]:
        """Create zero buffer matching model parameters."""
        buffer = {}
        param_count = 0
        total_elements = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                param_id = id(param)
                buffer[param_id] = torch.zeros_like(param, dtype=dtype, device=device)
                param_count += 1
                total_elements += param.numel()
                
        self.logger.debug(f"    zeros_like_params: {param_count} params, {total_elements} elements")
        return buffer

    def build_LX_from_S(self, S_dict: Dict[str, Any], weighting_mode: str = "dr_grpo") -> torch.Tensor:
        """Build X-loss with debug logging."""
        self.logger.debug(f"🔍 LOSS: build_LX_from_S(weighting_mode={weighting_mode})")
        
        # Extract required components
        S = S_dict['S']  # [batch_size, G] log-prob differences
        max_lengths = S_dict['max_lengths']  # [batch_size] max sequence lengths
        batch_size = S.shape[0]
        G = S.shape[1]
        
        self.logger.debug(f"    S.shape={S.shape}, batch_size={batch_size}, G={G}")
        self.logger.debug(f"    S stats: min={S.min().item():.6f}, max={S.max().item():.6f}, mean={S.mean().item():.6f}")
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_count = 0
        
        for b in range(batch_size):
            S_b = S[b]  # [G]
            L_max_b = max_lengths[b]  # scalar
            
            if 'advantages' in S_dict:
                A_b = S_dict['advantages'][b]  # [G]
                # L_X: use detached advantages (LOO baseline)
                with torch.no_grad():
                    if weighting_mode == "dr_grpo":
                        weights_b = (A_b - A_b.mean()) / L_max_b  # Centered advantages
                    elif weighting_mode == "per_token_avg":
                        if 'gen_lengths' in S_dict:
                            gen_lengths_b = S_dict['gen_lengths'][b]  # [G]
                            weights_b = (A_b - A_b.mean()) / gen_lengths_b.clamp(min=1.0)
                        else:
                            weights_b = (A_b - A_b.mean()) / L_max_b
                    else:
                        raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
            else:
                # Uniform weighting baseline
                with torch.no_grad():
                    if weighting_mode == "dr_grpo":
                        S_w_b = S_b / L_max_b
                    elif weighting_mode == "per_token_avg":
                        if 'gen_lengths' in S_dict:
                            gen_lengths_b = S_dict['gen_lengths'][b]  # [G]
                            S_w_b = S_b / gen_lengths_b.clamp(min=1.0)
                        else:
                            S_w_b = S_b / L_max_b
                    else:
                        raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
                    weights_b = S_w_b - S_w_b.mean()  # Centered
                    
            prompt_loss = (weights_b * S_b).mean()
            total_loss = total_loss + prompt_loss
            total_count += 1
        
        # Return average loss across prompts
        final_loss = total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
        self.logger.debug(f"    final_loss={final_loss.item():.6e}, total_count={total_count}")
        
        # Flag potential issues
        if abs(final_loss.item()) < 1e-10:
            self.logger.warning(f"    ⚠️  Very small loss magnitude: {final_loss.item():.6e}")
            
        return final_loss

    def build_LY_from_S(self, S_dict: Dict[str, Any]) -> torch.Tensor:
        """Build Y-loss with debug logging."""
        self.logger.debug(f"🔍 LOSS: build_LY_from_S")
        
        # Extract required components
        S = S_dict['S']  # [batch_size, G]
        max_lengths = S_dict['max_lengths']  # [batch_size]
        batch_size = S.shape[0]
        G = S.shape[1]
        
        self.logger.debug(f"    S.shape={S.shape}, batch_size={batch_size}, G={G}")
        self.logger.debug(f"    S stats: min={S.min().item():.6f}, max={S.max().item():.6f}, mean={S.mean().item():.6f}")
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_count = 0
        
        for b in range(batch_size):
            S_b = S[b]  # [G]
            A_b = S_dict['advantages'][b]  # [G]
            L_max_b = max_lengths[b]  # scalar
            
            # L_Y for this prompt: mean_g((A / L_max) * S)
            weights_b = A_b / L_max_b  # θ-independent weights
            prompt_loss = (weights_b * S_b).mean()
            total_loss = total_loss + prompt_loss
            total_count += 1
        
        # Return average loss across prompts
        final_loss = total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
        self.logger.debug(f"    final_loss={final_loss.item():.6e}, total_count={total_count}")
        
        # Flag potential issues
        if abs(final_loss.item()) < 1e-10:
            self.logger.warning(f"    ⚠️  Very small loss magnitude: {final_loss.item():.6e}")
            
        return final_loss

    def accumulate_sum_X(self, E_batch: Dict[str, Any], mb_size_prompts: int, 
                        weighting_mode: str = "dr_grpo") -> Tuple[Dict[int, torch.Tensor], int]:
        """Accumulate ΣX from E batch with comprehensive debug logging."""
        self.logger.info("🔍🔍 STARTING accumulate_sum_X 🔍🔍")
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.logger.info(f"Model has {len(params)} trainable parameters")
        
        # Initialize sum buffer on CPU in fp32
        sum_X_buf = self.zeros_like_params(dtype=torch.float32, device='cpu')
        self.log_buffer_stats(sum_X_buf, "initial_sum_X_buf")
        
        # Count prompts processed
        B_local = 0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.info(f"Starting X accumulation: mb_size_prompts={mb_size_prompts}, weighting_mode={weighting_mode}")
        
        microbatch_count = 0
        
        # Process microbatches
        with no_sync_context():  # Prevent DDP gradient averaging
            for microbatch in self.iter_microbatches(E_batch, mb_size_prompts):
                microbatch_count += 1
                self.logger.info(f"🔍 MICROBATCH {microbatch_count}")
                
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                initial_grad_norm = self.compute_total_grad_norm()
                self.logger.debug(f"    After zero_grad: grad_norm={initial_grad_norm:.6e}")
                
                # Forward pass with teacher forcing
                S_dict = self._teacher_force_logprobs(microbatch)
                
                # Build X-loss with detached LOO coefficient
                L_X_mb = self.build_LX_from_S(S_dict, weighting_mode)
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    mb_prompt_count = len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    mb_prompt_count = microbatch['advantages'].shape[0]
                else:
                    mb_prompt_count = len(microbatch.get('max_lengths', []))
                
                self.logger.info(f"    mb_prompt_count={mb_prompt_count}, L_X_mb={L_X_mb.item():.6e}")
                
                # Backward pass - populates param.grad with raw X gradients  
                self.logger.debug("    Calling backward()...")
                L_X_mb.backward()
                
                grad_norm_after_backward = self.compute_total_grad_norm()
                self.logger.info(f"    After backward: grad_norm={grad_norm_after_backward:.6e}")
                
                # Scale gradients by microbatch size to convert from average to sum
                # build_LX_from_S returns average over prompts, but we need sum
                self.logger.info(f"    Scaling gradients by mb_prompt_count={mb_prompt_count}")
                self.scale_param_gradients(mb_prompt_count)
                
                # Accumulate gradients into sum buffer
                self.logger.debug("    Adding gradients to sum_X_buf...")
                self.add_into_param_buffer(sum_X_buf)
                
                B_local += mb_prompt_count
                self.logger.info(f"    B_local now = {B_local}")
                
                # Log current buffer state
                if self.debug_detailed:
                    self.log_buffer_stats(sum_X_buf, f"sum_X_buf_after_mb_{microbatch_count}")
                
                # Clear gradients for next microbatch
                self.model.zero_grad(set_to_none=True)
        
        # Final buffer statistics
        self.logger.info("🔍🔍 COMPLETED accumulate_sum_X 🔍🔍")
        self.log_buffer_stats(sum_X_buf, "FINAL_sum_X_buf")
        self.logger.info(f"Total microbatches processed: {microbatch_count}")
        self.logger.info(f"Total prompts processed: {B_local}")
        
        return sum_X_buf, B_local

    def accumulate_sum_Y(self, U_batch: Dict[str, Any], mb_size_prompts: int,
                        adam_preconditioner) -> Tuple[Dict[int, torch.Tensor], int]:
        """Accumulate ΣY from U batch with comprehensive debug logging."""
        self.logger.info("🔍🔍 STARTING accumulate_sum_Y 🔍🔍")
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize sum buffer on CPU in fp32
        sum_Y_buf = self.zeros_like_params(dtype=torch.float32, device='cpu')
        self.log_buffer_stats(sum_Y_buf, "initial_sum_Y_buf")
        
        # Count prompts processed
        B_local = 0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.info(f"Starting Y accumulation: mb_size_prompts={mb_size_prompts}")
        
        microbatch_count = 0
        
        # Process microbatches
        with no_sync_context():  # Prevent DDP gradient averaging
            for microbatch in self.iter_microbatches(U_batch, mb_size_prompts):
                microbatch_count += 1
                self.logger.info(f"🔍 MICROBATCH {microbatch_count}")
                
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                initial_grad_norm = self.compute_total_grad_norm()
                self.logger.debug(f"    After zero_grad: grad_norm={initial_grad_norm:.6e}")
                
                # Forward pass with teacher forcing
                S_dict = self._teacher_force_logprobs(microbatch)
                
                # Build Y-loss with live advantages
                L_Y_mb = self.build_LY_from_S(S_dict)
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    mb_prompt_count = len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    mb_prompt_count = microbatch['advantages'].shape[0]
                else:
                    mb_prompt_count = len(microbatch.get('max_lengths', []))
                
                self.logger.info(f"    mb_prompt_count={mb_prompt_count}, L_Y_mb={L_Y_mb.item():.6e}")
                
                # Backward pass - populates param.grad with raw ∇J
                self.logger.debug("    Calling backward()...")
                L_Y_mb.backward()
                
                grad_norm_after_backward = self.compute_total_grad_norm()
                self.logger.info(f"    After backward: grad_norm={grad_norm_after_backward:.6e}")
                
                # Scale gradients by microbatch size to convert from average to sum
                # build_LY_from_S returns average over prompts, but we need sum
                self.logger.info(f"    Scaling gradients by mb_prompt_count={mb_prompt_count}")
                self.scale_param_gradients(mb_prompt_count)
                
                # Apply preconditioner in-place: grad ← P(grad) 
                self.logger.debug("    Applying Adam preconditioner...")
                grad_norm_before_preconditioner = self.compute_total_grad_norm()
                
                for param in params:
                    if param.grad is not None:
                        preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                        param.grad.copy_(preconditioned_grad)
                
                grad_norm_after_preconditioner = self.compute_total_grad_norm()
                self.logger.info(f"    After preconditioner: norm={grad_norm_before_preconditioner:.6e} → {grad_norm_after_preconditioner:.6e}")
                
                # Accumulate preconditioned gradients into sum buffer  
                self.logger.debug("    Adding preconditioned gradients to sum_Y_buf...")
                self.add_into_param_buffer(sum_Y_buf)
                
                B_local += mb_prompt_count
                self.logger.info(f"    B_local now = {B_local}")
                
                # Log current buffer state
                if self.debug_detailed:
                    self.log_buffer_stats(sum_Y_buf, f"sum_Y_buf_after_mb_{microbatch_count}")
                
                # Clear gradients for next microbatch
                self.model.zero_grad(set_to_none=True)
        
        # Final buffer statistics
        self.logger.info("🔍🔍 COMPLETED accumulate_sum_Y 🔍🔍")
        self.log_buffer_stats(sum_Y_buf, "FINAL_sum_Y_buf")
        self.logger.info(f"Total microbatches processed: {microbatch_count}")
        self.logger.info(f"Total prompts processed: {B_local}")
        
        return sum_Y_buf, B_local

    def dot_param_buffers(self, buf1: Dict[int, torch.Tensor], buf2: Dict[int, torch.Tensor]) -> float:
        """Compute dot product of two parameter buffers with debug logging."""
        self.logger.info("🔍 COMPUTING: dot_param_buffers")
        
        self.log_buffer_stats(buf1, "buf1")
        self.log_buffer_stats(buf2, "buf2")
        
        total_dot = 0.0
        param_count = 0
        
        for param_id in buf1:
            if param_id in buf2:
                dot_contrib = (buf1[param_id] * buf2[param_id]).sum().item()
                total_dot += dot_contrib
                param_count += 1
                
                if self.debug_detailed and abs(dot_contrib) > 1e-8:
                    self.logger.debug(f"    param {param_count}: dot_contrib={dot_contrib:.6e}")
        
        self.logger.info(f"    total_dot={total_dot:.10e} (from {param_count} parameters)")
        
        # Flag potential issues
        if abs(total_dot) < 1e-12:
            self.logger.warning(f"    ⚠️  Very small dot product: {total_dot:.6e}")
            
        return total_dot

    # =================================================================
    # PLACEHOLDER METHODS (simplified versions for testing)
    # =================================================================
    
    def iter_microbatches(self, batch_dict: Dict[str, Any], size: int):
        """Simplified microbatch iterator for testing."""
        # This is a simplified version - in real implementation would need
        # to handle the full batch structure properly
        batch_size = len(batch_dict.get('advantages', []))
        
        for start_idx in range(0, batch_size, size):
            end_idx = min(start_idx + size, batch_size)
            microbatch = {}
            
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    microbatch[key] = value[start_idx:end_idx]
                elif isinstance(value, (list, tuple)):
                    microbatch[key] = value[start_idx:end_idx]
                else:
                    microbatch[key] = value
                    
            yield microbatch
    
    def _teacher_force_logprobs(self, batch_dict):
        """Placeholder for teacher forcing - returns dummy S_dict for testing."""
        # This would need to be implemented properly for real use
        batch_size = len(batch_dict.get('advantages', [1]))
        G = self.G
        
        return {
            'S': torch.randn(batch_size, G, device=self.device) * 0.1,  # Small random values
            'max_lengths': torch.full((batch_size,), 50.0, device=self.device),
            'advantages': batch_dict.get('advantages', torch.randn(batch_size, G, device=self.device))
        }
        
    def _get_learning_rate(self, optimizer):
        """Extract learning rate from optimizer."""
        if hasattr(optimizer, 'param_groups'):
            lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"🔍 LEARNING RATE: {lr:.2e}")
            return lr
        else:
            self.logger.warning("Cannot extract learning rate from optimizer")
            return 1e-3  # Default fallback