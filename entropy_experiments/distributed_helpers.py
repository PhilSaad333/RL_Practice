"""
Distributed Helpers

Multi-GPU coordination for entropy probe computations.
Based on Section VIII of offline_entropy_probe_strategy.txt.

Provides O(1) scalar communication without passing parameter-sized vectors between GPUs.
All communication is reduced to scalar all-reduce operations.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any, Union
import logging


class DistributedHelpers:
    """
    Handles distributed coordination for entropy probe computations.
    
    Key principle: Only communicate scalars between GPUs, never parameter vectors.
    This keeps communication cost O(1) regardless of model size.
    """
    
    def __init__(self, world_size: int, rank: int, config: Dict[str, Any], logger: logging.Logger):
        self.world_size = world_size
        self.rank = rank
        self.config = config
        self.logger = logger
        
        # Distributed settings
        self.reduce_dtype = getattr(torch, config['distributed']['reduce_dtype'])
        
        # Validate distributed setup
        if not dist.is_initialized():
            self.logger.warning("torch.distributed not initialized - distributed operations will be no-ops")
            self.is_distributed = False
        else:
            self.is_distributed = True
            actual_world_size = dist.get_world_size()
            actual_rank = dist.get_rank()
            
            if actual_world_size != world_size or actual_rank != rank:
                self.logger.warning(f"Mismatch: provided rank/world_size ({rank}/{world_size}) vs actual ({actual_rank}/{actual_world_size})")
                
        self.logger.info(f"DistributedHelpers initialized: rank={rank}/{world_size}, distributed={self.is_distributed}")
        
    def reduce_scalar(self, scalar: Union[float, torch.Tensor]) -> float:
        """
        All-reduce a single scalar across all GPUs.
        
        Args:
            scalar: Scalar value to reduce (sum across all ranks)
            
        Returns:
            Reduced scalar value
        """
        if not self.is_distributed or self.world_size == 1:
            return float(scalar)
            
        # Convert to tensor if needed
        if isinstance(scalar, (int, float)):
            tensor = torch.tensor(scalar, dtype=self.reduce_dtype)
        else:
            tensor = scalar.to(dtype=self.reduce_dtype)
            
        # Ensure tensor is on correct device
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            
        # All-reduce sum
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor.item()
        except Exception as e:
            self.logger.error(f"Failed to reduce scalar: {e}")
            return float(scalar)  # Fallback to original value
            
    def reduce_scalars(self, *scalars: Union[float, torch.Tensor]) -> Tuple[float, ...]:
        """
        All-reduce multiple scalars in a single operation.
        
        Args:
            *scalars: Variable number of scalar values
            
        Returns:
            Tuple of reduced scalar values
        """
        if not self.is_distributed or self.world_size == 1:
            return tuple(float(s) for s in scalars)
            
        if len(scalars) == 0:
            return ()
            
        # Pack scalars into a tensor
        scalar_values = []
        for s in scalars:
            if isinstance(s, (int, float)):
                scalar_values.append(float(s))
            else:
                scalar_values.append(s.item())
                
        tensor = torch.tensor(scalar_values, dtype=self.reduce_dtype)
        
        # Ensure tensor is on correct device
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            
        # All-reduce
        try:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tuple(tensor.tolist())
        except Exception as e:
            self.logger.error(f"Failed to reduce scalars: {e}")
            return tuple(float(s) for s in scalars)  # Fallback
            
    def reduce_mean(self, scalar: Union[float, torch.Tensor]) -> float:
        """
        All-reduce a scalar and divide by world size to get mean.
        
        Args:
            scalar: Scalar value to average across ranks
            
        Returns:
            Mean scalar value across all ranks
        """
        reduced_sum = self.reduce_scalar(scalar)
        return reduced_sum / self.world_size
        
    def reduce_statistics(self, values: torch.Tensor) -> Dict[str, float]:
        """
        Compute distributed statistics (mean, std, min, max) for a tensor.
        
        Args:
            values: Tensor values to compute statistics for
            
        Returns:
            Dictionary with distributed statistics
        """
        if not self.is_distributed or self.world_size == 1:
            return {
                "mean": values.mean().item(),
                "std": values.std().item(),
                "min": values.min().item(),
                "max": values.max().item(),
                "count": values.numel()
            }
            
        # Local statistics
        local_sum = values.sum()
        local_sum_sq = (values ** 2).sum()
        local_min = values.min()
        local_max = values.max()
        local_count = torch.tensor(values.numel(), dtype=self.reduce_dtype)
        
        # Ensure on correct device
        if torch.cuda.is_available():
            local_sum = local_sum.cuda()
            local_sum_sq = local_sum_sq.cuda()
            local_min = local_min.cuda()
            local_max = local_max.cuda()
            local_count = local_count.cuda()
            
        try:
            # Reduce sum and count
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            
            # Reduce min and max
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            
            # Compute global statistics
            global_count = local_count.item()
            global_mean = local_sum.item() / global_count
            global_var = local_sum_sq.item() / global_count - global_mean ** 2
            global_std = max(0, global_var) ** 0.5  # Protect against negative variance due to numerical error
            
            return {
                "mean": global_mean,
                "std": global_std,
                "min": local_min.item(),
                "max": local_max.item(),
                "count": global_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to reduce statistics: {e}")
            # Fallback to local statistics
            return {
                "mean": values.mean().item(),
                "std": values.std().item(),
                "min": values.min().item(),
                "max": values.max().item(),
                "count": values.numel()
            }
            
    def synchronize_random_state(self, seed: Optional[int] = None) -> int:
        """
        Synchronize random state across all ranks.
        
        Args:
            seed: Optional seed (rank 0 will broadcast its seed if not provided)
            
        Returns:
            Synchronized seed value
        """
        if not self.is_distributed or self.world_size == 1:
            if seed is None:
                seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            return seed
            
        try:
            if seed is None and self.rank == 0:
                # Rank 0 generates seed
                seed = torch.randint(0, 2**31, (1,)).item()
                
            # Broadcast seed from rank 0
            seed_tensor = torch.tensor(seed if seed is not None else 0, dtype=torch.long)
            if torch.cuda.is_available():
                seed_tensor = seed_tensor.cuda()
                
            dist.broadcast(seed_tensor, src=0)
            final_seed = seed_tensor.item()
            
            # Set seed on all ranks
            torch.manual_seed(final_seed)
            
            return final_seed
            
        except Exception as e:
            self.logger.error(f"Failed to synchronize random state: {e}")
            fallback_seed = 42
            torch.manual_seed(fallback_seed)
            return fallback_seed
            
    def gather_results(self, local_results: Dict[str, Any], root: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Gather results from all ranks to root rank.
        
        Args:
            local_results: Local results dictionary
            root: Root rank to gather to
            
        Returns:
            List of all results (only valid on root rank)
        """
        if not self.is_distributed or self.world_size == 1:
            return [local_results]
            
        try:
            # Convert to tensors that can be gathered
            # This is simplified - in practice you might need more sophisticated serialization
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local_results)
            
            if self.rank == root:
                return gathered
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to gather results: {e}")
            return [local_results] if self.rank == root else None
            
    def barrier(self) -> None:
        """Synchronization barrier across all ranks."""
        if self.is_distributed and self.world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                self.logger.error(f"Barrier failed: {e}")
                
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
        
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about distributed communication setup."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "is_distributed": self.is_distributed,
            "reduce_dtype": str(self.reduce_dtype),
            "backend": dist.get_backend() if self.is_distributed else None,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
    def validate_distributed_setup(self) -> bool:
        """
        Validate that distributed setup is working correctly.
        
        Returns:
            True if validation passes
        """
        if not self.is_distributed:
            self.logger.info("Single-process mode - distributed validation skipped")
            return True
            
        try:
            # Test scalar reduction
            test_value = float(self.rank + 1)  # Each rank contributes different value
            reduced_value = self.reduce_scalar(test_value)
            
            # Expected sum: 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
            expected_sum = self.world_size * (self.world_size + 1) // 2
            
            if abs(reduced_value - expected_sum) > 1e-6:
                self.logger.error(f"Scalar reduction test failed: got {reduced_value}, expected {expected_sum}")
                return False
                
            # Test barrier
            self.barrier()
            
            self.logger.info("Distributed validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Distributed validation failed: {e}")
            return False