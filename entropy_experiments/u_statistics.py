"""
U-Statistics Calculator

Implements U-statistic computation and variance estimation for the entropy probe.
Based on Section IV.C of offline_entropy_probe_strategy.txt.

Provides both plug-in and jackknife variance estimates for the cross-prompt U-statistic.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
import math


class UStatisticsCalculator:
    """
    Computes U-statistics and variance estimates for entropy probe analysis.
    
    The key U-statistic is:
    U_B^cross = [B/(B-1)] * (X̄ * Ȳ) - [1/(B(B-1))] * Σ_n (X_n * Y_n)
    
    This provides an unbiased estimator for the cross-prompt entropy kernel.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Variance estimation settings
        self.compute_plugin_se = config['stats_config']['compute_plugin_se']
        self.compute_jackknife_se = config['stats_config']['compute_jackknife_se']
        
        self.logger.info(f"UStatisticsCalculator initialized: plugin_se={self.compute_plugin_se}, jackknife_se={self.compute_jackknife_se}")
        
    def compute_variance_estimates(self, U_cross: float, U: int, mode: str, 
                                 prompt_contributions: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compute variance estimates and standard errors for U-statistic.
        
        UPDATED (P2): Use row means (prompt_contributions) for both estimators.
        
        Args:
            U_cross: The computed U-statistic value
            U: Number of units (B for exact mode, M for blocks mode)
            mode: "exact" or "blocks" computation mode
            prompt_contributions: Row means r_u for each unit (required for both estimators)
            
        Returns:
            Dictionary with variance estimates and standard errors
        """
        results = {}
        
        if not prompt_contributions:
            self.logger.warning("No row means provided - variance estimates will be invalid")
            return {
                "se_plugin": float('inf'),
                "zeta1_plugin": float('inf'),
                "se_jack": float('inf'),
                "zeta1_jack": float('inf')
            }
        
        if self.compute_plugin_se:
            plugin_results = self._compute_plugin_variance(U_cross, U, prompt_contributions)
            results.update(plugin_results)
            
        if self.compute_jackknife_se:
            jackknife_results = self._compute_jackknife_variance(U_cross, U, prompt_contributions)
            results.update(jackknife_results)
            
        return results
        
    def _compute_plugin_variance(self, U_cross: float, U: int, row_means: List[float]) -> Dict[str, Any]:
        """
        Compute plug-in variance estimate using exact formula from fix guide.
        
        CRITICAL FIX (P2): Implement real plug-in ζ̂₁ estimator:
        zeta1_plugin = (U / (4*(U-1))) * Σ_u (r_u - r̄)²
        se_plugin = sqrt(4 * zeta1_plugin / U)
        
        Args:
            U_cross: U-statistic value
            U: Number of units (B for exact mode, M for blocks mode)
            row_means: List of row means r_u for each unit
        """
        if U <= 1 or not row_means:
            return {
                "se_plugin": float('inf'),
                "zeta1_plugin": float('inf'),
                "var_plugin": float('inf')
            }
        
        # Compute r̄ = mean of row means
        r_bar = sum(row_means) / len(row_means)
        
        # Compute sum of squared deviations: Σ_u (r_u - r̄)²
        sum_sq_dev = sum((r_u - r_bar) ** 2 for r_u in row_means)
        
        # Plug-in ζ̂₁ estimator
        zeta1_plugin = (U / (4 * (U - 1))) * sum_sq_dev
        
        # Plug-in variance and standard error
        var_plugin = 4 * zeta1_plugin / U
        se_plugin = math.sqrt(var_plugin)
        
        self.logger.debug(f"Plug-in: r̄={r_bar:.6f}, Σ(r_u-r̄)²={sum_sq_dev:.6f}, ζ̂₁={zeta1_plugin:.6f}, SE={se_plugin:.6f}")
        
        return {
            "se_plugin": se_plugin,
            "zeta1_plugin": zeta1_plugin,
            "var_plugin": var_plugin,
            "r_bar": r_bar,
            "sum_sq_deviations": sum_sq_dev
        }
        
    def _compute_jackknife_variance(self, U_cross: float, U: int, 
                                  row_means: List[float]) -> Dict[str, Any]:
        """
        Compute delete-1 jackknife variance estimate using exact formula from fix guide.
        
        CRITICAL FIX (P2): Implement exact delete-1 jackknife for order-2 U-statistics:
        U_minus_u = (U*U_cross - 2*r_u) / (U-2)
        var_jack = ((U-1)/U) * Σ_u (U_minus_u - Ūbar_minus)²
        
        Args:
            U_cross: U-statistic value
            U: Number of units
            row_means: List of row means r_u for each unit
        """
        if U <= 2 or len(row_means) != U:
            return {
                "se_jack": float('inf'),
                "zeta1_jack": float('inf'),
                "var_jack": float('inf')
            }
            
        # Compute delete-1 jackknife estimates using exact formula for order-2 U-statistics
        jackknife_estimates = []
        
        for u in range(U):
            r_u = row_means[u]
            # Exact delete-1 formula for order-2 U-statistic
            if U > 2:
                U_minus_u = (U * U_cross - 2 * r_u) / (U - 2)
            else:
                U_minus_u = 0.0  # Edge case
            jackknife_estimates.append(U_minus_u)
            
        # Compute jackknife mean
        U_bar_minus = sum(jackknife_estimates) / U
        
        # Compute jackknife variance with correct scaling
        sum_sq_jack = sum((U_minus_u - U_bar_minus) ** 2 for U_minus_u in jackknife_estimates)
        var_jack = ((U - 1) / U) * sum_sq_jack
        
        se_jack = math.sqrt(var_jack)
        
        # Estimate ζ₁ from jackknife variance 
        zeta1_jack = (U / 4) * var_jack
        
        self.logger.debug(f"Jackknife: Ūbar_minus={U_bar_minus:.6f}, var_jack={var_jack:.6f}, ζ̂₁_jack={zeta1_jack:.6f}, SE={se_jack:.6f}")
        
        return {
            "se_jack": se_jack,
            "zeta1_jack": zeta1_jack,
            "var_jack": var_jack,
            "jackknife_estimates": jackknife_estimates,
            "U_bar_minus": U_bar_minus
        }
        
    def compute_cross_prompt_ustatistic(self, x_values: List[float], y_values: List[float]) -> Dict[str, Any]:
        """
        Compute cross-prompt U-statistic directly from X and Y values.
        
        Args:
            x_values: List of X_n values (entropy side)
            y_values: List of Y_n values (objective side)
            
        Returns:
            Dictionary with U-statistic and components
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have same length")
            
        B = len(x_values)
        if B <= 1:
            return {"U_cross": 0.0, "B": B, "valid": False}
            
        # Compute means
        x_bar = sum(x_values) / B
        y_bar = sum(y_values) / B
        
        # Compute diagonal term
        diagonal_sum = sum(x * y for x, y in zip(x_values, y_values))
        
        # U-statistic formula: U_B^cross = [B/(B-1)] * (X̄ * Ȳ) - [1/(B(B-1))] * Σ(X_n * Y_n)
        cross_term = (B / (B - 1)) * x_bar * y_bar
        diagonal_term = diagonal_sum / (B * (B - 1))
        U_cross = cross_term - diagonal_term
        
        return {
            "U_cross": U_cross,
            "x_bar": x_bar,
            "y_bar": y_bar,
            "cross_term": cross_term,
            "diagonal_term": diagonal_term,
            "diagonal_contributions": [x * y for x, y in zip(x_values, y_values)],
            "B": B,
            "valid": True
        }
        
    def compute_block_ustatistic(self, block_x_values: List[float], block_y_values: List[float],
                               cross_contributions: List[List[float]]) -> Dict[str, Any]:
        """
        Compute block-level U-statistic.
        
        Args:
            block_x_values: X̃_b values for each block b
            block_y_values: Ỹ_b values for each block b
            cross_contributions: cross_contributions[b][c] = X̃_b · Ỹ_c for b ≠ c
            
        Returns:
            Dictionary with block U-statistic
        """
        M = len(block_x_values)
        if M != len(block_y_values):
            raise ValueError("block_x_values and block_y_values must have same length")
            
        if M <= 1:
            return {"U_cross": 0.0, "M": M, "valid": False}
            
        # Block means
        x_bar_blocks = sum(block_x_values) / M
        y_bar_blocks = sum(block_y_values) / M
        
        # Cross-block contributions
        cross_sum = 0.0
        pair_count = 0
        
        for b in range(M):
            for c in range(M):
                if b != c:
                    if b < len(cross_contributions) and c < len(cross_contributions[b]):
                        cross_sum += cross_contributions[b][c]
                        pair_count += 1
                        
        # Diagonal contributions
        diagonal_sum = sum(x * y for x, y in zip(block_x_values, block_y_values))
        
        # Block U-statistic
        if pair_count > 0:
            cross_term = (M / (M - 1)) * x_bar_blocks * y_bar_blocks
            diagonal_term = diagonal_sum / (M * (M - 1))
            U_cross = cross_term - diagonal_term
        else:
            U_cross = 0.0
            
        return {
            "U_cross": U_cross,
            "x_bar_blocks": x_bar_blocks,
            "y_bar_blocks": y_bar_blocks,
            "cross_sum": cross_sum,
            "diagonal_sum": diagonal_sum,
            "pair_count": pair_count,
            "M": M,
            "valid": pair_count > 0
        }
        
    def validate_ustatistic_computation(self, test_size: int = 10) -> bool:
        """
        Validate U-statistic computation with synthetic data.
        
        Returns:
            True if validation passes
        """
        try:
            # Generate synthetic test data
            x_values = [float(i) + 0.1 for i in range(test_size)]
            y_values = [float(i) * 0.5 + 0.2 for i in range(test_size)]
            
            # Test exact U-statistic
            result = self.compute_cross_prompt_ustatistic(x_values, y_values)
            
            if not result['valid']:
                self.logger.error("U-statistic computation returned invalid result")
                return False
                
            # Check for reasonable values
            U_cross = result['U_cross']
            if math.isnan(U_cross) or math.isinf(U_cross):
                self.logger.error(f"U-statistic produced invalid value: {U_cross}")
                return False
                
            # Test variance estimation
            variance_result = self.compute_variance_estimates(
                U_cross=U_cross, 
                B=test_size,
                mode="exact",
                prompt_contributions=result['diagonal_contributions']
            )
            
            # Check variance estimates are finite
            for key in variance_result:
                value = variance_result[key]
                if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
                    self.logger.error(f"Variance estimate {key} is invalid: {value}")
                    return False
                    
            self.logger.info("U-statistics validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"U-statistics validation failed: {e}")
            return False