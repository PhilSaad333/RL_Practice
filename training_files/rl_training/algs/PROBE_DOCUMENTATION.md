# RL Training Probes Documentation

This document describes the various diagnostic probes available in the RL training pipeline. Probes are optional measurement tools that provide insights into training dynamics without affecting the training process itself.

## Table of Contents
1. [Gradient Noise Scale (GNS) Probe](#gradient-noise-scale-gns-probe)
2. [Entropy Probe](#entropy-probe) (documentation pending)

---

## Gradient Noise Scale (GNS) Probe

### Overview
The GNS probe measures the ratio of gradient noise to gradient signal during training, providing an estimate of the "critical batch size" - the batch size where noise and signal are roughly equal. This helps determine whether your current batch size is optimal for training efficiency.

### Motivation
- **Too small batch size**: Gradient noise dominates → unstable training, poor convergence
- **Too large batch size**: Diminishing returns → wasted computation, longer wall-clock time
- **Optimal batch size**: Balances noise reduction with computational efficiency

The GNS probe helps you find this sweet spot without expensive hyperparameter sweeps.

### What We Compute

#### Core Metric: Gradient Noise Scale (B_simple)
For a gradient estimator with batch size B:
```
E[||g_B||²] = ||G||² + tr(Σ)/B
```
Where:
- `G` = true gradient (expected value)
- `Σ` = per-example gradient covariance matrix
- `g_B` = mini-batch gradient estimate

The "simple" gradient noise scale is:
```
B_simple = tr(Σ) / ||G||²
```

This represents the batch size where the noise (tr(Σ)/B) equals the signal (||G||²).

#### Micro-batch Level Estimation
With gradient accumulation over m micro-batches:
- Track incremental micro-gradients: `g_i = grad_after_i - grad_after_(i-1)`
- Compute: `E[||g_i||²] - ||G||²` estimates `tr(Σ_micro)`
- Scale to per-example: `B_simple ≈ gns_mu × B_micro_global`

### Implementation Strategy

#### Zero-Overhead Design
The probe adds negligible overhead by:
1. **No extra forward/backward passes** - uses existing training gradients
2. **Incremental computation** - tracks gradient changes during accumulation
3. **Minimal communication** - only 4 scalars all-reduced per optimizer step
4. **CPU gradient storage** - snapshots moved to CPU to avoid GPU memory pressure

#### DDP (Distributed Data Parallel) Compatibility
The implementation carefully handles distributed training:
- During accumulation (`no_sync`): Track local incremental gradients
- On sync step: Use globally-averaged gradients for ||G||²
- All-reduce: Sum micro-gradient norms and counts across ranks
- Synchronized: Probe operations align with training's natural sync points

#### Weighted Loss Handling (Dr-GRPO Specific)
Dr-GRPO uses weighted losses where each generation has weight:
```
w_{p,i} = 1 / (G × L_max(p))
```

To account for this, we compute the Effective Sample Size (ESS):
```
ESS = (Σ w_i)² / (Σ w_i²)
```

This gives the equivalent number of equally-weighted samples for variance reduction.

### Algorithm Details

#### Per Micro-batch (after backward):
```python
1. Take gradient snapshot: snap = flatten(all .grad tensors)
2. Compute increment: g_micro = snap - prev_snap
3. Accumulate: sum_sq += ||g_micro||²
4. Track ESS weights: sum_w += weights, sum_w2 += weights²
5. Update: prev_snap = snap, m += 1
```

#### On Sync Step (before optimizer.step):
```python
1. Compute ||G||² from synchronized gradients
2. All-reduce across ranks: (sum_sq, m, sum_w, sum_w2)
3. Compute metrics:
   - E[||g||²] = sum_sq_total / m_total
   - tr(Σ_micro) = E[||g||²] - ||G||²
   - gns_mu = tr(Σ_micro) / ||G||²
   - ESS = (sum_w)² / sum_w2
   - B_simple = gns_mu × ESS
```

### Output Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `gns_mu` | Gradient noise scale at micro-batch level | Noise-to-signal ratio (lower = less noisy) |
| `Bsimple_from_mu` | Critical batch size estimate | Batch size where noise ≈ signal |
| `gns_ess` | Effective sample size | Accounts for weighted losses |
| `gns_consistency_ratio` | Validation metric | Should ≈ 1.0 (sanity check) |
| `gns_G2` | Global gradient L2 norm squared | Signal strength |
| `gns_tr_sigma` | Trace of gradient covariance | Total gradient variance |

### Usage Guidelines

#### Enabling the Probe
```yaml
gns_probe:
  enabled: true  # false by default
```

#### Interpreting Results
1. **During early training**: `gns_mu` often starts small (low noise)
2. **As loss decreases**: `gns_mu` typically increases (harder optimization)
3. **Batch size selection**:
   - If `batch_size < Bsimple_from_mu`: Consider increasing batch size
   - If `batch_size >> Bsimple_from_mu`: Could reduce batch size for efficiency
4. **ESS vs raw count**: `gns_ess < B×G` shows impact of weight inequality

#### Common Patterns
- **Stable training**: `Bsimple_from_mu` relatively constant after warmup
- **Approaching convergence**: `Bsimple_from_mu` increases (need larger batches)
- **Overfitting**: Very high `Bsimple_from_mu` (gradient signal weak)

### Technical Notes

#### Memory Considerations
- Gradient snapshots: 2 × model_size × 4 bytes (float32) CPU memory
- Negligible GPU impact (snapshots immediately moved to CPU)

#### Numerical Stability
- Clamping: `tr(Σ) = max(E[||g||²] - ||G||², 0)` ensures non-negative variance
- Division guards: Uses `max(denominator, 1e-12)` to avoid division by zero

#### Limitations
1. Assumes gradient noise is approximately isotropic
2. "Simple" GNS ignores momentum/Adam preconditioning effects
3. Critical batch size is a guideline, not a hard threshold
4. Most useful for large-scale training where batch size matters

### References
- McCandlish et al. (2018): "An Empirical Model of Large-Batch Training"
- Appendix A provides mathematical identities for gradient noise scale
- PyTorch DDP documentation for distributed gradient averaging

---

## Entropy Probe

*Documentation to be added when probe is finalized*

The entropy probe analyzes the relationship between policy entropy and advantage estimates to understand exploration-exploitation dynamics during RL training.

Key measurements:
- Per-sequence entropy statistics
- Entropy-advantage correlation
- Projected entropy changes under policy updates

---

## Adding New Probes

When implementing new diagnostic probes, follow these guidelines:

1. **Zero-impact principle**: Probes should not affect training dynamics
2. **Configuration**: Add enable/disable flag in config
3. **DDP compatibility**: Handle distributed training correctly
4. **Memory efficiency**: Move large tensors to CPU when possible
5. **Documentation**: Update this file with probe description

### Probe Interface Pattern
```python
class YourProbe:
    def __init__(self, cfg):
        self._enabled = cfg.get("your_probe", {}).get("enabled", False)
    
    def accumulate(self, batch_data):
        if not self._enabled:
            return
        # Accumulate statistics
    
    def compute_and_log(self):
        if not self._enabled:
            return {}
        # Compute final metrics
        return metrics_dict
```