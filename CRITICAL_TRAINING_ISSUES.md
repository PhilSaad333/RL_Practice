# Critical Training Issues and Fixes

## 🚨 Reference Model KL Computation Bug

**Issue**: When resuming training from a checkpoint, the reference model for KL divergence is incorrectly set to the model state at the resume step, rather than the original fine-tuned checkpoint.

### Current Behavior (INCORRECT)
```python
# In rl_runner.py line 112
self.ref_model = copy.deepcopy(_unwrap(self.model)).eval().requires_grad_(False)
```

**Problem Flow**:
1. Initial training: `ref_model` = fine-tuned checkpoint ✅
2. Resume from step N: Load model state → `ref_model` = model at step N ❌
3. KL divergence computed against **step N model** instead of **original fine-tuned model**

### Correct Behavior (NEEDED)
The reference model should **always** be the original fine-tuned checkpoint, regardless of resume step.

```python
# PROPOSED FIX: Store original checkpoint path and reload for ref_model
def _create_reference_model(self):
    """Create reference model from original checkpoint, not current model state."""
    if hasattr(self, 'original_ckpt_path') and self.original_ckpt_path:
        # Load original fine-tuned checkpoint for KL reference
        ref_model = self._load_model_from_checkpoint(self.original_ckpt_path)
    else:
        # Fallback: use current model (initial training case)
        ref_model = copy.deepcopy(_unwrap(self.model))
    
    return ref_model.eval().requires_grad_(False)
```

### Impact
- **Training stability**: KL penalty computed against wrong baseline
- **Reproducibility**: Different KL values when resuming vs continuous training
- **Scientific validity**: Training dynamics change based on resume point

### Solution Requirements
1. **Store original checkpoint path** during initialization
2. **Always load reference model from original checkpoint**
3. **Independent of current model state** (training or resumed)
4. **Preserve across all resume operations**

### Files to Modify
- `rl_training/runners/rl_runner.py`: Fix reference model creation
- Configuration files: Document checkpoint path requirements
- Resume workflow: Ensure original checkpoint path preservation

---

## 🎯 Priority: HIGH
This affects **all resumed training runs** and could explain training instabilities or unexpected KL behavior during long runs with checkpointing.

## 🧪 Test Plan
1. Start training from fine-tuned checkpoint
2. Resume from intermediate step
3. Verify KL divergence computed against **original fine-tuned model**, not resume step model
4. Compare KL trajectories: continuous vs resumed training should match