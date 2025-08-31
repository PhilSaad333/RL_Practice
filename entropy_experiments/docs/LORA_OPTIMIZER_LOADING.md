# LoRA Optimizer State Loading - Technical Documentation

## Overview

Loading optimizer states for LoRA-adapted models requires careful handling of parameter mismatches and group structures. This document explains the challenges and solutions implemented in the entropy experiments.

## The Problem

When using LoRA (Low-Rank Adaptation) with saved optimizer states, several issues arise:

### 1. Parameter Count Mismatch
- **Total model parameters**: 730 (392 LoRA + 338 base model)
- **Trainable parameters**: 392 (LoRA adapters only)
- **Saved optimizer state**: 392 parameters (only LoRA adapters were trained)

**Critical Bug**: The original code used `model.parameters()` (730 total) instead of trainable-only parameters when creating the optimizer, causing a 392→730 parameter mismatch.

### 2. Parameter Group Structure Mismatch
- **Saved optimizer**: 2 parameter groups (392 LoRA params + 0 base params)
- **Current optimizer**: 1 parameter group (392 LoRA params)

This happens because some LoRA training frameworks create separate parameter groups for LoRA vs base model parameters, even if the base group ends up empty.

### 3. Parameter ID Remapping
PyTorch model reconstruction creates new parameter IDs that don't match the saved optimizer state keys, requiring custom ID mapping based on parameter order.

## The Solution

### Fixed Parameter Inclusion (offline_entropy_probe.py:379-384)

```python
# OLD (BUGGY): Includes all parameters (730 total)
optimizer = optim.AdamW(
    self.model.parameters(),  # ← BUG: includes frozen base model params
    lr=lr, weight_decay=weight_decay
)

# NEW (FIXED): Only trainable LoRA parameters
trainable_params = [p for p in self.model.parameters() if p.requires_grad]
optimizer = optim.AdamW(
    trainable_params,  # ← CORRECT: only LoRA adapters (392 params)
    lr=lr, weight_decay=weight_decay
)
```

### Parameter Group Structure Consolidation (offline_entropy_probe.py:485-528)

```python
if len(saved_param_groups) != len(current_optimizer.param_groups):
    self.logger.warning(f"Parameter group count mismatch: saved={len(saved_param_groups)}, current={len(current_optimizer.param_groups)}")
    self.logger.info("Consolidating saved parameter groups to match current structure")
    
    # Consolidate all saved parameters into current group structure
    all_new_param_ids = []
    merged_group_config = {}
    
    # Collect all successfully remapped parameter IDs
    for group in saved_param_groups:
        if 'params' in group and len(group['params']) > 0:
            # Use configuration from the non-empty group
            if not merged_group_config:
                merged_group_config = {k: v for k, v in group.items() if k != 'params'}
            
            # Add remapped parameter IDs
            for old_id in group['params']:
                if old_id in id_mapping:
                    all_new_param_ids.append(id_mapping[old_id])
    
    # Create remapped groups to match current structure
    for i, current_group in enumerate(current_optimizer.param_groups):
        if i == 0:
            # First group gets all the parameters
            new_group = merged_group_config.copy()
            new_group['params'] = all_new_param_ids
        else:
            # Additional groups (if any) get current group config but no params
            new_group = {k: v for k, v in current_group.items() if k != 'params'}
            new_group['params'] = []
        remapped_param_groups.append(new_group)
```

### Parameter ID Remapping (offline_entropy_probe.py:424-532)

The `_remap_optimizer_state_ids` function handles PyTorch's parameter ID changes during model reconstruction:

1. **Extract parameter order**: Both saved and current optimizers maintain parameter order
2. **Create ID mapping**: Map old parameter IDs to new ones based on position  
3. **Remap optimizer state**: Update all `exp_avg`, `exp_avg_sq`, etc. with new IDs
4. **Handle group mismatches**: Consolidate parameter groups when structure differs

## Verification

The debug script `entropy_experiments/debug_optimizer_loading.py` verifies the fixes:

```
======================================================================
OPTIMIZER STATE LOADING DEBUG
======================================================================
1. Loading model...
   Total parameters: 730
   Trainable parameters: 392
   Frozen parameters: 338

2. Loading saved optimizer state...
   Saved parameter groups: 2
   Saved parameters total: 392
   Parameter group details:
     Group 0: 392 params, lr=0.0, weight_decay=0.0
     Group 1: 0 params, lr=0.0, weight_decay=0.0

4. FIXED APPROACH:
   Fixed optimizer sees: 392 parameters (trainable only)
   Perfect match: 392 saved = 392 current
   Current parameter group details:
     Group 0: 392 params, lr=0.0, weight_decay=0.0
   Group structure mismatch: 2 saved vs 1 current

5. TESTING PARAMETER REMAPPING:
   ✅ SUCCESS: Optimizer state loaded successfully!
   ✅ Adam state loaded for 392/392 parameters
   ✅ exp_avg_sq found - Adam preconditioning will work!
```

## Why This Matters

The entropy experiments require **actual Adam states** (`exp_avg_sq`) for the `AdamPreconditioner` component, not just fresh optimizer initialization. This enables:

1. **Rao-Blackwell estimation**: Uses Adam's second moment estimates for variance reduction
2. **Proper gradient preconditioning**: Applies P^{-1/2} using loaded Adam states
3. **Accurate entropy computation**: δH₁ estimates depend on proper preconditioning

## Key Takeaways

1. **Always filter for trainable parameters** when creating optimizers for LoRA models
2. **Handle parameter group structure mismatches** - different training frameworks create different group structures
3. **Parameter ID remapping is essential** - PyTorch model reconstruction changes parameter IDs
4. **Test with debug scripts** - Complex optimizer loading requires verification
5. **LoRA context matters** - Understanding the 392 trainable vs 730 total parameter split is crucial

## Files Modified

- `entropy_experiments/offline_entropy_probe.py`: Fixed parameter inclusion and group handling
- `entropy_experiments/debug_optimizer_loading.py`: Comprehensive debugging and verification
- `entropy_experiments/configs/test_deltaH1.yaml`: Complete test configuration

## Testing

Run the debug script to verify optimizer loading:
```bash
cd ~/RL_Practice
PYTHONPATH=. python entropy_experiments/debug_optimizer_loading.py
```

Expected output: "✅ SUCCESS: Optimizer state loaded successfully!"