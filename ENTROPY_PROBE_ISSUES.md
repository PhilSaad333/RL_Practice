# Entropy Probe Open Issues

## Status: ON HOLD per Lord Krang (Aug 21, 2025)

The dual-buffer entropy probe implementation is functionally working but has remaining issues. This document tracks the open problems for future resolution.

## Working Components ✅

1. **Dual-buffer infrastructure**: Successfully implemented sequential collection with immediate cleanup
2. **Memory management**: Gradient accumulation fixes resolved 26GB allocation issue
3. **Non-zero entropy values**: Entropy probe now produces meaningful values (previously was identically zero)
4. **Sequential approach**: Using .backward() instead of torch.autograd.grad() successfully computes gradients

## Open Issues ❌

### 1. Tensor Dimension Mismatch (Critical)
**Error**: `The size of tensor a (372) must match the size of tensor b (200)`
- Occurs during entropy gradient computation
- Related to variable sequence lengths in rollout batches
- Causes fallback to sequential approach instead of dual-buffer
- **Location**: `simple_entropy_probe.py:compute_entropy_gradients_only()`

### 2. torch.autograd.grad() Returns All None Gradients
**Problem**: Clean gradient accumulation approach fails in DDP context
- All 392/392 gradients return as None when using torch.autograd.grad()
- Attempted fixes:
  - Detaching centered_log_probs
  - Using DDP-wrapped parameters vs unwrapped
  - Various retain_graph configurations
- **Root cause**: Unknown - gradients work fine with .backward() but not autograd.grad()

### 3. Extremely Large Entropy Values
**Problem**: Delta H values around 1e6 magnitude
- May be artifact of small batch sizes during testing
- Unclear if this represents actual instability or measurement issue
- **Needs investigation**: Test with production batch sizes

## Technical Details

### Dual-Buffer Approach
```python
# 1. Collect entropy buffer (half size)
entropy_per_rank = per_rank // 2
entropy_rb = self.collector.collect_batch(batch_prompts=entropy_per_rank)

# 2. Convert and compute entropy gradients with gradient accumulation
entropy_batch = entropy_rb.to_batch(device=f"cuda:{self.local_rank}")
entropy_grads = self.algo.simple_entropy_probe.compute_entropy_gradients_only(
    entropy_rollouts=entropy_batch,
    trainable_params=ddp_trainable_params,
    policy_model=self.model,
    cfg=self.cfg,
    step_number=self.step_id + 1,
    microbatch_size=self.cfg["microbatch_size"]
)

# 3. Immediate cleanup
del entropy_rb
torch.cuda.empty_cache()
```

### Memory Optimizations Applied
- `rollout_batch_size`: 24 → 12 (50% reduction)
- `tf_micro_batch`: 16 → 12 (25% reduction)  
- `buffer_size`: 128 → 64 (50% reduction)
- Gradient accumulation implemented to match training pattern

## Files Modified

1. **rl_training/algs/simple_entropy_probe.py**
   - Added `compute_entropy_gradients_only()` method
   - Implemented microbatch processing for memory efficiency

2. **rl_training/algs/dr_grpo.py**  
   - Modified `step()` and `_backward_and_step()` to accept `entropy_grads`
   - Integrated dual-buffer approach

3. **rl_training/runners/rl_runner.py**
   - Implemented dual-buffer collection logic
   - Added immediate cleanup after entropy gradient computation

4. **rl_training/cfg/entropy_probe_test_optimized.yaml**
   - Progressively optimized memory settings
   - Enabled simple_entropy_probe with debug output

## Next Steps (When Resumed)

1. **Fix tensor dimension mismatch**
   - Investigate variable sequence length handling in entropy gradient computation
   - Ensure proper padding/masking for different sequence lengths

2. **Investigate torch.autograd.grad() failure**
   - Debug why gradients are None in DDP context
   - Consider if this is a fundamental limitation or solvable issue

3. **Validate entropy magnitude**
   - Test with production batch sizes to see if large values persist
   - Compare with theoretical expected entropy ranges

4. **Production testing**
   - Once issues resolved, test with overnight_config.yaml settings
   - Validate entropy probe stability over longer training runs

## Configuration Reference

**Test Settings Used**:
- rollout_batch_size: 12 per GPU (24 total)
- tf_micro_batch: 12 per GPU (24 total)
- buffer_size: 64
- microbatch_size: 4
- total_steps: 10
- simple_entropy_probe.enabled: true