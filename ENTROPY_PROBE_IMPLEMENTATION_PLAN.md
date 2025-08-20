# Entropy Probe Implementation Plan

## Problem Analysis

### Current State
- Entropy probe computes Fisher kernel per **microbatch** (2 prompts × 8 gens = 16 sequences)
- Theory requires computation over **full buffer** (32-128 prompts × 8 gens = 256-1024 sequences)
- Need to handle both single-GPU and multi-GPU scenarios efficiently

### Theoretical Requirements

The first-order entropy change formula is:
```
δH = -η * (1/(B*G)) * Σ_{i=1}^{B*G} * (1/L_max(p_i)) * E_t[(S(t) - S̄) * K_1(t,t_i') * A(t_i')]
```

Where the Fisher kernel is:
```
K_1(t,t') = Σ_α ∂_α S(t) ∂_α S(t') P_α
```

And `B*G` should be the **full buffer size**, not microbatch size.

## Key Insights

### 1. δH vs Full Fisher Kernel Computation
For routine entropy change estimation, we don't need the full Fisher kernel matrix K₁(t,t'). 

The entropy change can be computed as:
```
δH ≈ Σ_α (∂_α H) × (δθ_α) × P_α
```

Where:
- `∂_α H = -E_t[(S(t) - S̄) × ∂_α S(t)]` (already aggregated over sequences)
- `δθ_α = η × ∂_α J` (the actual parameter update)
- `P_α` are Adam conditioning factors

This is much cheaper than computing the full N×N Fisher kernel.

### 2. Multi-GPU Considerations
In distributed training:
- Each GPU i computes partial gradients: `A_α^i = (∂_α H)^i` and `B_α^i = (δθ_α)^i`
- Total entropy change: `δH ≈ Σ_α (Σ_i A_α^i) × (Σ_j B_α^j) × P_α`
- This requires communication of gradient statistics, not full per-sequence gradients

### 3. Two-Tier Computational Strategy
- **Routine**: Cheap δH estimation every step (no Fisher kernel storage)
- **Detailed**: Full Fisher kernel computation every N steps (expensive, for analysis)

## Implementation Plan

### Phase 1: Single GPU Full Buffer (Priority 1)
**Goal**: Fix current implementation to compute Fisher kernel over full buffer while respecting memory constraints

**Key Insight**: We can't fit all 256 sequences in GPU memory at once (same constraint as training), so we need microbatch gradient accumulation for the entropy probe.

**Approach**: Microbatch gradient collection + CPU Fisher kernel computation
1. **Before optimizer step**: Process rollout buffer in microbatches to collect per-sequence gradients
2. **Gradient storage**: Store W_α,t = ∂_α log π(t) on CPU for all 256 sequences  
3. **Fisher kernel computation**: Compute 256×256 Fisher kernel on CPU using stored gradients
4. **Memory management**: Never exceed microbatch size in GPU memory

**Implementation Details**:
- Process same rollout buffer (`rb`) used for training, but before optimizer step
- For each microbatch: compute per-sequence gradients, store on CPU
- After all microbatches: compute Fisher kernel K₁(t,t') = Σ_α W_α,t W_α,t' P_α on CPU
- Predict entropy change, then validate against actual change after optimizer step

**Timing Strategy**:
- **Entropy probe**: Before optimizer step (using current model state)
- **Prediction validation**: Compare predicted vs actual entropy change after step

**Adam Preconditioning Strategy**:
The Fisher kernel includes Adam preconditioning factors P_α, but computing the "correct" factors requires the update we're trying to predict (chicken-and-egg problem).

**Solutions**:
1. **No preconditioning**: P_α = 1 (for studying raw Fisher kernel structure)
2. **Previous step factors**: Use Adam EMA state from previous update (smoothly varying)
3. **Current step factors**: Compute entropy probe after optimizer step (future enhancement)

**Configuration**:
```yaml
entropy_probe:
  preconditioning:
    mode: "previous_step"  # "none", "previous_step", "current_step"
    fallback_to_none: true
```

### Phase 2: Efficient δH-Only Computation (Priority 2)
**Goal**: Implement cheap δH estimation without full Fisher kernel

**Approach**: Use the insight that δH = Σ_α (∂_α H) × (δθ_α) × P_α
1. **Gradient aggregation**: Instead of storing per-sequence gradients, compute aggregated `∂_α H`
2. **Parameter updates**: Use actual `δθ_α` from optimizer
3. **Fast computation**: Inner product over parameters, no sequence×sequence matrix

**Benefits**:
- Memory: O(n_params) instead of O(n_sequences × n_params)
- Computation: O(n_params) instead of O(n_sequences²)
- Suitable for routine monitoring

### Phase 3: Multi-GPU δH Computation (Priority 3)
**Goal**: Extend efficient δH computation to distributed training

**Approach**: Aggregate gradient statistics across GPUs
1. **Local computation**: Each GPU computes local `∂_α H` and `δθ_α`
2. **AllReduce operations**: Sum gradient statistics across all GPUs
3. **Global δH**: Compute using aggregated values

**Communication Cost**: O(n_params) per step (same as normal gradient sync)

### Phase 4: Full Multi-GPU Fisher Kernel (Priority 4)
**Goal**: Compute full Fisher kernel across all GPUs (for detailed analysis)

**Approach**: Expensive but comprehensive computation
1. **Gradient collection**: Each GPU stores per-sequence gradients `W_{α,t}`
2. **Gradient sharing**: AllGather operations to collect all gradients on all GPUs
3. **Fisher computation**: Compute full cross-GPU Fisher kernel
4. **Frequency**: Only every N steps due to high cost

**Communication Cost**: O(n_sequences × n_params) - very expensive

### Phase 5: Two-Tier Production System (Priority 5)
**Goal**: Efficient routine monitoring + detailed periodic analysis

**Configuration Options**:
```yaml
entropy_probe:
  routine_monitoring:
    enabled: true
    compute_delta_h: true      # Cheap δH every step
    compute_fisher_kernel: false
  
  detailed_analysis:
    enabled: true
    every_n_steps: 10          # Expensive Fisher kernel every 10 steps
    compute_fisher_kernel: true
    save_gradients: true       # For offline analysis
```

## Technical Implementation Details

### Single GPU Microbatch Gradient Accumulation
```python
# NEW APPROACH: Collect per-sequence gradients in microbatches, compute Fisher kernel on CPU

# In training runner - BEFORE optimizer steps
if entropy_probe.enabled:
    print("[DEBUG] Starting entropy probe gradient collection")
    all_sequence_gradients = []
    all_sequence_metadata = []
    
    # Process buffer in same microbatches as training (respect memory constraints)
    for idx in rollout_buffer.iter_minibatches(microbatch_size):
        microbatch = rollout_buffer.get_batch(idx, device="cuda")
        
        # Compute per-sequence gradients for this microbatch (e.g., 16 sequences)
        mb_gradients, mb_metadata = entropy_probe.compute_microbatch_gradients(
            microbatch, model, trainable_params
        )
        
        # Store on CPU to save GPU memory
        for grad in mb_gradients:
            all_sequence_gradients.append(grad.cpu())
        all_sequence_metadata.extend(mb_metadata)
        
        # Clear GPU memory
        del mb_gradients, microbatch
        torch.cuda.empty_cache()
    
    # Compute Fisher kernel on CPU (256×256 matrix)
    fisher_kernel = entropy_probe.compute_fisher_kernel_cpu(
        all_sequence_gradients, preconditioning_factors
    )
    
    # Predict entropy change
    predicted_delta_h = entropy_probe.compute_delta_h(
        fisher_kernel, all_sequence_metadata, learning_rate
    )
    
    print(f"[EntropyProbe] Predicted δH: {predicted_delta_h}")

# Continue with normal training (microbatch gradient accumulation)
for microbatch in buffer:
    stats = algo.step(microbatch, call_entropy_probe=False)
```

### Efficient δH Computation
```python
def compute_delta_h_efficient(self, rollouts, advantages, log_probs, param_updates):
    """Compute δH without storing full Fisher kernel."""
    
    # Compute aggregated entropy gradients: ∂_α H = -E[(S-S̄) ∂_α S]
    entropy_gradients = self._compute_entropy_gradients(rollouts, log_probs)
    
    # Get actual parameter updates: δθ_α = η ∂_α J
    param_deltas = param_updates  # From optimizer
    
    # Apply Adam conditioning: P_α
    conditioned_deltas = self._apply_conditioning(param_deltas)
    
    # Compute δH = Σ_α (∂_α H) × (δθ_α) × P_α
    delta_h = torch.sum(entropy_gradients * conditioned_deltas)
    
    return delta_h
```

### Multi-GPU Communication
```python
def compute_delta_h_distributed(self, local_entropy_grads, local_param_updates):
    """Compute δH across multiple GPUs."""
    
    # Sum entropy gradients across all GPUs
    global_entropy_grads = dist.all_reduce(local_entropy_grads, op=dist.ReduceOp.SUM)
    
    # Sum parameter updates across all GPUs  
    global_param_updates = dist.all_reduce(local_param_updates, op=dist.ReduceOp.SUM)
    
    # Compute global δH
    delta_h = torch.sum(global_entropy_grads * global_param_updates)
    
    return delta_h
```

## Testing Strategy

### Phase 1 Testing
- Run single GPU with buffer_size=32, verify 256×256 Fisher kernel
- Compare δH estimates with theoretical predictions
- Memory profiling to ensure feasibility

### Phase 2 Testing  
- Compare efficient δH with full Fisher kernel computation
- Verify identical results within numerical precision
- Performance benchmarking

### Phase 3 Testing
- Multi-GPU training with 2-4 GPUs
- Verify δH computation matches single GPU results
- Communication overhead measurement

## Resource Requirements

**Note**: Assumes LoRA fine-tuning with ~90M trainable parameters (not full 1.5B model)

### Memory Estimates (Microbatch Gradient Accumulation Approach)
- **GPU memory**: Same as training (microbatch size only, ~16 sequences)
- **CPU storage**: 256 sequences × 90M params ≈ 90GB (manageable for gradient storage)
- **Fisher kernel**: 256×256 matrix ≈ 512KB (trivial)
- **Efficient δH**: 90M params ≈ 360MB (trivial)

### Computation Estimates  
- **Per-sequence gradients**: 256 separate forward/backward passes (expensive but necessary)
- **Fisher kernel**: O(256² × 90M) = O(6×10¹²) operations on CPU (feasible)
- **Efficient δH**: O(90M) = O(10⁸) operations (very fast)

### Performance Impact
- **Additional computation**: ~2x training time (256 individual gradient computations)
- **Memory constraints**: Respected (never exceed microbatch size on GPU)
- **Feasibility**: LoRA makes this computationally practical

## Key Design Decisions

### Timing: Before Optimizer Step
- Compute entropy probe using **current model state** to predict entropy change
- Validate predictions against actual entropy change after optimizer step
- Enables proper scientific validation of the theoretical framework

### Memory Management: Microbatch Gradient Accumulation  
- Process rollout buffer in same microbatches as training (respect GPU memory limits)
- Store per-sequence gradients W_α,t on CPU (90GB total)
- Compute 256×256 Fisher kernel on CPU after collecting all gradients

### Adam Preconditioning: Configurable Strategy
- **Default**: Use previous step's Adam factors (smooth EMA evolution)
- **Research**: Option to disable preconditioning (P_α = 1) for studying raw Fisher structure
- **Future**: Option to compute after optimizer step for "correct" factors

### Computational Cost: 2x Training Time
- 256 individual forward/backward passes for per-sequence gradients
- Acceptable cost for detailed entropy dynamics analysis
- Can be disabled for production runs

## Next Steps

1. **Immediate**: Implement Phase 1 (microbatch gradient accumulation)
2. **Short term**: Add Phase 2 (efficient δH computation)  
3. **Medium term**: Multi-GPU support (Phases 3-4)
4. **Long term**: Production two-tier system (Phase 5)

This plan balances theoretical correctness with computational efficiency, providing both routine monitoring capabilities and detailed analysis tools.