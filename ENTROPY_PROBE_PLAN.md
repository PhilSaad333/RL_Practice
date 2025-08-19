# Entropy Probe Implementation Plan

## Overview
Implement first-order entropy change analysis: `δH = -η * (1/(B*G)) * Σ (S(t) - S̄) * K_1(t,t') * A(t')`

Where `K_1(t,t') = Σ_α ∂_α log π(t) ∂_α log π(t') P_α` (Fisher kernel with Adam conditioning)

## Key Technical Challenge: Per-Sequence Gradients

### Problem
- Current implementation: `loss.backward()` gives batch-level gradients `∂L_total/∂θ`
- Need: Per-sequence gradients `∂L(b,g)/∂θ` for each sequence (b,g)
- This is fundamental to computing `K_1(t,t') = Σ_α ∂_α log π(t) ∂_α log π(t') P_α`

### Solution Approaches

#### Option 1: Separate Forward/Backward Per Sequence (Expensive but Correct)
```python
def compute_per_sequence_gradients(self, rollouts, advantages):
    B, G = advantages.shape
    sequence_gradients = []
    
    for b in range(B):
        for g in range(G):
            # Isolate single sequence
            single_seq_rollout = extract_sequence(rollouts, b, g)
            single_advantage = advantages[b:b+1, g:g+1]
            
            # Forward pass for this sequence only
            self.opt.zero_grad()
            loss = self._compute_loss_single_sequence(single_seq_rollout, single_advantage)
            loss.backward()
            
            # Collect gradients
            grad_vector = self._collect_gradients()
            sequence_gradients.append(grad_vector)
            
    return sequence_gradients
```

**Pros**: Mathematically correct, captures exact per-sequence gradients
**Cons**: Very expensive (B*G forward/backward passes per step)

#### Option 2: Gradient Hooks (Complex but Efficient)
```python
def setup_gradient_hooks(self):
    """Set up hooks to capture per-sequence gradient contributions."""
    self.sequence_grad_contributions = {}
    
    def create_hook(param_name):
        def hook(grad):
            # Decompose batch gradient into per-sequence contributions
            # This requires careful analysis of how gradients aggregate
            per_seq_grads = self._decompose_batch_gradient(grad, param_name)
            self.sequence_grad_contributions[param_name] = per_seq_grads
        return hook
    
    for name, param in self.policy.named_parameters():
        if param.requires_grad:
            param.register_hook(create_hook(name))
```

**Pros**: Efficient (single forward/backward pass)
**Cons**: Complex implementation, requires deep understanding of gradient aggregation

#### Option 3: Approximation Using Microbatch Decomposition
```python
def approximate_per_sequence_gradients(self, rollouts, advantages):
    """
    Approximate per-sequence gradients by computing gradients on smaller microbatches
    and using the linearity of gradients to estimate individual contributions.
    """
    B, G = advantages.shape
    
    # Compute gradients for pairs/triplets of sequences
    # Use linearity: grad(A+B) = grad(A) + grad(B)
    # Solve system of equations to recover individual gradients
    
    # This is a compromise between accuracy and efficiency
```

**Pros**: More efficient than Option 1, more tractable than Option 2
**Cons**: Approximate, requires solving linear systems

### Recommended Approach: Start with Option 1 (Testing), Move to Option 2 (Production)

1. **Phase 1**: Implement Option 1 for proof-of-concept and validation
   - Test on small batches (B=2, G=4)
   - Verify Fisher kernel computation and δH estimation
   - Study prompt structure (same vs different prompt correlations)

2. **Phase 2**: Optimize with Option 2 for production use
   - Implement gradient hooks for efficient per-sequence gradient capture
   - Maintain same API as Option 1

## Integration Plan

### 1. Dr-GRPO Integration Points
```python
# In dr_grpo.py _backward_and_step method:
def _backward_and_step(self, loss, sync_grads, rollouts=None):
    # ... existing code ...
    
    if sync_grads:
        # GNS probe (existing)
        if self.gns_probe.enabled:
            self.gns_probe.store_gradient(...)
        
        # NEW: Entropy probe
        if self.entropy_probe.enabled and rollouts is not None:
            # Compute advantages and log probs
            advantages = self._compute_advantage(rollouts.reward)
            log_probs = self._compute_sequence_log_probs(rollouts)
            
            # Store entropy probe data (this will compute per-sequence gradients)
            self.entropy_probe.store_step_data(
                rollouts=rollouts,
                advantages=advantages,
                log_probs=log_probs,
                trainable_params=self._trainable_params(),
                optimizer=self.opt,
                learning_rate=self._get_current_lr(),
                step_idx=self.actual_opt_step
            )
```

### 2. Configuration Addition
```yaml
# In config files:
entropy_probe:
  enabled: true                 # Enable the probe
  debug: true                   # Print debug information
  max_sequences: 1000           # Memory safety limit
  store_full_kernel: true       # Store complete K_1(t,t') matrix
  save_every: 10                # Save detailed data every N steps
  per_sequence_method: "separate"  # "separate" or "hooks" or "microbatch"
```

### 3. Memory Management
- Store gradients on CPU to save GPU memory
- Limit maximum sequences per step (default: 1000)
- Clear data after analysis to prevent memory leaks

### 4. Analysis Outputs
- **Metrics**: δH estimate, Fisher kernel statistics, prompt structure analysis
- **Detailed Data**: Sequence metadata, full K_1 matrix, per-sequence gradients
- **Visualization**: Same-prompt vs different-prompt correlation patterns

## Testing Protocol

### Phase 1: Small Scale Validation
1. Run 4-step training with `B=2, G=4` (8 sequences total)
2. Use separate per-sequence gradient computation
3. Verify Fisher kernel is positive semi-definite
4. Compare δH estimate with actual entropy change
5. Analyze same-prompt vs different-prompt K_1 values

### Phase 2: Structure Analysis
1. Run larger batches with `B=8, G=8` (64 sequences)
2. Study block structure of K_1 matrix
3. Compare diagonal vs off-diagonal elements
4. Analyze correlation patterns between prompt groups

### Phase 3: Multi-Device Considerations
1. Test on single device first (current approach)
2. Design cross-device gradient sharing for full δH computation
3. Implement distributed Fisher kernel computation

## Implementation Priority

1. ✅ **Created base EntropyProbe class** with full interface
2. 🔄 **Implement per-sequence gradient computation** (Option 1 first)
3. **Integrate into Dr-GRPO training loop**
4. **Add configuration and testing**
5. **Validate on small batches**
6. **Optimize for larger batches**
7. **Analyze prompt structure and δH estimation**

The key insight is that we need to solve the per-sequence gradient challenge first, then everything else follows naturally from the theoretical framework.