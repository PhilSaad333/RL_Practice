# Entropy Experiments Implementation Plan

## Overview

**Two-Tier Approach for Entropy Analysis**:
1. **Regular Training**: Simple δH estimation during normal RL training (minimal overhead)
2. **Detailed Analysis**: Specialized entropy experiments with double-sampling and full Fisher kernels (checkpoint-based)

## Key Scientific Insight

### The Double-Sampling Problem
In our theoretical derivation:
```
δH = -η * E_t∼π[(S(t) - S̄) * Σ_i K₁(t,t_i') * A(t_i') / L_max(p_i)]
```

**Current approach**: E_t[...] estimated using same training samples t_i'
**Better approach**: E_t[...] estimated using **separate samples t**, then compute cross-terms K₁(t,t_i')

**⚠️ CRITICAL: Different sampling criteria for the two buffers:**

**Training Buffer (t_i')**: 
- Current filtered behavior: reject sequences with zero advantages
- Apply scheduler difficulty weighting
- Only samples that provide gradient signal

**Evaluation Buffer (t)**:
- **UNFILTERED** random sampling from dataset
- NO rejection based on correctness 
- NO difficulty-based sampling bias
- Represents true data distribution

**Scientific value**: Study how filtered training batch t_i' affects logprobs of **any sequence t**, including:
- Same-prompt vs different-prompt interactions
- Training batch vs held-out sequences
- Fisher kernel structure across diverse sequence types
- Effect of training on the broader data distribution (not just "learnable" sequences)

## Implementation Architecture

### Tier 1: Regular Training (entropy_probe.py)
**Goal**: Lightweight δH estimation during normal training
**Location**: Existing `rl_training/algs/entropy_probe.py`
**Integration**: Called from `dr_grpo.py` during training

**Features**:
- Simple δH = Σ_α (∂_α H) × (δθ_α) × P_α computation
- No Fisher kernel storage (memory efficient)
- Every step monitoring capability
- Minimal performance impact (~5% overhead)

### Tier 2: Detailed Analysis (entropy_experiments/)
**Goal**: Comprehensive Fisher kernel analysis with double-sampling
**Location**: New `entropy_experiments/` folder
**Integration**: Standalone analysis of saved checkpoints

**Features**:
- Load checkpoint + optimizer state from regular training
- Double-sampling: separate batches for E_t[...] and training updates
- Full Fisher kernel computation and storage
- Cross-sequence analysis (training vs evaluation samples)
- Detailed entropy dynamics across training trajectory

## File Structure

```
entropy_experiments/
├── entropy_runner.py          # Main specialized training runner
├── fisher_analyzer.py         # Fisher kernel computation and analysis
├── double_sampling.py         # Separate sampling for E_t and training batches
├── checkpoint_loader.py       # Load training state from regular checkpoints
├── configs/
│   ├── fisher_analysis.yaml   # Configuration for detailed Fisher analysis
│   └── entropy_sweep.yaml     # Multi-checkpoint analysis config
├── results/
│   └── analysis_YYYY-MM-DD/   # Timestamped analysis results
└── notebooks/
    └── fisher_visualization.py # Analysis and plotting tools
```

## Tier 1: Simple δH Probe (During Training)

### Implementation
```python
class SimpleEntropyProbe:
    """Lightweight entropy change estimation during regular training."""
    
    def compute_delta_h_fast(self, rollouts, advantages, param_update):
        """Compute δH without storing Fisher kernel."""
        
        # Compute aggregated entropy gradient: ∂_α H = -E[(S-S̄) ∂_α S]
        entropy_grads = self._compute_entropy_gradients(rollouts)
        
        # Apply Adam conditioning to parameter update
        conditioned_update = self._apply_conditioning(param_update)
        
        # δH = Σ_α (∂_α H) × (δθ_α) × P_α
        delta_h = torch.sum(entropy_grads * conditioned_update)
        
        return delta_h
```

### Integration with Training
```python
# In dr_grpo.py
if self.entropy_probe.enabled:
    # Simple δH estimation (fast)
    predicted_delta_h = self.entropy_probe.compute_delta_h_fast(
        rollouts, advantages, param_update
    )
    self._last_entropy_metrics = {"delta_h_predicted": predicted_delta_h}
```

## Tier 2: Detailed Fisher Analysis (Offline)

### Double-Sampling Strategy
```python
# entropy_experiments/entropy_runner.py

def run_fisher_analysis(checkpoint_path, config):
    """Run detailed Fisher kernel analysis on saved checkpoint."""
    
    # Load checkpoint and training state
    model, optimizer, scheduler = load_checkpoint(checkpoint_path)
    
    # Sample evaluation sequences (for E_t[...] computation)
    eval_sequences = sample_evaluation_batch(config.eval_batch_size)  # e.g., 128 sequences
    
    # Sample training sequences (for gradient updates t_i')
    train_sequences = sample_training_batch(config.train_batch_size)  # e.g., 256 sequences
    
    # Compute cross-Fisher kernel K₁(t_eval, t_train)
    fisher_kernel = compute_cross_fisher_kernel(
        eval_sequences, train_sequences, model, trainable_params
    )
    
    # Analyze Fisher kernel structure
    analysis = analyze_fisher_structure(fisher_kernel, eval_sequences, train_sequences)
    
    return fisher_kernel, analysis
```

### Fisher Kernel Computation
```python
def compute_cross_fisher_kernel(eval_seqs, train_seqs, model, params):
    """Compute Fisher kernel between evaluation and training sequences."""
    
    # Compute gradients for evaluation sequences
    eval_gradients = []  # W_α,t for t in eval_seqs
    for seq in eval_seqs:
        grad = compute_per_sequence_gradient(seq, model, params)
        eval_gradients.append(grad.cpu())
    
    # Compute gradients for training sequences  
    train_gradients = []  # W_α,t' for t' in train_seqs
    for seq in train_seqs:
        grad = compute_per_sequence_gradient(seq, model, params)
        train_gradients.append(grad.cpu())
    
    # Fisher kernel: K₁(t,t') = Σ_α W_α,t W_α,t' P_α
    eval_matrix = torch.stack(eval_gradients)    # (n_eval, n_params)
    train_matrix = torch.stack(train_gradients)  # (n_train, n_params)
    
    # Cross-kernel computation
    fisher_kernel = torch.mm(eval_matrix, train_matrix.t())  # (n_eval, n_train)
    
    return fisher_kernel
```

## Configuration

### Regular Training Config
```yaml
# rl_training/cfg/testconfig.yaml
entropy_probe:
  enabled: true
  mode: "simple"          # Only compute fast δH
  log_every: 1            # Log every step
  preconditioning: "previous_step"
```

### Detailed Analysis Config
```yaml
# entropy_experiments/configs/fisher_analysis.yaml
entropy_analysis:
  eval_batch_size: 128     # Sequences for E_t[...] computation
  train_batch_size: 256    # Sequences for training updates
  
  sampling:
    eval_prompts: "random"  # or "fixed_set" for reproducibility
    train_prompts: "random"
    same_prompts: false     # Whether eval and train can overlap
  
  fisher_kernel:
    compute_full: true
    save_gradients: true    # For detailed analysis
    preconditioning: "checkpoint"  # Use Adam state from checkpoint
  
  analysis:
    prompt_structure: true   # Same-prompt vs different-prompt analysis
    sequence_clustering: true
    entropy_prediction: true
```

## Scientific Analysis Capabilities

### 1. **Cross-Sequence Fisher Structure**
- **Same-prompt pairs**: K₁(t₁|p, t₂|p) for sequences from same prompt
- **Different-prompt pairs**: K₁(t₁|p₁, t₂|p₂) for sequences from different prompts  
- **Training vs evaluation**: K₁(t_eval, t_train) cross-terms

### 2. **Entropy Dynamics Across Training**
```python
# Run analysis on multiple checkpoints
checkpoints = ["step_100", "step_200", "step_500", "step_1000"]
fisher_evolution = []

for ckpt in checkpoints:
    fisher, analysis = run_fisher_analysis(ckpt, config)
    fisher_evolution.append({
        "step": ckpt,
        "fisher_kernel": fisher,
        "analysis": analysis
    })

# Study how Fisher kernel structure evolves during training
plot_fisher_evolution(fisher_evolution)
```

### 3. **Validation of Theoretical Predictions**
- Compare predicted vs actual entropy changes
- Study first-order vs higher-order effects  
- Validate Dr-GRPO weighting schemes

## Resource Requirements

### Tier 1 (Regular Training)
- **Memory**: O(n_params) ≈ 360MB for LoRA
- **Computation**: O(n_params) ≈ 5% training overhead
- **Frequency**: Every step (if desired)

### Tier 2 (Detailed Analysis)  
- **Memory**: O(n_eval × n_train × n_params) ≈ 128×256×90M ≈ 2.9TB
- **Storage**: Store gradients on CPU/disk, compute Fisher kernel in chunks
- **Computation**: ~10x training time for comprehensive analysis
- **Frequency**: Selected checkpoints only

## Implementation Timeline

### Phase 1: Simple δH Probe
1. Implement lightweight `SimpleEntropyProbe` class
2. Integrate with existing `dr_grpo.py`
3. Test with regular training runs
4. Validate δH predictions

### Phase 2: Detailed Analysis Framework
1. Create `entropy_experiments/` folder structure
2. Implement checkpoint loading and state restoration
3. Build double-sampling framework
4. Implement Fisher kernel computation

### Phase 3: Scientific Analysis
1. Cross-sequence Fisher kernel analysis
2. Entropy dynamics across training trajectory
3. Validation of theoretical predictions
4. Visualization and reporting tools

This approach cleanly separates routine monitoring from detailed scientific analysis, enabling both practical training insights and rigorous theoretical validation.