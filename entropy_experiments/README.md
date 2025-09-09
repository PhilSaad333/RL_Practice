# Entropy Change Experiments

This package implements entropy change measurement for reinforcement learning models, providing both ground-truth and approximate estimators for policy entropy changes after parameter updates.

## Overview

### What We Measure

The core quantity of interest is **ΔH(η)** - the change in policy entropy after taking a gradient step of size η:

```
ΔH(η) = H(π_θ+ηv) - H(π_θ)
```

Where:
- π_θ is the current policy (language model)
- v is the normalized update direction (Δθ/lr from RL gradients)
- η is the step size multiplier
- H(π) is the policy entropy

### Theoretical Foundation

The implementation is based on the theoretical framework described in `docs/RL_studies.pdf`, which establishes:

1. **First-order approximation**: δH₁ ≈ η∑ₐ ∂ₐH × ∂ₐJ (learning rate × entropy gradient × policy gradient)
2. **Fisher kernel relationships**: K₁(t,t') = ∑ₐ ∂ₐS(t)∂ₐS(t') relating sequence gradients  
3. **Variance reduction techniques**: Rao-Blackwellization for entropy estimation
4. **Importance sampling**: For ground-truth entropy change measurement

## Quick Start

```bash
# Basic entropy experiments
python run_entropy_experiments.py

# Using the main class directly
python -c "
from entropy_experiments.entropy_experiment_runner import EntropyMeasurements
import yaml
config = yaml.safe_load(open('entropy_experiments/configs/config_template.yaml'))
probe = EntropyMeasurements(config)
results = probe.run_experiments()
print('Results:', results)
"
```

## Directory Structure

```
entropy_experiments/
├── entropy_experiment_runner.py   # 🎯 Main orchestrator class (EntropyMeasurements)
├── update_vector.py               # 🔧 Parameter update computation using AdamW math
├── delta_entropy_true.py          # 🎲 Ground truth via importance sampling (SNIS)
├── delta_entropy_approx.py        # ⚡ First-order approximation (incomplete)
├── utils/                         # 🛠️ Utility modules (see utils/README.md)
│   ├── model_loader.py            # 📦 LoRA/QLoRA model loading with precision control
│   ├── sequence_processor.py      # 📝 Batch sampling and generation
│   ├── param_overrides.py         # 🔄 Functional parameter override system
│   ├── precision_utils.py         # 🎯 Numerical precision utilities
│   └── [other utility files]
├── configs/                       # ⚙️ Configuration templates
│   └── config_template.yaml       # 📝 Complete configuration template
├── results/                       # 📈 Experimental results and analysis
│   ├── entropy_variance/          # Variance estimation studies
│   ├── estimation_experiments/    # Convergence analysis
│   └── testing_linear_regime/     # Linear regime validation
└── README.md                      # This file
```

## Architecture

### Two-Batch Estimator Design

The system uses separate batches for different purposes:

- **E-batch**: "Evaluation" batch for measuring entropy changes (G=1 responses per prompt)
- **U-batch**: "Update" batch for computing parameter updates (G>1 responses per prompt with advantages)

This separation allows clean measurement of entropy changes without contamination between update computation and evaluation.

### Two Estimation Methods

1. **Ground Truth (ΔH_true)**: Uses importance sampling with parameter overrides
   - Computes actual entropy at θ and θ+ηv
   - Self-normalized importance sampling (SNIS) for numerical stability
   - Implemented in `delta_entropy_true.py`

2. **First-Order Approximation (δH₁)**: Linear approximation using gradients
   - δH₁ ≈ η × ⟨∇H, v⟩ where ∇H is entropy gradient on E-batch
   - Much faster than ground truth but approximate
   - Implemented in `delta_entropy_approx.py` (currently incomplete)

## Core Components

### EntropyMeasurements (Main Orchestrator)

The central class that coordinates all entropy analysis:

```python
from entropy_experiments.entropy_experiment_runner import EntropyMeasurements
import yaml

# Load from config
config = yaml.safe_load(open('configs/config_template.yaml'))
probe = EntropyMeasurements(config)

# Run analysis
results = probe.run_experiments()
```

**Key Features:**
- **Batch sampling**: E/U batch sampling via SequenceProcessor
- **Update vector computation**: Normalized parameter updates using AdamW math
- **Eta sweeps**: Test multiple step sizes to validate linear approximations
- **Ground-truth measurement**: Importance sampling with functional parameter overrides
- **Precision control**: Configurable numerical precision for stability

### Key Algorithms

#### update_vector.py
Computes normalized update directions using AdamW mathematics:
- Builds v = Δθ/lr from U-batch gradients and optimizer state
- Supports mixed precision with configurable autocast
- Returns CPU float32 tensors for numerical stability

#### delta_entropy_true.py  
Ground-truth entropy measurement via importance sampling:
- Uses functional parameter overrides to evaluate model at θ+ηv
- SNIS reducer for stable importance weight computation
- Caches baseline (η=0) evaluations for efficiency

## Configuration System

The system uses YAML configuration files to control all aspects of the experiments. See `configs/config_template.yaml` for a complete template.

### Key Configuration Sections

#### Model and Checkpoint
```yaml
checkpoint:
  checkpoint_path: "/path/to/checkpoint/model"
  optimizer_path: "/path/to/checkpoint/optimizer.pt"  
  backbone: "qwen2_5_15"  # Model registry key
  device_map: "cuda"
```

#### Batch Configuration
```yaml
batch_config:
  dataset_name: "gsm8k_r1_template"
  E_split: "test"        # Evaluation batch split
  U_split: "train"       # Update batch split  
  B_E: 512              # Evaluation batch size
  B_U: 64               # Update batch size
  G: 8                  # Responses per prompt (U-batch only)
```

#### Estimator Settings
```yaml
estimator:
  use_simple_entropy_for_x: false    # Use RB entropy vs simple surprisal
  eta_sweep: true                    # Test multiple step sizes
  single_eta: 2e-5                   # Single step size (if eta_sweep=false)
  eta_list: [1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4, 6.4e-4, 1.28e-3]
```

#### Precision Control
```yaml
precision:
  allow_tf32: false                  # Global TF32 setting
  matmul_precision: high             # PyTorch matmul precision
  runtime_dtype: float32             # Model parameter dtype
  entropy_dtype: float64             # Entropy computation dtype
  
  update_vector:
    use_amp: false                   # Mixed precision for gradients
    amp_dtype: bfloat16             # AMP dtype if enabled
    grads_fp32: true                # Force FP32 gradient storage
```

#### Detailed Logging
```yaml
detailed_logging:
  enabled: true
  level: "standard"                  # minimal | standard | detailed | debug
  log_sequences: false               # Include sequence text
  log_tokens: false                  # Include token-level data  
  log_raw_tensors: false            # Include raw tensor dumps
  output_directory: "entropy_experiments/logs"
  compress: true                     # Gzip compression
  max_files: 50                     # Log rotation
```

## Usage Examples  

### Basic Entropy Analysis

```python
import yaml
from entropy_experiments.entropy_experiment_runner import EntropyMeasurements

# Load configuration
with open('entropy_experiments/configs/config_template.yaml') as f:
    config = yaml.safe_load(f)

# Update paths for your setup
config['checkpoint']['checkpoint_path'] = "/path/to/your/checkpoint/model"
config['checkpoint']['optimizer_path'] = "/path/to/your/checkpoint/optimizer.pt"

# Initialize and run experiments
probe = EntropyMeasurements(config)
results = probe.run_experiments()

# Examine results
for result in results["sweep"]:
    eta = result["eta"]
    delta_h_true = result["deltaH_true"]  
    delta_h_approx = result["deltaH_approx"]  # Currently 0.0 (incomplete)
    print(f"η={eta:.2e}: ΔH_true={delta_h_true:.6e}, ΔH₁={delta_h_approx:.6e}")

# Check timing and batch sizes
print(f"Batch sizes: E={results['B_E']}, U={results['B_U']}")
print(f"Total time: {results['timing']['total']:.2f}s")
```

### Single Step Size Analysis

```python
# Disable eta sweep for faster testing
config['estimator']['eta_sweep'] = False  
config['estimator']['single_eta'] = 1e-4   # Test specific step size

probe = EntropyMeasurements(config)
results = probe.run_experiments()

# Single result
eta = results["sweep"][0]["eta"]
delta_h_true = results["sweep"][0]["deltaH_true"]
print(f"ΔH_true({eta:.1e}) = {delta_h_true:.6e}")
```

### Precision Comparison

```python
# Test different entropy precisions
configs = []

# FP32 entropy computation
config_fp32 = config.copy()
config_fp32['precision']['entropy_dtype'] = 'float32'

# FP64 entropy computation  
config_fp64 = config.copy()
config_fp64['precision']['entropy_dtype'] = 'float64'

for name, cfg in [("FP32", config_fp32), ("FP64", config_fp64)]:
    probe = EntropyMeasurements(cfg)
    results = probe.run_experiments()
    delta_h = results["sweep"][0]["deltaH_true"]
    print(f"{name}: ΔH_true = {delta_h:.8e}")
```

## Model Requirements and Notes

- **LoRA/PEFT models**: System designed for LoRA adapters over frozen base models
- **Adam optimizers**: Update vector computation assumes AdamW/Adam with momentum  
- **Decoder-only LMs**: Tested with models like Qwen2.5-1.5B
- **Numerical precision**: Emphasizes stability with configurable FP32/FP64 options

## Research Context

This package supports research into:
- Entropy changes during RL fine-tuning
- Variance estimation for entropy measurements  
- Linear regime analysis for gradient-based updates
- Importance sampling methods for policy evaluation

See `results/` directory for experimental studies and `docs/RL_studies.pdf` for theoretical background.