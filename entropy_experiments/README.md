# 🧪 Entropy Experiments - Offline Entropy Probe Analysis

**Advanced entropy change analysis for reinforcement learning training with comprehensive debugging capabilities.**

## 🎯 Overview

This directory contains the offline entropy probe system for analyzing entropy changes in RL-trained language models. The system can predict first-order entropy changes (δH₁) using gradient-based estimators and measure ground-truth entropy changes via importance sampling.

## 🚀 Quick Start

```bash
# Basic entropy probe run with ground truth comparison
python -c "
from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe
probe = OfflineEntropyProbe.from_config_file('entropy_experiments/configs/detailed_logging_example.yaml')
results = probe.run_mixed_probe()
print(f'δH₁ prediction: {results[\"deltaH1\"]:.8f}')
print(f'Ground truth ΔH: {results[\"deltaH_true\"]:.8f}')
"

# Run from command line with custom checkpoint
python entropy_experiments/run_probe_with_comprehensive_logging.py \
  --config entropy_experiments/configs/detailed_logging_example.yaml \
  --checkpoint /path/to/checkpoint
```

## 📁 Directory Structure

```
entropy_experiments/
├── offline_entropy_probe.py         # 🎯 Main orchestrator class
├── detailed_logger.py               # 📊 Comprehensive logging system  
├── delta_entropy_is.py               # 🎲 Importance sampling for ground truth
├── probe_components.py               # 🔧 Core gradient computation components
├── adam_preconditioner.py            # ⚡ Adam preconditioning utilities
├── model_loader.py                   # 📦 LoRA/QLoRA model loading
├── distributed_helpers.py            # 🌐 Multi-GPU utilities
├── configs/                          # ⚙️ Configuration templates
│   ├── detailed_logging_example.yaml # 📝 Full configuration with logging
│   ├── test_deltaH1.yaml             # 🧪 Quick testing configuration
│   └── [other config files]
├── logs/                             # 📋 Detailed probe logs (auto-created)
│   └── YYYY-MM-DD/                   # Organized by date
│       ├── entropy_probe_HH-MM-SS_checkpoint.json.gz
│       ├── entropy_probe_HH-MM-SS_checkpoint_summary.json
│       └── entropy_probe_HH-MM-SS_checkpoint_config.yaml
├── results/                          # 📈 Analysis results
├── docs/                             # 📚 Documentation
├── logging_plan.txt                  # 📋 Comprehensive logging architecture plan
└── README.md                         # This file
```

## 🔬 Core Components

### OfflineEntropyProbe (Main Orchestrator)

The central class that coordinates all entropy analysis:

```python
from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe

# Load from config
probe = OfflineEntropyProbe.from_config_file('configs/my_config.yaml')

# Run analysis
results = probe.run_mixed_probe(checkpoint_path='/path/to/checkpoint')
```

**Key Features:**
- **Phase 0**: E/U batch sampling via SequenceProcessor
- **Phase 1-3**: δH₁ gradient-based prediction (optional)
- **Phase 5**: Ground-truth entropy change via importance sampling (optional)
- **Comprehensive logging**: Detailed debugging information
- **Multi-GPU support**: Distributed computation capabilities

### DeltaEntropyIS (Ground Truth Measurement)

Importance sampling-based ground truth entropy measurement:

- **RL-aligned updates**: Uses GRPO-style optimization on U batch
- **SNIS estimator**: Self-normalized importance sampling with RB payload
- **Model snapshots**: Preserves original model state
- **Rich diagnostics**: ESS, weight statistics, convergence metrics

### ProbeComponents (Gradient Estimation)

Core gradient computation for δH₁ estimation:

- **X gradients**: Entropy gradients from E batch (∇H_w)
- **Y gradients**: Preconditioned policy gradients from U batch  
- **Adam preconditioning**: Uses actual optimizer state
- **Estimator**: δH₁ = learning_rate × (X̄ · Ȳ)

## 📊 Detailed Logging System

### Logging Levels

The probe supports four levels of logging detail:

#### 1. **Minimal** (~1KB)
Core metrics only - perfect for automated analysis:
```yaml
detailed_logging:
  enabled: true
  level: "minimal"
```
**Contents**: bars_dot, deltaH1, deltaH_true, timing, batch sizes

#### 2. **Standard** (~5-10KB) 
Core metrics + batch statistics + diagnostics:
```yaml  
detailed_logging:
  enabled: true
  level: "standard"
  log_sequences: true
```
**Contents**: + Generation length stats, reward/advantage stats, ESS, weight statistics

#### 3. **Detailed** (~50-500KB)
Standard + individual sequence data with text:
```yaml
detailed_logging:
  enabled: true
  level: "detailed"
  log_sequences: true
```
**Contents**: + Individual prompts/responses, logprobs, RB entropies, importance weights

#### 4. **Debug** (~1-10MB)
Everything + token-level data + raw tensors:
```yaml
detailed_logging:
  enabled: true
  level: "debug"
  log_sequences: true
  log_tokens: true
  log_raw_tensors: true
```
**Contents**: + Token-by-token logprobs, raw tensor dumps, complete intermediate state

### Log File Structure

Each run generates three files:
- **Main log**: Complete structured JSON with all requested data
- **Summary**: Quick overview with core metrics only  
- **Config**: Exact configuration used for the run

```
logs/2025-01-15/
├── entropy_probe_14-30-25_step_60.json.gz     # Main detailed log
├── entropy_probe_14-30-25_step_60_summary.json # Quick summary  
└── entropy_probe_14-30-25_step_60_config.yaml  # Exact config
```

## ⚙️ Configuration

### Basic Configuration Template

```yaml
# Model checkpoint
checkpoint:
  checkpoint_path: "/path/to/checkpoint/model"
  backbone: "qwen2_5_15"  
  dtype: "bfloat16"

# Batch settings
batch_config:
  dataset_name: "gsm8k_r1_template"
  split: "test"
  B_E: 64    # Evaluation batch size  
  B_U: 16    # Update batch size
  G: 8       # Responses per prompt (U batch)

# Enable both prediction and ground truth
probe_rework:
  compute_delta_h1: true         # Phase 1-3: δH₁ prediction
  
importance:
  enabled: true                  # Phase 5: Ground truth ΔH
  is_mode: "snis" 
  report_per_token: true

# Detailed logging
detailed_logging:
  enabled: true
  level: "standard"              # Choose your detail level
  log_sequences: true
  output_directory: "entropy_experiments/logs"
  compress: true
```

### Available Configurations

- **`detailed_logging_example.yaml`**: Complete example with all options explained
- **`test_deltaH1.yaml`**: Quick testing with small batches
- **Custom configs**: Create your own based on the templates

## 🔍 Analysis Workflow

### 1. Basic Entropy Change Analysis
```python
# Load probe  
probe = OfflineEntropyProbe.from_config_file('configs/detailed_logging_example.yaml')

# Run analysis
results = probe.run_mixed_probe('/path/to/checkpoint')

# Extract key metrics
delta_h1_pred = results['deltaH1']        # First-order prediction  
delta_h_true = results['deltaH_true']     # Ground truth measurement
bars_dot = results['bars_dot']            # Gradient dot product
learning_rate = results['learning_rate']  # Effective learning rate

print(f"δH₁ = {delta_h1_pred:.8f}")
print(f"ΔH_true = {delta_h_true:.8f}") 
print(f"Error = {abs(delta_h1_pred - delta_h_true):.8f}")
```

### 2. Debugging with Detailed Logs
```python
# Enable debug logging
config['detailed_logging'] = {
    'enabled': True,
    'level': 'debug',
    'log_sequences': True,
    'log_tokens': True, 
    'log_raw_tensors': True
}

probe = OfflineEntropyProbe(config)
results = probe.run_mixed_probe('/path/to/checkpoint')

# Log file path will be printed - examine for detailed debugging info
```

### 3. Ground Truth Only (Skip δH₁)
```python
# For direct entropy measurement without gradient estimation
config['probe_rework']['compute_delta_h1'] = False  # Skip Phase 1-3
config['importance']['enabled'] = True              # Enable Phase 5 only

probe = OfflineEntropyProbe(config)  
results = probe.run_mixed_probe('/path/to/checkpoint')

# Results will only contain ground truth measurements
delta_h_true = results['deltaH_true']
```

## 📈 Understanding Results

### Core Metrics

- **`bars_dot`**: Gradient dot product X̄ · Ȳ (key quantity for prediction)
- **`deltaH1`**: First-order entropy change prediction (bars_dot × learning_rate)  
- **`deltaH_true`**: Ground truth entropy change via importance sampling
- **`learning_rate`**: Effective learning rate used in prediction

### Ground Truth Diagnostics

- **`H_orig`**: Original entropy before update (RB-based)
- **`H_upd`**: Updated entropy after RL update (RB-based)  
- **`ESS`**: Effective Sample Size (importance sampling quality metric)
- **`w_max/w_min`**: Importance weight range (stability indicator)

### Batch Statistics

- **`B_E/B_U`**: Global batch sizes used
- **`avg_generation_length`**: Average response length
- **`avg_reward`**: Average reward for U batch responses

## 🐛 Troubleshooting

### Common Issues

**"Checkpoint not found"**
```bash
# Ensure checkpoint contains LoRA adapter
ls /path/to/checkpoint/model/
# Should show: adapter_config.json, adapter_model.safetensors

# Also ensure optimizer.pt exists
ls /path/to/checkpoint/optimizer.pt
```

**Memory errors with large batches**
```yaml
batch_config:
  B_E: 16    # Reduce from default 64
  B_U: 8     # Reduce from default 16
  
memory_config:
  microbatch_size: 1  # Reduce from default 2
```

**Import errors**
```bash
# Run from project root  
cd /path/to/RL_Practice
python -c "from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe"
```

**Detailed logging too large**
```yaml
detailed_logging:
  level: "standard"    # Reduce from "detailed" or "debug"
  compress: true       # Enable gzip compression
  log_sequences: false # Disable sequence text logging
```

### Debug Mode

For comprehensive debugging, use the debug configuration:

```yaml
detailed_logging:
  enabled: true
  level: "debug"
  log_sequences: true
  log_tokens: true
  log_raw_tensors: true
```

This captures complete intermediate state for detailed analysis.

## 🔬 Research Applications

### δH₁ Estimator Validation
Compare first-order predictions against ground truth:
```python
results = probe.run_mixed_probe(checkpoint_path)
prediction_error = abs(results['deltaH1'] - results['deltaH_true'])
relative_error = prediction_error / abs(results['deltaH_true'])
```

### Batch Size Scaling Studies  
Test how δH₁ prediction quality changes with batch size:
```python
for B_E in [16, 32, 64, 128]:
    config['batch_config']['B_E'] = B_E
    probe = OfflineEntropyProbe(config)
    results = probe.run_mixed_probe(checkpoint_path)
    # Analyze prediction accuracy vs batch size
```

### Importance Sampling Quality Analysis
Monitor ESS and weight statistics:
```python
results = probe.run_mixed_probe(checkpoint_path)
ess = results['diagnostics']['ESS']
w_max = results['diagnostics']['w_max'] 
# ESS should be high, weight ratios should be moderate
```

## 🚨 Important Notes

### Model Requirements
- **LoRA/QLoRA models**: Supports both via model_loader.py  
- **Adam optimizer**: Requires saved optimizer state for preconditioning
- **Compatible models**: Tested with Qwen2.5-1.5B, should work with other causal LMs

### Performance Tips
1. **Start small**: Use B_E=16, B_U=8 for initial testing
2. **Use compression**: Enable gzip for detailed logs  
3. **Batch efficiently**: Balance memory usage vs statistical power
4. **Monitor ESS**: Low ESS indicates poor importance sampling

### Limitations
- **Single GPU optimized**: Multi-GPU support available but less tested
- **Memory intensive**: Debug logging can be very large
- **RL-specific**: Designed for GRPO-style RL training

## 📚 Additional Documentation

- **`logging_plan.txt`**: Comprehensive logging system architecture
- **Code comments**: Extensive inline documentation
- **Config examples**: Multiple templates in `configs/` directory

## 🔄 Migration from Old Systems

If you have old entropy probe scripts, the new system provides:
- **Unified interface**: Single OfflineEntropyProbe class
- **Configurable analysis**: Skip δH₁ or importance sampling as needed
- **Rich diagnostics**: Detailed logging at multiple levels
- **Better reliability**: Improved error handling and state management

---

**This system provides comprehensive entropy analysis with extensive debugging capabilities for RL training research.** 🎉