# Entropy Experiments Utilities

This directory contains utility modules supporting the entropy change measurement system.

## Core Utilities

### model_loader.py
PEFT model and optimizer loading with precision control:
- **`load_peft_for_probe()`**: Load LoRA/QLoRA models for entropy experiments
- **`load_adam_optimizer_from_path()`**: Load optimizer state with parameter remapping
- **Model registry support**: Resolves shortcut names like "qwen2_5_15"
- **Precision enforcement**: Forces fp32 runtime for numerical stability
- **Left-padding**: Configures tokenizer correctly for decoder-only models

### sequence_processor.py  
Batch sampling and generation pipeline:
- **`SequenceProcessor`**: Main class for generating batched sequences
- **`BatchedSequences`**: Data structure for sequences with metadata
- **Generation methods**: With/without gradients, logprob computation
- **Precision control**: Configurable autocast and dtype settings
- **Dataset integration**: Works with gsm8k_r1_template and other datasets

### param_overrides.py
Functional parameter override system for importance sampling:
- **`build_functional_params_named()`**: Create parameter dictionaries for functional_call
- **Update vector integration**: Apply η*v updates to specific parameters
- **Precision control**: Force specific dtypes during override computation
- **LoRA-safe validation**: Only updates trainable parameters by default

### precision_utils.py
Numerical precision utilities:
- **`apply_global_precision()`**: Set global TF32 and matmul precision
- **`str_to_dtype()`**: Convert string names to torch dtypes
- **`force_grads_fp32()`**: Ensure FP32 gradient storage
- **Context managers**: Autocast control utilities

### param_registry.py
Parameter enumeration and manipulation:
- **`get_trainable_named()`**: Enumerate trainable parameters
- **`get_optimizer_named_params()`**: Get parameters updated by optimizer
- **`to_cpu_fp32_named()`**: Convert parameter dictionaries to CPU FP32
- **`dot_named()`**: Compute dot products between parameter dictionaries

## Support Utilities

### detailed_logger.py
Comprehensive experiment logging system:
- Multi-level logging (minimal, standard, detailed, debug)
- Structured JSON output with compression
- Automatic file organization by date
- Configurable data retention and rotation

### distributed_helpers.py  
Multi-GPU computation support:
- Distributed batch processing utilities
- Gradient synchronization helpers
- Memory-efficient distributed operations

## Usage Pattern

Most utilities are used internally by the main entropy measurement classes:

```python
# Typical usage is through the main EntropyMeasurements class
from entropy_experiments.entropy_experiment_runner import EntropyMeasurements

# But individual utilities can be used directly:
from entropy_experiments.utils.model_loader import load_peft_for_probe

model, tokenizer = load_peft_for_probe(
    base_id="qwen2_5_15",
    adapter_path="/path/to/checkpoint/model", 
    force_fp32_runtime=True
)
```

## Design Principles

- **Numerical stability**: Emphasis on FP32/FP64 precision where needed
- **Memory efficiency**: CPU offloading and dtype control
- **LoRA compatibility**: Safe handling of frozen vs trainable parameters
- **Configuration-driven**: Behavior controlled through config files
- **Error resilience**: Graceful handling of edge cases and failures