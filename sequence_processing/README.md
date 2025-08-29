# SequenceProcessor

**Unified sequence generation and logprob computation for the RL_Practice project**

`SequenceProcessor` consolidates patterns from `collect_rollouts.py` and `dr_grpo.py` into a single, reusable interface for sequence generation across the project. It eliminates scattered custom generation code and provides consistent, battle-tested generation logic.

## Key Features

- **Dataset Integration**: Sample prompts directly from datasets (e.g., `gsm8k_r1_template`) with optional seeds
- **Batched Generation**: Memory-efficient generation with configurable batch sizes
- **Teacher Forcing**: Compute logprobs with or without gradients
- **DDP Support**: Works with distributed models (DistributedDataParallel)
- **StopAfterAnswer**: Built-in logic processor for math problems
- **Memory Management**: Optimized for H100 80GB GPU (configurable batch sizes)

## Quick Start

```python
from sequence_processing import SequenceProcessor, GenerationConfig

# Initialize with model and tokenizer
config = GenerationConfig(
    temperature=0.8,
    max_new_tokens=200,
    gen_batch_size=64,   # Generation batch size
    tf_batch_size=128    # Teacher forcing batch size
)
processor = SequenceProcessor(model, tokenizer, config)

# Option 1: Use explicit prompts
sequences, logprobs = processor.generate_with_logprobs(
    prompts=["Solve: 2+2=", "What is 5*3?"],
    G=8  # Generate 8 responses per prompt
)

# Option 2: Sample from dataset
sequences, logprobs = processor.generate_with_logprobs(
    dataset_name="gsm8k_r1_template",
    split="train",
    num_prompts=64,
    G=8,
    seed=42,  # Reproducible sampling
    with_grad=False  # No gradients for inference
)
```

## Classes and Configuration

### GenerationConfig

Controls generation and batching parameters:

```python
@dataclass
class GenerationConfig:
    temperature: float = 1.0          # Sampling temperature
    top_p: float = 1.0               # Nucleus sampling parameter
    max_new_tokens: int = 200        # Maximum tokens to generate
    do_sample: bool = True           # Enable sampling vs greedy
    
    # Batch sizes (optimized for H100 80GB)
    gen_batch_size: int = 32         # Generation batch size
    tf_batch_size: int = 64          # Teacher forcing batch size
```

### BatchedSequences

Container for generation results:

```python
@dataclass
class BatchedSequences:
    sequences: torch.Tensor           # [B, G, total_len] - full sequences
    prompt_lens: List[int]            # [B] - length of each prompt
    gen_lens: List[List[int]]         # [B][G] - generation lengths
    attention_masks: torch.Tensor     # [B, G, total_len] - attention masks
    responses_text: List[List[str]]   # [B][G] - decoded response texts
```

### LogprobResults

Container for logprob computation results:

```python
@dataclass
class LogprobResults:
    logprobs: List[List[torch.Tensor]]     # [B][G] - per-token logprobs
    entropies: List[List[np.ndarray]]      # [B][G] - per-token entropies
    sequence_logprobs: List[List[float]]   # [B][G] - total sequence logprobs
```

## Core Methods

### generate_with_logprobs() - Main Interface

The primary method that combines generation and logprob computation:

```python
def generate_with_logprobs(
    self,
    prompts: Optional[List[str]] = None,        # Explicit prompts
    G: int = 8,                                 # Responses per prompt
    dataset_name: Optional[str] = None,         # Dataset to sample from
    split: str = "train",                       # Dataset split
    num_prompts: Optional[int] = None,          # Number of prompts to sample
    seed: Optional[int] = None,                 # Random seed
    with_grad: bool = False,                    # Compute gradients?
    gen_batch_size: Optional[int] = None,       # Override generation batch size
    tf_batch_size: Optional[int] = None         # Override teacher forcing batch size
) -> Tuple[BatchedSequences, LogprobResults]:
```

**Usage Patterns:**

```python
# Explicit prompts
sequences, logprobs = processor.generate_with_logprobs(
    prompts=["Question 1", "Question 2"],
    G=4
)

# Dataset sampling
sequences, logprobs = processor.generate_with_logprobs(
    dataset_name="gsm8k_r1_template",
    split="test", 
    num_prompts=32,
    G=8,
    seed=42
)

# With gradients (for training)
sequences, logprobs = processor.generate_with_logprobs(
    prompts=training_prompts,
    G=8,
    with_grad=True,
    tf_batch_size=64
)
```

### Individual Methods

For more control, use individual methods:

```python
# Just generation
sequences = processor.generate_batched(prompts, G=8)

# Just teacher forcing  
logprobs = processor.teacher_force_logprobs(sequences, with_grad=False)

# Dataset sampling
prompts = processor.sample_prompts("gsm8k_r1_template", "train", 64, seed=42)
```

## Performance Optimization

### Batch Size Guidelines

For **H100 80GB GPU**:
- `gen_batch_size`: 64-128 (memory permitting)
- `tf_batch_size`: 128-256 (teacher forcing can handle larger batches)

For **smaller GPUs**:
- Start with defaults (32/64) and adjust based on memory

### Memory Management

The processor automatically handles:
- **Tensor padding** for different sequence lengths
- **Batch splitting** to prevent OOM errors
- **DDP unwrapping** for distributed models
- **BFloat16 conversion** for mixed precision

```python
# Auto-detect optimal batch sizes (future feature)
config = GenerationConfig()  # Uses optimized defaults
```

## Integration Examples

### Replace Custom Generation Code

**Before** (scattered custom code):
```python
# Custom generation logic scattered across files
def my_custom_generation(model, tokenizer, prompts, G):
    # 50+ lines of generation, tokenization, logprob computation...
    pass
```

**After** (unified SequenceProcessor):
```python
processor = SequenceProcessor(model, tokenizer)
sequences, logprobs = processor.generate_with_logprobs(prompts, G)
```

### Entropy Variance Experiments

```python
# entropy_experiments/other_scripts/entropy_variance/entropy_variance.py
from sequence_processing import SequenceProcessor, GenerationConfig

def batched_generation(model, tokenizer, prompts, G, gen_cfg):
    sp_config = GenerationConfig(
        temperature=gen_cfg.temperature,
        max_new_tokens=gen_cfg.max_new_tokens,
        gen_batch_size=128,  # Optimized for H100
        tf_batch_size=256
    )
    processor = SequenceProcessor(model, tokenizer, sp_config)
    sequences, logprob_results = processor.generate_with_logprobs(
        prompts=prompts, G=G, with_grad=False
    )
    # Convert to expected format...
```

### Evaluation Scripts

```python
# evals/utils_io.py replacement
processor = SequenceProcessor(model, tokenizer)
sequences, logprobs = processor.generate_with_logprobs(
    dataset_name="gsm8k_r1_template",
    split="test",
    num_prompts=100,
    G=4,
    seed=42
)
```

## Error Handling

The processor handles common issues:

- **Missing pad tokens**: Automatically sets to EOS token
- **DDP models**: Unwraps for compatibility
- **BFloat16 tensors**: Converts for numpy operations
- **Empty generations**: Returns empty arrays gracefully
- **Memory overflow**: Splits batches automatically

## Dataset Support

Currently supports datasets from `rlp_datasets.DATASET_REGISTRY`:

- `gsm8k_r1_template` (primary math dataset)
- Any dataset following the `Example` dataclass pattern

```python
# Add custom datasets to rlp_datasets registry
from rlp_datasets import DATASET_REGISTRY
```

## Development Notes

### Proven Patterns Used

- **Generation**: Based on `collect_rollouts.py:159-177` (battle-tested)
- **Teacher Forcing**: Based on `collect_rollouts.py:184-208` and `dr_grpo.py:372-401`
- **StopAfterAnswer**: Extracted from working implementations
- **Batching**: Memory-optimized patterns from production training code

### Testing

Tested on:
- ✅ Lambda H100 80GB with LoRA checkpoints
- ✅ Qwen2.5-1.5B model with bf16 precision
- ✅ Large-scale experiments (8,192+ generations)
- ✅ Dataset sampling with reproducible seeds
- ✅ Both grad and no-grad teacher forcing

### Future Enhancements

- [ ] Automatic batch size detection based on available GPU memory
- [ ] Support for more dataset formats
- [ ] Streaming generation for very large experiments
- [ ] Integration with Weights & Biases logging
- [ ] Multi-GPU generation support

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce `gen_batch_size` or `tf_batch_size`
2. **Slow Generation**: Increase batch sizes if memory allows
3. **Missing Datasets**: Ensure `rlp_datasets` is importable
4. **DDP Issues**: Processor automatically unwraps models
5. **BFloat16 Errors**: Fixed in current version

### Performance Monitoring

```python
# Monitor GPU utilization during generation
# Should see 50-80% memory usage on H100 with optimized settings
```

---

**Created**: August 2025  
**Last Updated**: August 2025  
**Tested On**: Lambda H100 80GB, Qwen2.5-1.5B + LoRA  
**Status**: Production Ready ✅