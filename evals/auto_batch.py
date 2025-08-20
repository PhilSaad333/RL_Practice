# evals/auto_batch.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Auto-batch sizing for evaluation - dynamically find optimal batch sizes
that maximize GPU memory utilization without OOM errors.
"""

import torch
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Global cache for batch size results per model/GPU combination
BATCH_SIZE_CACHE: Dict[str, Dict[str, int]] = {}


def get_gpu_signature() -> str:
    """Create unique signature for current GPU setup."""
    if not torch.cuda.is_available():
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    return f"{gpu_name}_{gpu_memory//1024**3}GB"


def get_model_signature(model) -> str:
    """Create signature for model to cache batch sizes."""
    model_name = getattr(model, '_name_or_path', 'unknown')
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', 'unknown')
        return f"{model_name}_{hidden_size}"
    return model_name


def test_batch_size_generation(
    model, 
    tokenizer, 
    batch_size: int, 
    max_tokens: int, 
    num_sequences: int,
    test_prompt_length: int = 100
) -> bool:
    """
    Test if batch_size fits in memory for generation phase.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer
        batch_size: Batch size to test
        max_tokens: Max new tokens to generate
        num_sequences: Number of return sequences per prompt
        test_prompt_length: Length of test prompts in tokens
    
    Returns:
        True if batch size fits, False if OOM
    """
    try:
        # Create realistic test prompts
        test_text = "What is the solution to this math problem? " * (test_prompt_length // 10)
        test_prompts = [test_text] * batch_size
        
        with torch.no_grad():
            # Test generation phase
            enc = tokenizer(
                test_prompts, 
                padding=True, 
                truncation=True,
                max_length=test_prompt_length,
                return_tensors="pt"
            ).to(model.device)
            
            # Test actual generation (this is usually memory-intensive)
            _ = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                num_return_sequences=num_sequences,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
            
        torch.cuda.empty_cache()
        return True
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        logger.warning(f"Unexpected error testing batch size {batch_size}: {e}")
        torch.cuda.empty_cache()
        return False


def test_batch_size_teacher_forcing(
    model, 
    tokenizer, 
    tf_batch_size: int,
    sequence_length: int = 300
) -> bool:
    """
    Test if tf_batch_size fits in memory for teacher forcing phase.
    This is often the bottleneck due to full forward pass through all tokens.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer  
        tf_batch_size: Teacher forcing batch size to test
        sequence_length: Total sequence length (prompt + generation)
    
    Returns:
        True if batch size fits, False if OOM
    """
    try:
        with torch.no_grad():
            # Create test sequences of realistic length
            test_seq = torch.randint(
                0, min(tokenizer.vocab_size, 50000),  # Use subset of vocab for speed
                (tf_batch_size, sequence_length),
                device=model.device
            )
            
            # Test teacher forcing forward pass (memory bottleneck)
            _ = model(test_seq).logits
            
        torch.cuda.empty_cache()
        return True
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False
    except Exception as e:
        logger.warning(f"Unexpected error testing TF batch size {tf_batch_size}: {e}")
        torch.cuda.empty_cache()
        return False


def binary_search_batch_size(
    test_func,
    min_size: int = 1,
    max_size: int = 128,
    **test_kwargs
) -> int:
    """
    Binary search to find maximum batch size that fits in memory.
    
    Args:
        test_func: Function that tests if a batch size works
        min_size: Minimum batch size to try
        max_size: Maximum batch size to try
        **test_kwargs: Additional arguments to pass to test_func
    
    Returns:
        Maximum batch size that fits in memory
    """
    low, high = min_size, max_size
    best_size = min_size
    
    # Quick check if even min_size works
    if not test_func(min_size, **test_kwargs):
        logger.error(f"Even minimum batch size {min_size} doesn't fit in memory!")
        return 1
    
    # Quick check if max_size works (save time if we have lots of memory)
    if test_func(max_size, **test_kwargs):
        return max_size
    
    # Binary search between min and max
    while low <= high:
        mid = (low + high) // 2
        
        if test_func(mid, **test_kwargs):
            best_size = mid
            low = mid + 1  # Try larger
        else:
            high = mid - 1  # Too big, try smaller
    
    return best_size


def auto_detect_batch_sizes(
    model,
    tokenizer,
    max_tokens: int = 200,
    num_sequences: int = 8,
    prompt_length: int = 100,
    safety_factor: float = 0.85
) -> Tuple[int, int]:
    """
    Auto-detect optimal batch sizes for both generation and teacher forcing.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer
        max_tokens: Maximum new tokens to generate
        num_sequences: Number of return sequences per prompt
        prompt_length: Typical prompt length in tokens
        safety_factor: Reduce final batch size by this factor for safety
    
    Returns:
        Tuple of (rollout_batch_size, tf_micro_batch_size)
    """
    model_sig = get_model_signature(model)
    gpu_sig = get_gpu_signature()
    
    # Create cache key
    cache_key = f"{model_sig}_{gpu_sig}_{max_tokens}_{num_sequences}_{prompt_length}"
    
    if model_sig in BATCH_SIZE_CACHE and cache_key in BATCH_SIZE_CACHE[model_sig]:
        cached = BATCH_SIZE_CACHE[model_sig][cache_key]
        logger.info(f"Using cached batch sizes: rollout={cached['rollout']}, tf={cached['tf']}")
        return cached['rollout'], cached['tf']
    
    logger.info("Auto-detecting optimal batch sizes...")
    start_time = time.time()
    
    # 1. Find optimal rollout batch size (generation phase)
    logger.info("Testing generation batch sizes...")
    rollout_batch_size = binary_search_batch_size(
        test_batch_size_generation,
        min_size=1,
        max_size=64,  # Conservative max for generation
        model=model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        num_sequences=num_sequences,
        test_prompt_length=prompt_length
    )
    
    # 2. Find optimal teacher forcing batch size
    # This operates on flattened sequences (batch_size * num_sequences)
    logger.info("Testing teacher forcing batch sizes...")
    total_sequence_length = prompt_length + max_tokens
    
    # Start search from rollout_batch_size * num_sequences as baseline
    tf_baseline = rollout_batch_size * num_sequences
    tf_micro_batch = binary_search_batch_size(
        test_batch_size_teacher_forcing,
        min_size=max(1, tf_baseline // 4),  # Start conservative
        max_size=min(128, tf_baseline * 2),  # Don't go too high
        model=model,
        tokenizer=tokenizer,
        sequence_length=total_sequence_length
    )
    
    # Apply safety factor
    rollout_batch_size = max(1, int(rollout_batch_size * safety_factor))
    tf_micro_batch = max(1, int(tf_micro_batch * safety_factor))
    
    elapsed = time.time() - start_time
    logger.info(f"Auto-detection complete ({elapsed:.1f}s): "
                f"rollout_batch_size={rollout_batch_size}, "
                f"tf_micro_batch={tf_micro_batch}")
    
    # Cache the results
    if model_sig not in BATCH_SIZE_CACHE:
        BATCH_SIZE_CACHE[model_sig] = {}
    
    BATCH_SIZE_CACHE[model_sig][cache_key] = {
        'rollout': rollout_batch_size,
        'tf': tf_micro_batch
    }
    
    return rollout_batch_size, tf_micro_batch


def get_recommended_batch_sizes(
    model,
    tokenizer, 
    max_tokens: int = 200,
    num_sequences: int = 8,
    prompt_length: int = 100,
    mode: str = "auto"
) -> Tuple[int, int]:
    """
    Get recommended batch sizes with multiple strategies.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        max_tokens: Maximum new tokens
        num_sequences: Number of return sequences  
        prompt_length: Typical prompt length
        mode: "auto" (detect), "conservative" (safe defaults), "aggressive" (push limits)
    
    Returns:
        Tuple of (rollout_batch_size, tf_micro_batch_size)
    """
    if mode == "conservative":
        # Safe defaults that work on most GPUs
        return 8, 16
    
    elif mode == "aggressive":
        # Push limits with minimal safety factor
        return auto_detect_batch_sizes(
            model, tokenizer, max_tokens, num_sequences, prompt_length,
            safety_factor=0.95
        )
    
    else:  # mode == "auto"
        # Balanced approach with reasonable safety margin
        return auto_detect_batch_sizes(
            model, tokenizer, max_tokens, num_sequences, prompt_length,
            safety_factor=0.85
        )


def clear_batch_size_cache():
    """Clear the global batch size cache."""
    global BATCH_SIZE_CACHE
    BATCH_SIZE_CACHE.clear()
    logger.info("Batch size cache cleared")


def print_cache_stats():
    """Print statistics about cached batch sizes."""
    total_entries = sum(len(model_cache) for model_cache in BATCH_SIZE_CACHE.values())
    logger.info(f"Batch size cache: {len(BATCH_SIZE_CACHE)} models, {total_entries} configurations")
    
    for model_sig, configs in BATCH_SIZE_CACHE.items():
        logger.info(f"  {model_sig}: {len(configs)} cached configurations")