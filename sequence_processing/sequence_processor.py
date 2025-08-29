"""
SequenceProcessor: Master class for unified generation and logprob computation.

Consolidates patterns from collect_rollouts.py and dr_grpo.py into a single,
reusable interface for sequence generation across the RL_Practice project.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import random
import numpy as np

try:
    import rlp_datasets
except ImportError:
    rlp_datasets = None

from transformers import LogitsProcessor


class StopAfterAnswer(LogitsProcessor):
    """Stop generation after seeing 'The answer is' followed by a number or calculation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_tokens = []
        
        # Try different phrasings of "The answer is"
        variations = [
            "The answer is ",
            "The answer is:",
            "\nThe answer is ",
            "\nThe answer is:",
        ]
        
        for variation in variations:
            tokens = tokenizer.encode(variation, add_special_tokens=False)
            if tokens:
                self.stop_tokens.extend(tokens)
        
        self.stop_tokens = list(set(self.stop_tokens))
    
    def __call__(self, input_ids, scores):
        if len(input_ids[0]) < 10:
            return scores
            
        # Check last few tokens for stop patterns
        recent_tokens = input_ids[0][-10:].tolist()
        recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True).lower()
        
        if "the answer is" in recent_text:
            # Force EOS token
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                scores[:, eos_token_id] += 10000
        
        return scores


@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    do_sample: bool = True
    
    # Batch sizes
    gen_batch_size: int = 8
    tf_batch_size: int = 32


@dataclass
class BatchedSequences:
    """Container for batched generation results."""
    sequences: torch.Tensor           # [B, G, total_len]
    prompt_lens: List[int]            # [B] 
    gen_lens: List[List[int]]         # [B][G] actual generation lengths
    attention_masks: torch.Tensor     # [B, G, total_len]
    responses_text: List[List[str]]   # [B][G] decoded responses


@dataclass  
class LogprobResults:
    """Container for logprob computation results."""
    logprobs: List[List[torch.Tensor]]     # [B][G] per-token logprobs
    entropies: List[List[np.ndarray]]      # [B][G] per-token entropies  
    sequence_logprobs: List[List[float]]   # [B][G] total sequence logprobs


class SequenceProcessor:
    """Master class for unified sequence generation and logprob computation."""
    
    def __init__(self, model, tokenizer, config: Optional[GenerationConfig] = None):
        """Initialize SequenceProcessor.
        
        Args:
            model: The language model (may be DDP wrapped)
            tokenizer: The tokenizer
            config: Generation configuration (uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.is_ddp = hasattr(model, 'module')
        
    def _unwrap(self, model):
        """Unwrap DDP model if needed."""
        return model.module if self.is_ddp else model
        
    def sample_prompts(self, dataset_name: str, split: str, num_prompts: int, 
                      seed: Optional[int] = None) -> List[str]:
        """Sample prompts from a dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'gsm8k_r1_template')
            split: Dataset split (e.g., 'train', 'test')  
            num_prompts: Number of prompts to sample
            seed: Optional random seed for reproducible sampling
            
        Returns:
            List of prompt strings
        """
        if rlp_datasets is None:
            raise ImportError("rlp_datasets not available. Cannot sample from dataset.")
            
        if seed is not None:
            random.seed(seed)
            
        # Load dataset following the pattern from schedulers/mix_passrate.py
        dataset = rlp_datasets.DATASET_REGISTRY[dataset_name](split=split)
        
        # Convert to list and sample
        dataset_list = list(dataset)
        if len(dataset_list) < num_prompts:
            raise ValueError(f"Dataset {dataset_name}:{split} has only {len(dataset_list)} examples, "
                           f"but {num_prompts} prompts were requested.")
                           
        sampled_examples = random.sample(dataset_list, num_prompts)
        
        # Extract text prompts (following gsm8k_r1_template pattern)
        prompts = [example.text for example in sampled_examples]
        
        return prompts
    
    def generate_batched(self, prompts: List[str], G: int, 
                        gen_batch_size: Optional[int] = None) -> BatchedSequences:
        """Generate G responses per prompt using batched generation.
        
        Args:
            prompts: List of prompt strings
            G: Number of responses to generate per prompt  
            gen_batch_size: Batch size for generation (uses config default if None)
            
        Returns:
            BatchedSequences containing all generation results
        """
        if gen_batch_size is None:
            gen_batch_size = self.config.gen_batch_size
            
        B = len(prompts)
        model = self._unwrap(self.model)
        
        # Tokenize all prompts
        tokenized_prompts = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        prompt_input_ids = tokenized_prompts['input_ids'].to(model.device)
        prompt_attention_mask = tokenized_prompts['attention_mask'].to(model.device)
        prompt_lens = prompt_attention_mask.sum(dim=1).tolist()
        
        # Expand prompts for G generations: [B, seq_len] -> [B*G, seq_len]
        expanded_input_ids = prompt_input_ids.repeat_interleave(G, dim=0)
        expanded_attention_mask = prompt_attention_mask.repeat_interleave(G, dim=0)
        
        # Initialize logits processor
        stop_processor = StopAfterAnswer(self.tokenizer)
        
        # Generate in batches to manage memory
        all_sequences = []
        total_batches = (B * G + gen_batch_size - 1) // gen_batch_size
        
        for i in range(0, B * G, gen_batch_size):
            end_idx = min(i + gen_batch_size, B * G)
            batch_input_ids = expanded_input_ids[i:end_idx]
            batch_attention_mask = expanded_attention_mask[i:end_idx]
            
            # Generate batch
            with torch.no_grad():
                gen_output = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    logits_processor=[stop_processor],
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                generated = gen_output.sequences
            
            all_sequences.append(generated)
        
        # Pad and concatenate all generated sequences to handle different lengths
        if len(all_sequences) == 1:
            sequences = all_sequences[0]
        else:
            # Find max length across all batches
            max_len = max(seq.size(1) for seq in all_sequences)
            
            # Pad sequences to max length
            padded_sequences = []
            for seq_batch in all_sequences:
                if seq_batch.size(1) < max_len:
                    pad_size = max_len - seq_batch.size(1)
                    padding = torch.full(
                        (seq_batch.size(0), pad_size),
                        self.tokenizer.pad_token_id,
                        device=seq_batch.device,
                        dtype=seq_batch.dtype
                    )
                    seq_batch = torch.cat([seq_batch, padding], dim=1)
                padded_sequences.append(seq_batch)
            
            sequences = torch.cat(padded_sequences, dim=0)  # [B*G, total_len]
        
        # Reshape to [B, G, total_len]  
        max_len = sequences.size(1)
        sequences = sequences.view(B, G, max_len)
        
        # Create attention masks
        attention_masks = (sequences != self.tokenizer.pad_token_id)
        
        # Calculate generation lengths for each sequence
        gen_lens = []
        responses_text = []
        
        for b in range(B):
            prompt_len = prompt_lens[b]
            batch_gen_lens = []
            batch_responses = []
            
            for g in range(G):
                seq = sequences[b, g]
                # Find actual sequence length (excluding padding)
                non_pad_mask = seq != self.tokenizer.pad_token_id
                if non_pad_mask.any():
                    actual_len = non_pad_mask.sum().item()
                    gen_len = max(0, actual_len - prompt_len)
                else:
                    gen_len = 0
                
                batch_gen_lens.append(gen_len)
                
                # Decode response text (generation part only)
                if gen_len > 0:
                    gen_tokens = seq[prompt_len:prompt_len + gen_len]
                    response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                else:
                    response_text = ""
                    
                batch_responses.append(response_text)
            
            gen_lens.append(batch_gen_lens)
            responses_text.append(batch_responses)
        
        return BatchedSequences(
            sequences=sequences,
            prompt_lens=prompt_lens, 
            gen_lens=gen_lens,
            attention_masks=attention_masks,
            responses_text=responses_text
        )
    
    def teacher_force_logprobs(self, sequences: BatchedSequences, 
                              with_grad: bool = False,
                              tf_batch_size: Optional[int] = None) -> LogprobResults:
        """Compute logprobs for generated sequences using teacher forcing.
        
        Args:
            sequences: BatchedSequences from generation
            with_grad: Whether to compute gradients (True) or not (False)
            tf_batch_size: Batch size for teacher forcing (uses config default if None)
            
        Returns:
            LogprobResults containing logprobs and entropies
        """
        if tf_batch_size is None:
            tf_batch_size = self.config.tf_batch_size
            
        if with_grad:
            return self._teacher_force_with_grad(sequences, tf_batch_size)
        else:
            return self._teacher_force_no_grad(sequences, tf_batch_size)
    
    def _teacher_force_no_grad(self, sequences: BatchedSequences, tf_batch_size: int) -> LogprobResults:
        """Teacher forcing without gradients - simplified approach."""
        model = self._unwrap(self.model)
        B, G = sequences.sequences.shape[:2]
        
        # Initialize results
        all_logprobs = [[] for _ in range(B)]
        all_entropies = [[] for _ in range(B)]
        all_sequence_logprobs = [[] for _ in range(B)]
        
        with torch.no_grad():
            # Process sequences one by one for simplicity (can optimize later)
            for b in range(B):
                for g in range(G):
                    seq = sequences.sequences[b, g]  # [seq_len]
                    prompt_len = sequences.prompt_lens[b]
                    gen_len = sequences.gen_lens[b][g]
                    
                    if gen_len > 0 and seq.size(0) > prompt_len:
                        # Get the actual sequence length (non-padded)
                        non_pad_mask = seq != self.tokenizer.pad_token_id
                        if non_pad_mask.sum() > prompt_len:
                            actual_len = min(prompt_len + gen_len, non_pad_mask.sum().item())
                            input_seq = seq[:actual_len].unsqueeze(0)  # [1, actual_len]
                            
                            # Forward pass
                            outputs = model(input_seq)
                            logits = outputs.logits[0]  # [actual_len, vocab_size]
                            
                            # Compute log probabilities for generated tokens
                            gen_start = prompt_len
                            gen_end = min(actual_len, prompt_len + gen_len)
                            
                            if gen_end > gen_start:
                                gen_logits = logits[gen_start-1:gen_end-1]  # [gen_len, vocab_size] 
                                gen_tokens = seq[gen_start:gen_end]  # [gen_len]
                                
                                # Get logprobs for the actual tokens
                                log_probs = torch.log_softmax(gen_logits, dim=-1)
                                token_logprobs = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)
                                
                                # Convert to numpy
                                gen_logprobs_np = token_logprobs.cpu().numpy()
                                gen_logprobs_np = np.nan_to_num(gen_logprobs_np, nan=0.0, posinf=0.0, neginf=-100.0)
                                
                                # Compute entropies (placeholder)
                                entropies = np.ones_like(gen_logprobs_np) * 0.5
                                
                                # Total sequence logprob
                                seq_logprob = float(gen_logprobs_np.sum())
                            else:
                                gen_logprobs_np = np.array([])
                                entropies = np.array([])
                                seq_logprob = 0.0
                        else:
                            gen_logprobs_np = np.array([])
                            entropies = np.array([])
                            seq_logprob = 0.0
                    else:
                        gen_logprobs_np = np.array([])
                        entropies = np.array([])
                        seq_logprob = 0.0
                    
                    all_logprobs[b].append(torch.from_numpy(gen_logprobs_np))
                    all_entropies[b].append(entropies)
                    all_sequence_logprobs[b].append(seq_logprob)
        
        return LogprobResults(
            logprobs=all_logprobs,
            entropies=all_entropies,
            sequence_logprobs=all_sequence_logprobs
        )
    
    def _teacher_force_with_grad(self, sequences: BatchedSequences, tf_batch_size: int) -> LogprobResults:
        """Teacher forcing with gradients - following dr_grpo.py pattern."""
        model = self._unwrap(self.model)
        B, G = sequences.sequences.shape[:2]
        
        # For now, return a simple implementation
        # TODO: Implement full gradient-enabled version following dr_grpo.py _policy_logp
        all_logprobs = [[] for _ in range(B)]
        all_entropies = [[] for _ in range(B)]
        all_sequence_logprobs = [[] for _ in range(B)]
        
        # Placeholder implementation
        for b in range(B):
            for g in range(G):
                gen_len = sequences.gen_lens[b][g]
                if gen_len > 0:
                    # Placeholder tensors that require grad
                    logprobs = torch.zeros(gen_len, requires_grad=True)
                    entropies = np.ones(gen_len) * 0.5
                    seq_logprob = 0.0
                else:
                    logprobs = torch.tensor([], requires_grad=True)
                    entropies = np.array([])
                    seq_logprob = 0.0
                
                all_logprobs[b].append(logprobs)
                all_entropies[b].append(entropies)
                all_sequence_logprobs[b].append(seq_logprob)
        
        return LogprobResults(
            logprobs=all_logprobs,
            entropies=all_entropies,
            sequence_logprobs=all_sequence_logprobs
        )
    
    def generate_with_logprobs(self, prompts: Optional[List[str]] = None, G: int = 8,
                              dataset_name: Optional[str] = None, split: str = "train",
                              num_prompts: Optional[int] = None, seed: Optional[int] = None,
                              with_grad: bool = False,
                              gen_batch_size: Optional[int] = None,
                              tf_batch_size: Optional[int] = None) -> Tuple[BatchedSequences, LogprobResults]:
        """Main interface: Generate sequences and compute logprobs.
        
        Args:
            prompts: Explicit prompts OR None to sample from dataset
            G: Number of responses per prompt
            dataset_name: Dataset name if sampling (e.g., 'gsm8k_r1_template')  
            split: Dataset split if sampling (e.g., 'train')
            num_prompts: Number of prompts to sample if using dataset
            seed: Random seed for dataset sampling
            with_grad: Whether to compute gradients in teacher forcing
            gen_batch_size: Generation batch size
            tf_batch_size: Teacher forcing batch size
            
        Returns:
            (BatchedSequences, LogprobResults)
        """
        # Get prompts either from parameter or by sampling dataset
        if prompts is None:
            if dataset_name is None or num_prompts is None:
                raise ValueError("Must provide either prompts OR (dataset_name + num_prompts)")
            prompts = self.sample_prompts(dataset_name, split, num_prompts, seed)
        
        # Generate sequences
        sequences = self.generate_batched(prompts, G, gen_batch_size)
        
        # Compute logprobs
        logprob_results = self.teacher_force_logprobs(sequences, with_grad, tf_batch_size)
        
        return sequences, logprob_results