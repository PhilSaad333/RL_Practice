"""
SequenceProcessor: Master class for unified generation and logprob computation.

Consolidates patterns from collect_rollouts.py and dr_grpo.py into a single,
reusable interface for sequence generation across the RL_Practice project.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict
import random
import numpy as np

try:
    import rlp_datasets
except ImportError:
    rlp_datasets = None

from transformers import LogitsProcessor


class StopAfterAnswer(LogitsProcessor):
    """Stop generation after seeing '</answer>' tag - correct logic from collect_rollouts.py"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Use the correct stop tag from the dataset format
        TAG_STOP = "</answer>"
        self.tag_ids = tokenizer(TAG_STOP, add_special_tokens=False).input_ids
        self.L = len(self.tag_ids)
    
    def __call__(self, input_ids, scores):
        tag = torch.tensor(self.tag_ids, device=input_ids.device)
        done = (input_ids[:, -self.L:] == tag).all(-1)
        if done.any():
            scores[done] = float("-inf")
            scores[done, self.tokenizer.pad_token_id] = 0.0
        return scores


@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 200
    do_sample: bool = True
    num_return_sequences: int = 8  # Match collect_rollouts.py default
    
    # Batch sizes (optimized for H100 80GB)
    gen_batch_size: int = 32  # Conservative default, can go higher
    tf_batch_size: int = 64   # Teacher forcing can handle larger batches
    
    # Phase 2: Enable differentiable RB entropies for gradient computation
    rb_requires_grad: bool = False


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
    rb_entropies: List[List[np.ndarray]]   # [B][G] per-token RB entropies under top-p (or full-softmax)
    rewards: List[List[float]]             # [B][G] rewards per sequence using tag_pref reward function
    # Phase 2: Optional torch tensors for differentiable RB entropies
    rb_entropies_torch: Optional[List[List[torch.Tensor]]] = None  # [B][G] torch RB entropies (for grad path)
    # Phase 3b: Optional per-step feature matrix φ_j for regression baseline
    baseline_feats_torch: Optional[List[List[torch.Tensor]]] = None  # [B][G] features [T, d] (detached)


@dataclass
class DiagnosticsResults:
    """Container for distribution diagnostics results."""
    diagnostics: List[List["DiagnosticsPack"]]  # [B][G] diagnostics per sequence




"""
Below are dataclasses for various sequence diagnostics, which primarily are going to be used to find potential
control variates for variance reduction in entropy estimation.
"""

@dataclass
class TokenStepDiagnostics:
    """Per-step token-distribution diagnostics (length T)."""
    rb_entropy: np.ndarray          # H(q_k)
    head_mass: np.ndarray           # s_k = mass kept by top-p (or 1.0 if full)
    tail_mass: np.ndarray           # eps_k = 1 - s_k
    two_point_entropy: np.ndarray   # H([s_k, eps_k])
    top1_prob: np.ndarray           # max_j q_{k,j}
    margin: np.ndarray              # a_{(1)} - a_{(2)}  (after temperature)
    collision: np.ndarray           # sum_j q_{k,j}^2
    renyi2: np.ndarray              # -log collision
    eff_support: np.ndarray         # exp(rb_entropy)
    logit_mean: np.ndarray          # E_q[a]
    logit_var: np.ndarray           # Var_q(a)
    eos_prob: Optional[np.ndarray]  # q_k(EOS) if eos_token_id is provided, else None

@dataclass
class SequenceDiagnostics:
    """Sequence-level aggregates computed from TokenStepDiagnostics."""
    T: int                          # number of generated tokens
    rb_entropy_sum: float
    rb_entropy_mean: float
    rb_entropy_max: float
    rb_entropy_min: float
    early_rb_entropy_mean: float    # mean over first K steps (if T>=K)
    late_rb_entropy_mean: float     # mean over remaining steps (if T>=K)
    naive_surprisal_sum: float      # sum(-log p(y_k)) for realized tokens
    naive_surprisal_mean: float
    margin_mean: float
    margin_sum: float
    top1_prob_mean: float
    collision_mean: float
    renyi2_mean: float
    eff_support_mean: float
    eos_prob_mean: Optional[float]

@dataclass
class DiagnosticsPack:
    """Container returned per sequence sample."""
    step: TokenStepDiagnostics
    seq: SequenceDiagnostics











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
        self.diag_results = []
        
    def _unwrap(self, model):
        """Unwrap DDP model if needed."""
        return model.module if self.is_ddp else model
        
    def _compute_rewards(self, prompts: List[str], sequences: BatchedSequences, examples: Optional[List] = None) -> List[List[float]]:
        """
        Compute rewards for generated sequences using the tag_pref reward function.
        
        Args:
            prompts: List of prompts used for generation [B]
            sequences: Generated sequences from generate_with_logprobs
            examples: Dataset examples containing gold answers [B] (optional, will try to infer from prompts)
            
        Returns:
            rewards: [B][G] list of rewards for each sequence
        """
        from rl_training.rewards.tag_pref import reward_fn, PROMPT2GOLD
        
        B, G = sequences.sequences.shape[:2]
        all_rewards = []
        
        # Store original PROMPT2GOLD mapping to restore later
        original_mapping = PROMPT2GOLD.copy()
        
        try:
            for b in range(B):
                # Get gold answer
                gold_answer = None
                if examples and b < len(examples):
                    gold_answer = examples[b].answer
                else:
                    # Try to infer from dataset if prompts match a known pattern
                    # This is a fallback - ideally examples should be provided
                    pass
                
                if gold_answer is None:
                    # No gold answer available, give zero rewards
                    all_rewards.append([0.0] * G)
                    continue
                
                # Set up temporary PROMPT2GOLD mapping
                prompt_id = b  # Use batch index as prompt ID
                PROMPT2GOLD[prompt_id] = gold_answer
                
                # Get responses for this prompt
                responses = sequences.responses_text[b]  # [G] list of response strings
                
                # Compute rewards using tag_pref function
                try:
                    reward_tensor = reward_fn(prompt_id, responses)  # returns torch.Tensor [G]
                    rewards = reward_tensor.tolist()  # Convert to list
                except Exception as e:
                    # Fallback to zero rewards if computation fails
                    rewards = [0.0] * G
                    print(f"Warning: Failed to compute rewards for prompt {b}: {e}")
                
                all_rewards.append(rewards)
                
        finally:
            # Restore original PROMPT2GOLD mapping
            PROMPT2GOLD.clear()
            PROMPT2GOLD.update(original_mapping)
        
        return all_rewards
        
    def sample_prompts(self, dataset_name: str, split: str, num_prompts: Optional[int], 
                      seed: Optional[int] = None) -> Tuple[List[str], List]:
        """Sample prompts from a dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'gsm8k_r1_template')
            split: Dataset split (e.g., 'train', 'test')  
            num_prompts: Number of prompts to sample (None = return all prompts)
            seed: Optional random seed for reproducible sampling
            
        Returns:
            Tuple of (prompt strings, dataset examples)
        """
        if rlp_datasets is None:
            raise ImportError("rlp_datasets not available. Cannot sample from dataset.")
            
        if seed is not None:
            random.seed(seed)
            
        # Load dataset following the pattern from schedulers/mix_passrate.py
        dataset = rlp_datasets.DATASET_REGISTRY[dataset_name](split=split)
        
        # Convert to list
        dataset_list = list(dataset)
        
        # Sample or return all
        if num_prompts is None:
            # Return all prompts (no sampling)
            sampled_examples = dataset_list
        else:
            # Sample specified number
            if len(dataset_list) < num_prompts:
                raise ValueError(f"Dataset {dataset_name}:{split} has only {len(dataset_list)} examples, "
                               f"but {num_prompts} prompts were requested.")
            sampled_examples = random.sample(dataset_list, num_prompts)
        
        # Extract question prompts (correct field for generation - ends with <think>)
        prompts = [example.question for example in sampled_examples]
        
        return prompts, sampled_examples
    
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
        
        # Tokenize all prompts (no truncation to avoid cutting off prompts)
        tokenized_prompts = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=False  # Allow full prompt length
        )
        prompt_input_ids = tokenized_prompts['input_ids'].to(model.device)
        prompt_attention_mask = tokenized_prompts['attention_mask'].to(model.device)
        # Use padded length for extraction, not attention mask sum
        # This is critical for batch processing with different prompt lengths
        padded_prompt_len = prompt_input_ids.size(1)  # All prompts padded to same length
        
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
                    do_sample=self.config.do_sample,
                    num_return_sequences=1,  # Fixed: we already expanded inputs for G generations
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    logits_processor=[stop_processor],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,  # Match collect_rollouts.py
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    synced_gpus=False  # Match collect_rollouts.py - prevents distributed deadlocks
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
        
        # Extract generation-only tokens following collect_rollouts.py approach
        # sequences shape: [B, G, total_len] where total_len includes prompt + generation
        gen_lens = []
        responses_text = []
        
        for b in range(B):
            # Use padded prompt length for ALL prompts in batch (critical fix!)
            batch_gen_lens = []
            batch_responses = []
            
            for g in range(G):
                full_seq = sequences[b, g]  # Full sequence including prompt
                
                # Extract generation-only tokens using padded length (works for all prompts in batch)
                gen_tokens = full_seq[padded_prompt_len:]  # Slice off padded prompt, keep generation
                
                # Remove padding tokens from generation
                non_pad_mask = gen_tokens != self.tokenizer.pad_token_id
                if non_pad_mask.any():
                    # Find last non-pad token
                    last_valid_idx = non_pad_mask.nonzero()[-1].item() + 1
                    gen_tokens = gen_tokens[:last_valid_idx]
                    gen_len = last_valid_idx
                else:
                    gen_tokens = torch.tensor([], dtype=gen_tokens.dtype, device=gen_tokens.device)
                    gen_len = 0
                
                batch_gen_lens.append(gen_len)
                
                # Decode generation-only tokens
                if gen_len > 0:
                    response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                else:
                    response_text = ""
                    
                batch_responses.append(response_text)
            
            gen_lens.append(batch_gen_lens)
            responses_text.append(batch_responses)
        
        return BatchedSequences(
            sequences=sequences,
            prompt_lens=[padded_prompt_len] * B,  # All prompts have same padded length
            gen_lens=gen_lens,
            attention_masks=attention_masks,
            responses_text=responses_text
        )
    
    def teacher_force_logprobs(self, sequences: BatchedSequences, 
                              with_grad: bool = False,
                              tf_batch_size: Optional[int] = None,
                              compute_rb: bool = False,
                              return_baseline_features: bool = False) -> LogprobResults:
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
            return self._teacher_force_with_grad(sequences, tf_batch_size, compute_rb, return_baseline_features)
        else:
            return self._teacher_force_no_grad(sequences, tf_batch_size, compute_rb, return_baseline_features)
    
    def teacher_force_logprobs_with_diagnostics(self, sequences: BatchedSequences, 
                                              with_grad: bool = False,
                                              tf_batch_size: Optional[int] = None,
                                              compute_rb: bool = False,
                                              return_baseline_features: bool = False) -> Tuple[LogprobResults, DiagnosticsResults]:
        """Compute logprobs and diagnostics for generated sequences using teacher forcing.
        
        Args:
            sequences: BatchedSequences from generation
            with_grad: Whether to compute gradients (True) or not (False)
            tf_batch_size: Batch size for teacher forcing (uses config default if None)
            compute_rb: Whether to compute RB entropies
            
        Returns:
            (LogprobResults, DiagnosticsResults)
        """
        if tf_batch_size is None:
            tf_batch_size = self.config.tf_batch_size
            
        if with_grad:
            return self._teacher_force_with_grad(sequences, tf_batch_size, compute_rb, return_baseline_features)
        else:
            return self._teacher_force_no_grad(sequences, tf_batch_size, compute_rb, return_baseline_features)
    
    def _teacher_force_no_grad(
        self, sequences: BatchedSequences, tf_batch_size: int, compute_rb: bool, return_baseline_features: bool = False
    ) -> Tuple[LogprobResults, DiagnosticsResults]:
        model = self._unwrap(self.model)
        B, G = sequences.sequences.shape[:2]

        all_logprobs = [[] for _ in range(B)]
        all_entropies = [[] for _ in range(B)]
        all_rb_entropies = [[] for _ in range(B)]
        all_sequence_logprobs = [[] for _ in range(B)]
        
        # Build diagnostics helper once (outside loops)
        diag = DistributionDiagnostics(
            top_p=getattr(self.config, "top_p", 1.0),
            temperature=getattr(self.config, "temperature", 1.0),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        
        # Storage for diagnostics
        all_diagnostics = [[] for _ in range(B)]
        
        # Helper to create empty diagnostics pack
        def empty_diagnostics_pack():
            empty_step = TokenStepDiagnostics(
                rb_entropy=np.array([]),
                head_mass=np.array([]),
                tail_mass=np.array([]),
                two_point_entropy=np.array([]),
                top1_prob=np.array([]),
                margin=np.array([]),
                collision=np.array([]),
                renyi2=np.array([]),
                eff_support=np.array([]),
                logit_mean=np.array([]),
                logit_var=np.array([]),
                eos_prob=None
            )
            empty_seq = SequenceDiagnostics(
                T=0,
                rb_entropy_sum=0.0,
                rb_entropy_mean=0.0,
                rb_entropy_max=0.0,
                rb_entropy_min=0.0,
                early_rb_entropy_mean=0.0,
                late_rb_entropy_mean=0.0,
                naive_surprisal_sum=0.0,
                naive_surprisal_mean=0.0,
                margin_mean=0.0,
                margin_sum=0.0,
                top1_prob_mean=0.0,
                collision_mean=0.0,
                renyi2_mean=0.0,
                eff_support_mean=0.0,
                eos_prob_mean=None
            )
            return DiagnosticsPack(step=empty_step, seq=empty_seq)

        with torch.no_grad():
            for b in range(B):
                for g in range(G):
                    seq = sequences.sequences[b, g]          # [total_len]
                    prompt_len_padded = sequences.prompt_lens[b]    # this is the batch's padded prompt length
                    gen_len = sequences.gen_lens[b][g]
                    seq_len_total = seq.size(0)

                    if gen_len > 0 and seq_len_total > prompt_len_padded:
                        # Take prompt + exactly gen_len tokens (prefix used for TF)
                        actual_len = min(prompt_len_padded + gen_len, seq_len_total)
                        input_seq = seq[:actual_len].unsqueeze(0)  # [1, actual_len]

                        # ATTENTION: use the attention mask so left pads are ignored correctly
                        attn = sequences.attention_masks[b, g, :actual_len].unsqueeze(0).to(input_seq.device)

                        outputs = model(input_seq, attention_mask=attn)
                        logits = outputs.logits[0]  # [actual_len, V]

                        gen_start = prompt_len_padded
                        gen_end   = prompt_len_padded + gen_len

                        # Logits aligned to next-token targets for the generated tokens:
                        # token t at position i uses logits at i-1
                        gen_logits = logits[gen_start-1 : gen_end-1]   # [T = gen_len, V]
                        gen_tokens = seq[gen_start : gen_end]          # [T]

                        if gen_logits.size(0) == gen_tokens.size(0) and gen_logits.size(0) > 0:
                                
                                # Compute diagnostics from logits
                                pack = diag.compute_from_logits(gen_logits, gen_tokens)
                                all_diagnostics[b].append(pack)

                                # Naïve per-token logprobs for realized tokens
                                log_probs = torch.log_softmax(gen_logits, dim=-1)                # [T, V]
                                token_logprobs = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)  # [T]

                                # Convert to numpy
                                gen_logprobs_np = token_logprobs.float().cpu().numpy()
                                gen_logprobs_np = np.nan_to_num(gen_logprobs_np, nan=0.0, posinf=0.0, neginf=-100.0)

                                # Naïve per-token "entropies" (surprisal of realized token)
                                entropies_naive = -gen_logprobs_np                                 # [T]

                                # RB per-step entropies under the SAME top-p/temperature policy
                                if compute_rb:
                                    rb_H = self._rb_entropies_top_p(
                                        gen_logits, self.config.top_p, self.config.temperature
                                    )                                                               # [T]
                                    rb_np = rb_H.float().cpu().numpy()
                                else:
                                    rb_np = np.array([])

                                seq_logprob = float(gen_logprobs_np.sum())
                        else:
                            # Size mismatch or zero generation length
                            gen_logprobs_np = np.array([])
                            entropies_naive = np.array([])
                            rb_np = np.array([])
                            seq_logprob = 0.0
                            all_diagnostics[b].append(empty_diagnostics_pack())
                    else:
                        # No generation found for this (b,g); record empties
                        gen_logprobs_np = np.array([])
                        entropies_naive = np.array([])
                        rb_np = np.array([])
                        seq_logprob = 0.0
                        all_diagnostics[b].append(empty_diagnostics_pack())

                    all_logprobs[b].append(torch.from_numpy(gen_logprobs_np))
                    all_entropies[b].append(entropies_naive)
                    all_rb_entropies[b].append(rb_np)
                    all_sequence_logprobs[b].append(seq_logprob)

        logprob_results = LogprobResults(
            logprobs=all_logprobs,
            entropies=all_entropies,
            sequence_logprobs=all_sequence_logprobs,
            rb_entropies=all_rb_entropies,
            rewards=[],  # Will be filled by generate_with_logprobs
            rb_entropies_torch=None,  # no grad path doesn't compute torch tensors
            baseline_feats_torch=None,  # no grad path doesn't compute baseline features
        )
        
        diagnostics_results = DiagnosticsResults(
            diagnostics=all_diagnostics
        )
        
        return logprob_results, diagnostics_results
    
    def _teacher_force_with_grad(
        self, sequences: BatchedSequences, tf_batch_size: int, compute_rb: bool, return_baseline_features: bool = False
    ) -> Tuple[LogprobResults, DiagnosticsResults]:
        """
        Gradient-enabled version. By default we do NOT build a gradient graph for RB entropies,
        since these are typically diagnostics. If you want grads for RB later, remove .detach().
        """
        model = self._unwrap(self.model)
        B, G = sequences.sequences.shape[:2]

        all_logprobs = [[] for _ in range(B)]
        all_entropies = [[] for _ in range(B)]
        all_rb_entropies = [[] for _ in range(B)]
        all_sequence_logprobs = [[] for _ in range(B)]
        rb_entropies_torch = [[] for _ in range(B)] if compute_rb and self.config.rb_requires_grad else None
        baseline_feats_torch = [[] for _ in range(B)] if return_baseline_features else None

        for b in range(B):
            for g in range(G):
                seq = sequences.sequences[b, g]          # [total_len]
                prompt_len_padded = sequences.prompt_lens[b]    # this is the batch's padded prompt length
                gen_len = sequences.gen_lens[b][g]
                seq_len_total = seq.size(0)

                if gen_len > 0 and seq_len_total > prompt_len_padded:
                    # Take prompt + exactly gen_len tokens (prefix used for TF)
                    actual_len = min(prompt_len_padded + gen_len, seq_len_total)
                    input_seq = seq[:actual_len].unsqueeze(0)  # [1, actual_len]

                    # ATTENTION: use the attention mask so left pads are ignored correctly
                    attn = sequences.attention_masks[b, g, :actual_len].unsqueeze(0).to(input_seq.device)

                    outputs = model(input_seq, attention_mask=attn)  # grads enabled
                    logits = outputs.logits[0]                # [actual_len, V]

                    gen_start = prompt_len_padded
                    gen_end   = prompt_len_padded + gen_len

                    # Logits aligned to next-token targets for the generated tokens:
                    # token t at position i uses logits at i-1
                    gen_logits = logits[gen_start-1 : gen_end-1]   # [T = gen_len, V]
                    gen_tokens = seq[gen_start : gen_end]          # [T]

                    if gen_logits.size(0) == gen_tokens.size(0) and gen_logits.size(0) > 0:
                        log_probs = torch.log_softmax(gen_logits, dim=-1)
                        token_logprobs = log_probs.gather(1, gen_tokens.unsqueeze(1)).squeeze(1)   # [T]

                        # naive surprisal as numpy for now (diagnostic); keep torch if you want grads
                        entropies_naive = (-token_logprobs).detach().float().cpu().numpy()

                        # --- RB entropy path (config-gated) ---
                        rb_np = np.array([])
                        rb_t = None
                        if compute_rb:
                            if self.config.rb_requires_grad:
                                # DIFFERENTIABLE RB: do NOT wrap in torch.no_grad()
                                rb_t = self._rb_entropies_top_p(gen_logits, self.config.top_p, self.config.temperature)  # [T], torch
                                rb_np = rb_t.detach().float().cpu().numpy()
                            else:
                                # Diagnostics-only RB: no grad
                                with torch.no_grad():
                                    rb_H = self._rb_entropies_top_p(gen_logits, self.config.top_p, self.config.temperature)
                                    rb_np = rb_H.float().cpu().numpy()

                        seq_logprob = float(token_logprobs.detach().sum().item())
                        
                        # --- Baseline features (Phase 3b) ---
                        phi = None
                        if return_baseline_features:
                            with torch.no_grad():
                                # Derive masked logits a_masked and q exactly as in RB computation
                                a = gen_logits / self.config.temperature  # [T, V]
                                if self.config.top_p < 1.0:
                                    p_full = torch.softmax(a, dim=-1)  # [T,V]
                                    p_sorted, idx_sorted = p_full.sort(dim=-1, descending=True)
                                    cumsum = p_sorted.cumsum(dim=-1)
                                    keep_sorted = (cumsum - p_sorted) <= self.config.top_p
                                    keep_sorted[..., 0] = True
                                    keep = torch.zeros_like(p_full, dtype=torch.bool)
                                    keep.scatter_(dim=-1, index=idx_sorted, src=keep_sorted)
                                    a_masked = a.masked_fill(~keep, float('-inf'))
                                    q = torch.softmax(a_masked, dim=-1)  # [T,V]
                                    s = (p_full * keep).sum(dim=-1)  # [T] head mass
                                    eps = (1.0 - s).clamp_min(0.0)
                                    Z_S = torch.logsumexp(a_masked, dim=-1)  # [T]
                                else:
                                    a_masked = a
                                    q = torch.softmax(a_masked, dim=-1)
                                    s = torch.ones(a.size(0), device=a.device)
                                    eps = torch.zeros_like(s)
                                    Z_S = torch.logsumexp(a_masked, dim=-1)
                                
                                # H (RB entropy) on q
                                H = Z_S - (q * a).sum(dim=-1)  # [T]
                                
                                # top1 prob and margin
                                q_sorted, _ = q.sort(dim=-1, descending=True)
                                top1 = q_sorted[..., 0]  # [T]
                                a_sorted, _ = a_masked.sort(dim=-1, descending=True)
                                a1 = a_sorted[..., 0]
                                if a_sorted.shape[-1] > 1:
                                    a2 = a_sorted[..., 1]
                                else:
                                    a2 = torch.full_like(a1, -float('inf'))
                                margin = (a1 - a2).masked_fill(~torch.isfinite(a2), 0.0)  # [T]
                                
                                # two-point entropy H([s,eps])
                                def _slog(x): 
                                    return torch.log(x.clamp_min(1e-38))
                                H2pt = -(s * _slog(s) + eps * _slog(eps))  # [T]
                                
                                # logit moments under q
                                Ea = (q * a).sum(dim=-1)
                                Ea2 = (q * (a * a)).sum(dim=-1)
                                var_a = (Ea2 - Ea * Ea).clamp_min(0.0)  # [T]
                                
                                # position fraction j/L
                                T = a.size(0)
                                pos_frac = torch.arange(T, device=a.device, dtype=torch.float32) / max(T, 1)
                                
                                # Stack features in order: ["H", "top1", "margin", "head_mass", "two_point_entropy", "logit_var", "pos_frac"]
                                feats = [H, top1, margin, s, H2pt, var_a, pos_frac]  # list of [T]
                                phi = torch.stack(feats, dim=-1).detach()  # [T, d]
                        
                        # store torch tensors with grad
                        all_logprobs[b].append(token_logprobs)
                        all_entropies[b].append(entropies_naive)
                        all_rb_entropies[b].append(rb_np)
                        all_sequence_logprobs[b].append(seq_logprob)
                        
                        # Store torch RB if available
                        if rb_entropies_torch is not None:
                            if rb_t is not None:
                                rb_entropies_torch[b].append(rb_t)
                            else:
                                rb_entropies_torch[b].append(torch.tensor([], device=gen_logits.device))
                        
                        # Store baseline features if available
                        if baseline_feats_torch is not None:
                            if phi is not None:
                                baseline_feats_torch[b].append(phi)
                            else:
                                baseline_feats_torch[b].append(torch.tensor([]).view(0, 7))  # empty [0, 7] tensor
                    else:
                        # Size mismatch or zero generation length
                        all_logprobs[b].append(torch.tensor([], requires_grad=True))
                        all_entropies[b].append(np.array([]))
                        all_rb_entropies[b].append(np.array([]))
                        all_sequence_logprobs[b].append(0.0)
                        
                        # Store empty torch RB if needed
                        if rb_entropies_torch is not None:
                            rb_entropies_torch[b].append(torch.tensor([], device=next(model.parameters()).device))
                        
                        # Store empty baseline features if needed
                        if baseline_feats_torch is not None:
                            baseline_feats_torch[b].append(torch.tensor([]).view(0, 7))

        logprob_results = LogprobResults(
            logprobs=all_logprobs,
            entropies=all_entropies,
            sequence_logprobs=all_sequence_logprobs,
            rb_entropies=all_rb_entropies,
            rewards=[],  # Will be filled by generate_with_logprobs
            rb_entropies_torch=rb_entropies_torch,  # Phase 2: torch RB tensors
            baseline_feats_torch=baseline_feats_torch,  # Phase 3b: baseline features
        )
        
        # For now, return empty diagnostics for the with_grad version
        # TODO: Add full diagnostics support if needed
        B, G = len(all_logprobs), len(all_logprobs[0]) if all_logprobs else 0
        empty_diagnostics = [[DiagnosticsPack(
            step=TokenStepDiagnostics(
                rb_entropy=np.array([]),
                head_mass=np.array([]),
                tail_mass=np.array([]),
                two_point_entropy=np.array([]),
                top1_prob=np.array([]),
                margin=np.array([]),
                collision=np.array([]),
                renyi2=np.array([]),
                eff_support=np.array([]),
                logit_mean=np.array([]),
                logit_var=np.array([]),
                eos_prob=None
            ),
            seq=SequenceDiagnostics(
                T=0, rb_entropy_sum=0.0, rb_entropy_mean=0.0, rb_entropy_max=0.0, rb_entropy_min=0.0,
                early_rb_entropy_mean=0.0, late_rb_entropy_mean=0.0, naive_surprisal_sum=0.0, 
                naive_surprisal_mean=0.0, margin_mean=0.0, margin_sum=0.0, top1_prob_mean=0.0,
                collision_mean=0.0, renyi2_mean=0.0, eff_support_mean=0.0, eos_prob_mean=None
            )
        ) for g in range(G)] for b in range(B)]
        
        diagnostics_results = DiagnosticsResults(diagnostics=empty_diagnostics)
        
        return logprob_results, diagnostics_results
    
    def generate_with_logprobs(self, prompts: Optional[List[str]] = None, G: int = 8,
                              dataset_name: Optional[str] = None, split: str = "train",
                              num_prompts: Optional[int] = None, seed: Optional[int] = None,
                              with_grad: bool = False,
                              gen_batch_size: Optional[int] = None,
                              tf_batch_size: Optional[int] = None,
                              compute_rb: bool = False,) -> Tuple[BatchedSequences, LogprobResults, DiagnosticsResults]:
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
            (BatchedSequences, LogprobResults, DiagnosticsResults)
        """
        # Get prompts and examples either from parameter or by sampling dataset
        examples = None
        if prompts is None:
            if dataset_name is None or num_prompts is None:
                raise ValueError("Must provide either prompts OR (dataset_name + num_prompts)")
            prompts, examples = self.sample_prompts(dataset_name, split, num_prompts, seed)
        
        # Generate sequences
        sequences = self.generate_batched(prompts, G, gen_batch_size)
        
        # Compute logprobs and diagnostics
        logprob_results, diagnostics_results = self.teacher_force_logprobs_with_diagnostics(sequences, with_grad, tf_batch_size, compute_rb)
        
        # Compute rewards 
        rewards = self._compute_rewards(prompts, sequences, examples)
        
        # Add rewards to logprob results
        logprob_results.rewards = rewards
        
        return sequences, logprob_results, diagnostics_results
    
    
    def generate_with_replacement_sampling(
        self,
        total_sequences: int,
        dataset_name: str,
        split: str = "train",
        max_prompts_pool: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[BatchedSequences, LogprobResults, DiagnosticsResults]:
        """
        Generate sequences by sampling prompts WITH REPLACEMENT from dataset.
        
        This achieves true independent sampling from D(p)π(t|p) where:
        - D(p): Uniform distribution over prompts (with replacement)  
        - π(t|p): Model's conditional distribution over responses given prompt
        
        This is useful for generating large numbers of sequences (e.g., 32K) from a 
        smaller pool of prompts (e.g., 4K), where each sequence represents an 
        independent draw from the joint distribution.
        
        Args:
            total_sequences: Number of sequences to generate (e.g., 32768)
            dataset_name: Dataset to sample prompts from (e.g., 'gsm8k_r1_template')
            split: Dataset split ('train', 'test', etc.)
            max_prompts_pool: Limit prompt pool size (None = use all available)
            seed: Random seed for prompt sampling reproducibility
            **kwargs: Additional arguments passed to generate_with_logprobs()
                     (G, gen_batch_size, tf_batch_size, with_grad, compute_rb, etc.)
        
        Returns:
            Same as generate_with_logprobs(): (BatchedSequences, LogprobResults, DiagnosticsResults)
            
        Example:
            # Generate 32K sequences from GSM8K prompts with replacement
            sequences, logprobs, diagnostics = processor.generate_with_replacement_sampling(
                total_sequences=32768,
                dataset_name="gsm8k_r1_template",
                split="train", 
                G=1,  # One generation per sampled prompt
                seed=42,
                compute_rb=True
            )
        """
        import numpy as np
        from collections import Counter
        
        print(f"🎲 Loading prompt pool from {dataset_name}:{split}")
        
        # 1. Load full prompt pool from dataset
        prompts, examples = self.sample_prompts(dataset_name, split, num_prompts=None, seed=None)
        
        if max_prompts_pool and len(prompts) > max_prompts_pool:
            prompts = prompts[:max_prompts_pool]
            if examples:
                examples = examples[:max_prompts_pool]
        
        pool_size = len(prompts)
        print(f"✅ Loaded {pool_size:,} prompts from dataset")
        
        # 2. Sample prompt indices with replacement
        print(f"🎯 Sampling {total_sequences:,} prompt indices with replacement")
        
        if seed is not None:
            np.random.seed(seed)
        
        sampled_indices = np.random.choice(pool_size, size=total_sequences, replace=True)
        unique_prompts_used = len(set(sampled_indices))
        
        print(f"✅ Sampled indices: {total_sequences:,} total, {unique_prompts_used:,} unique ({unique_prompts_used/pool_size:.1%} of pool)")
        
        # Show sampling statistics
        index_counts = Counter(sampled_indices)
        max_reuse = max(index_counts.values())
        print(f"📈 Max reuse of single prompt: {max_reuse}x")
        
        # 3. Create explicit prompt list
        sampled_prompts = [prompts[idx] for idx in sampled_indices]
        sampled_examples = [examples[idx] for idx in sampled_indices] if examples else None
        
        print(f"📝 Created {len(sampled_prompts):,} explicit prompts")
        
        # 4. Use existing generate_with_logprobs with explicit prompts
        # Force G=1 for true independence (one generation per sampled prompt)
        kwargs_copy = kwargs.copy()
        kwargs_copy['G'] = 1
        
        print("🔄 Generating sequences using explicit sampled prompts...")
        
        # Temporarily store examples for reward computation
        if sampled_examples:
            original_examples = getattr(self, '_temp_examples', None)
            self._temp_examples = sampled_examples
        
        try:
            sequences, logprob_results, diagnostics_results = self.generate_with_logprobs(
                prompts=sampled_prompts,
                dataset_name=None,  # Don't sample from dataset
                split=None,
                num_prompts=None,   # Ignored when prompts provided
                seed=None,          # Don't reseed in processor  
                **kwargs_copy
            )
        finally:
            # Restore original examples if they existed
            if sampled_examples:
                if original_examples is not None:
                    self._temp_examples = original_examples
                elif hasattr(self, '_temp_examples'):
                    delattr(self, '_temp_examples')
        
        print(f"✅ Generated {total_sequences:,} sequences with replacement sampling")
        
        return sequences, logprob_results, diagnostics_results
    
    
    # === NEW: helper for RB entropies with top-p (vectorized over timesteps) ===
    def _rb_entropies_top_p(
        self,
        gen_logits: torch.Tensor,   # [T, V] logits aligned to next-token positions
        top_p: float,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute per-step RB entropies H(q) with the SAME sampling policy you used:
          - temperature scaling
          - top-p truncation (if top_p < 1.0), else full softmax.

        Returns:
            rb_H: torch.Tensor [T] with per-step entropies (nats).
        """
        # Temperature
        a = gen_logits / max(temperature, 1e-8)  # [T, V]

        if top_p >= 1.0:
            # Full softmax RB: H = logsumexp(a) - sum softmax(a) * a
            Z_full = torch.logsumexp(a, dim=-1)                    # [T]
            p_full = torch.softmax(a, dim=-1)                      # [T, V]
            H = Z_full - (p_full * a).sum(dim=-1)                  # [T]
            return H

        # Probabilities for ranking only (no need to keep them for entropy)
        p = torch.softmax(a, dim=-1)                               # [T, V]
        # Sort descending
        p_sorted, idx_sorted = p.sort(dim=-1, descending=True)     # [T, V], [T, V]
        cumsum = p_sorted.cumsum(dim=-1)                           # [T, V]

        # Keep minimal set whose previous cumulative mass <= top_p
        # This keeps the "threshold token" that crosses p as well.
        keep_sorted = (cumsum - p_sorted) <= top_p                 # [T, V]
        # Safety: ensure at least one token kept
        keep_sorted[..., 0] = True

        # Scatter keep mask back to vocab order
        T, V = p.shape
        keep = torch.zeros_like(p, dtype=torch.bool)               # [T, V]
        keep.scatter_(dim=-1, index=idx_sorted, src=keep_sorted)

        # Masked logits: outside S -> -inf
        a_masked = a.masked_fill(~keep, float('-inf'))             # [T, V]

        # Renormalized (truncated) log-partition and probs on S
        Z_S = torch.logsumexp(a_masked, dim=-1)                    # [T]
        # Softmax with -inf outside S renormalizes on S
        q = torch.softmax(a_masked, dim=-1)                        # [T, V]

        # RB entropy on truncated distribution q
        H = Z_S - (q * a).sum(dim=-1)                              # [T]
        return H





class DistributionDiagnostics:
    """
    Compute token-distribution diagnostics under the SAME sampling policy used for generation:
    temperature + (optionally) top-p (truncated and renormalized).

    Used primarily for finding potential control variates for variance reduction in entropy estimation.
    """
    def __init__(self, *, top_p: float = 1.0, temperature: float = 1.0, eos_token_id: Optional[int] = None):
        self.top_p = float(top_p)
        self.temperature = float(max(temperature, 1e-8))
        self.eos_token_id = eos_token_id

    @torch.no_grad()
    def compute_from_logits(
        self,
        gen_logits: torch.Tensor,   # [T, V], logits aligned to next-token positions
        realized_tokens: torch.Tensor,  # [T], actual sampled token ids (for naive surprisal)
    ) -> DiagnosticsPack:
        """
        Returns per-step diagnostics and sequence aggregates (numpy), computed on the fly.
        """
        device = gen_logits.device
        T, V = gen_logits.shape

        # temperature scaling
        a = gen_logits / self.temperature  # [T, V]

        # full-softmax objects useful for head/tail accounting even if top_p<1
        Z_full = torch.logsumexp(a, dim=-1)             # [T]
        p_full = torch.softmax(a, dim=-1)               # [T, V]

        # nucleus selection
        if self.top_p < 1.0:
            p_sorted, idx_sorted = p_full.sort(dim=-1, descending=True)        # [T, V]
            cumsum = p_sorted.cumsum(dim=-1)
            keep_sorted = (cumsum - p_sorted) <= self.top_p
            keep_sorted[..., 0] = True
            keep = torch.zeros_like(p_full, dtype=torch.bool)
            keep.scatter_(dim=-1, index=idx_sorted, src=keep_sorted)
            a_masked = a.masked_fill(~keep, float('-inf'))                     # -inf outside S
            Z_S = torch.logsumexp(a_masked, dim=-1)                            # [T]
            q = torch.softmax(a_masked, dim=-1)                                # [T, V] on S
            s = (p_full * keep).sum(dim=-1)                                    # [T]
        else:
            # full vocabulary
            keep = torch.ones_like(p_full, dtype=torch.bool)
            a_masked = a
            Z_S = Z_full
            q = p_full
            s = torch.ones_like(Z_full)

        eps = (1.0 - s).clamp_min(0.0)                                         # tail mass
        # RB entropy H(q) = Z_S - sum q * a  (stable)
        H = Z_S - (q * a).sum(dim=-1)                                          # [T]

        # two-point entropy of [s, eps]
        # H_2pt = -s log s - eps log eps, treating 0 log 0 = 0
        def _safe_log(x): return torch.log(x.clamp_min(1e-38))
        H_two = -(s * _safe_log(s) + eps * _safe_log(eps))

        # top-1 prob and margin
        # NOTE: compute top1 from q (the actual sampling distribution)
        q_sorted, _ = q.sort(dim=-1, descending=True)
        top1 = q_sorted[..., 0]
        if q_sorted.shape[-1] > 1:
            top2 = q_sorted[..., 1]
        else:
            top2 = torch.zeros_like(top1)
        # margin in logit space: a_{(1)} - a_{(2)} (use masked a)
        a_sorted, _ = a_masked.sort(dim=-1, descending=True)
        a1 = a_sorted[..., 0]
        if a_sorted.shape[-1] > 1:
            a2 = a_sorted[..., 1]
        else:
            a2 = torch.full_like(a1, -float('inf'))
        margin = (a1 - a2).masked_fill(~torch.isfinite(a2), 0.0)

        # collision, Rényi-2, effective support
        collision = (q * q).sum(dim=-1)             # sum q^2
        renyi2 = -torch.log(collision.clamp_min(1e-38))
        eff_support = torch.exp(H)                  # exp(H)

        # moments of logits under q
        Ea = (q * a).sum(dim=-1)
        Ea2 = (q * (a * a)).sum(dim=-1)
        var_a = (Ea2 - Ea * Ea).clamp_min(0.0)

        # EOS probability under q (if provided)
        eos_prob = None
        if self.eos_token_id is not None and 0 <= self.eos_token_id < V:
            eos_prob = q[..., self.eos_token_id]

        # naive per-step surprisal for realized tokens (for comparison)
        log_probs = torch.log_softmax(gen_logits, dim=-1)
        token_lp = log_probs.gather(1, realized_tokens.view(-1, 1)).squeeze(1)  # [T]
        naive_surprisal = (-token_lp).float()                                   # [T]

        # convert to numpy
        to_np = lambda t: t.detach().float().cpu().numpy()
        step = TokenStepDiagnostics(
            rb_entropy=to_np(H),
            head_mass=to_np(s),
            tail_mass=to_np(eps),
            two_point_entropy=to_np(H_two),
            top1_prob=to_np(top1),
            margin=to_np(margin),
            collision=to_np(collision),
            renyi2=to_np(renyi2),
            eff_support=to_np(eff_support),
            logit_mean=to_np(Ea),
            logit_var=to_np(var_a),
            eos_prob=(to_np(eos_prob) if eos_prob is not None else None),
        )

        # aggregates
        T_int = int(T)
        K = min(16, T_int)  # early-phase window (tunable)
        rb_sum = float(H.sum().item())
        rb_mean = float(H.mean().item())
        rb_max = float(H.max().item())
        rb_min = float(H.min().item())
        early_mean = float(H[:K].mean().item()) if T_int >= 1 else 0.0
        late_mean = float(H[K:].mean().item()) if T_int > K else early_mean

        naive_sum = float(naive_surprisal.sum().item())
        naive_mean = float(naive_surprisal.mean().item()) if T_int > 0 else 0.0

        seq = SequenceDiagnostics(
            T=T_int,
            rb_entropy_sum=rb_sum,
            rb_entropy_mean=rb_mean,
            rb_entropy_max=rb_max,
            rb_entropy_min=rb_min,
            early_rb_entropy_mean=early_mean,
            late_rb_entropy_mean=late_mean,
            naive_surprisal_sum=naive_sum,
            naive_surprisal_mean=naive_mean,
            margin_mean=float(margin.mean().item()),
            margin_sum=float(margin.sum().item()),
            top1_prob_mean=float(top1.mean().item()),
            collision_mean=float(collision.mean().item()),
            renyi2_mean=float(renyi2.mean().item()),
            eff_support_mean=float(eff_support.mean().item()),
            eos_prob_mean=(float(eos_prob.mean().item()) if eos_prob is not None and T_int>0 else None),
        )
        return DiagnosticsPack(step=step, seq=seq)