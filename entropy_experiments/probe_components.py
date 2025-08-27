"""
Probe Components

Core computational components for the offline entropy probe.
Handles the main gradient computation pipeline including:

- Batch sampling from datasets
- Scalar probe loss construction (L_X and L_Y)  
- torch.autograd.grad extraction with microbatching
- Block-level U-statistic computation
- Memory-efficient "recompute X then Y" strategy

Based on Section III and VII of offline_entropy_probe_strategy.txt.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import nullcontext
import logging
from collections import defaultdict
import time
import math


class ProbeComponents:
    """
    Core computational components for entropy probe analysis.
    
    This class handles the main gradient computation pipeline while avoiding
    storing parameter-sized vectors through clever use of scalar probe losses.
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], logger: logging.Logger):
        self.model = model
        self.config = config  
        self.logger = logger
        self.device = next(model.parameters()).device
        
        # AMP settings to match training
        self.use_amp = config['memory_config']['amp']
        self.amp_dtype = getattr(torch, config['memory_config']['dtype'])
        
        # Sampling parameters
        self.B = config['batch_config']['B']  # prompts per batch
        self.G = config['batch_config']['G']  # responses per prompt
        self.microbatch_size = config['memory_config']['microbatch_size']
        
        # Mode configuration
        self.mode = config['probe_config']['mode']  # "exact" or "blocks"
        self.M = config['probe_config'].get('M', None)  # number of blocks if mode="blocks"
        
        self.logger.info(f"ProbeComponents initialized: mode={self.mode}, B={self.B}, G={self.G}")
        
    def sample_batch(self, B: int, G: int, indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Sample batch of B prompts, each with G responses.
        
        CRITICAL CHANGE: This no longer computes log-probs here (P0 fix).
        Instead, it returns tokenized sequences that will be used to compute
        log-probs with gradients enabled during the probe passes.
        
        Args:
            B: Number of prompts
            G: Number of responses per prompt
            indices: Optional list of specific dataset indices to use (for deterministic multi-GPU sampling)
            
        Returns:
            Dictionary containing:
            - sequences: Tensor[B, G, max_len] tokenized prompt+response sequences
            - prompt_lens: List[B] of prompt lengths for each prompt
            - advantages: Tensor[B, G] of advantages A_{n,g}
            - max_lengths: List[B] of L_max(p_n) for each prompt
            - attention_masks: Tensor[B, G, max_len] attention masks
            - prompt_ids: List[B] of original dataset indices for each prompt
        """
        self.logger.info(f"Sampling {B} prompts with {G} responses each")
        
        # Step 1: Load dataset and sample prompts
        from rlp_datasets import DATASET_REGISTRY
        dataset = DATASET_REGISTRY[self.config['batch_config']['dataset_name']]
        ds_examples = dataset(self.config['batch_config']['split'])
        
        # Sample B prompts (deterministically if indices provided, randomly otherwise)
        if indices is not None:
            # Use provided indices (for deterministic multi-GPU sampling)
            sampled_indices = indices[:B]  # Take exactly B indices
            if len(sampled_indices) < B:
                # Repeat if not enough indices provided
                sampled_indices = (sampled_indices * ((B // len(sampled_indices)) + 1))[:B]
        else:
            # Random sampling (legacy behavior)
            import random
            dataset_indices = list(range(len(ds_examples)))
            sampled_indices = random.sample(dataset_indices, min(B, len(dataset_indices)))
            if len(sampled_indices) < B:
                # Repeat if dataset is too small
                sampled_indices = (sampled_indices * ((B // len(sampled_indices)) + 1))[:B]
            
        sampled_examples = [ds_examples[i] for i in sampled_indices]
        prompts = [ex.question for ex in sampled_examples]
        prompt_ids = sampled_indices[:B]  # Keep track of original dataset indices
        
        # Step 2: Generate G responses per prompt
        from transformers import GenerationConfig
        
        # Setup generation config
        gen_cfg = GenerationConfig(
            max_new_tokens=self.config['generation']['max_new_tokens'],
            temperature=self.config['generation']['temperature'],
            top_p=self.config['generation']['top_p'],
            do_sample=self.config['generation']['do_sample'],
            num_return_sequences=G,
            pad_token_id=self.config['generation']['pad_token_id'],
            return_dict_in_generate=True,
            output_logits=False,
            output_attentions=False
        )
        
        # Setup tokenizer (we need access to it)
        if not hasattr(self, '_tokenizer'):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B", 
                trust_remote_code=True
            )
            self._tokenizer.padding_side = "left"
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        # Generate responses and collect sequences for later teacher forcing
        # OPTIMIZED: Use batched rollout generation instead of sequential prompts
        all_sequences = []
        all_prompt_lens = []
        all_advantages = []
        max_lengths = []
        all_attention_masks = []
        
        # Rollout batch size for parallel generation (optimized for 40GB A100)
        # With 27% usage = ~11GB used, we have 29GB free for larger batches
        rollout_batch_size = self.config.get('batch_config', {}).get('rollout_batch_size', 8)
        self.logger.info(f"Using rollout_batch_size={rollout_batch_size} for batched generation")
        
        # Process prompts in batches for parallel generation
        for batch_start in range(0, len(prompts), rollout_batch_size):
            batch_end = min(batch_start + rollout_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_examples = sampled_examples[batch_start:batch_end] 
            batch_prompt_ids = prompt_ids[batch_start:batch_end]
            
            self.logger.debug(f"Generating responses for prompts {batch_start}-{batch_end-1} ({len(batch_prompts)} prompts)")
            
            with torch.inference_mode():
                # ✅ FIX: Use LEFT padding like dr_grpo.py to avoid generating from pad tokens
                original_padding_side = self._tokenizer.padding_side
                self._tokenizer.padding_side = "left"
                
                # Tokenize all prompts in batch with LEFT padding
                batch_prompt_enc = self._tokenizer(batch_prompts, padding=True, return_tensors="pt").to(self.device)
                batch_prompt_lens = (batch_prompt_enc.attention_mask).sum(dim=1).tolist()  # Get actual prompt lengths
                
                # Restore original padding side
                self._tokenizer.padding_side = original_padding_side
                
                # Generate G responses for each prompt simultaneously
                # This generates len(batch_prompts) * G sequences in one call
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    gen_out = self.model.generate(
                        **batch_prompt_enc,
                        generation_config=gen_cfg,
                        logits_processor=self._get_stop_processor(),
                        return_dict_in_generate=True
                    )
                    
                # Extract sequences: [batch_size * G, total_len]
                all_gen_sequences = gen_out.sequences  
                
                # Reshape to [batch_size, G, total_len]
                batch_size = len(batch_prompts)
                total_len = all_gen_sequences.shape[1]
                sequences_reshaped = all_gen_sequences.view(batch_size, G, total_len)
                
                # With LEFT padding, find where generation actually starts
                # All sequences have same shape [G, total_len], generation starts at same position
                max_prompt_len = max(batch_prompt_lens)
                
                # Process each prompt in the batch  
                for local_b, (prompt, example, prompt_id, prompt_len) in enumerate(zip(
                    batch_prompts, batch_examples, batch_prompt_ids, batch_prompt_lens
                )):
                    sequences = sequences_reshaped[local_b]  # [G, total_len]
                    
                    # Decode generated text for advantage computation
                    responses = []
                    for g in range(G):
                        # ✅ FIX: With LEFT padding, generation starts at max_prompt_len for ALL sequences
                        gen_ids = sequences[g, max_prompt_len:]
                        # Trim at stop tag if present
                        gen_ids = self._trim_at_stop_tag(gen_ids)
                        response_text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
                        responses.append(response_text)
                    
                    # Store sequences and metadata for probe passes
                    all_sequences.append(sequences)  # [G, total_len]
                    # ✅ FIX: Store max_prompt_len (where generation starts) not actual prompt_len
                    all_prompt_lens.append(max_prompt_len)
                    
                    # Create attention masks
                    attention_mask = (sequences != self._tokenizer.pad_token_id).long()
                    all_attention_masks.append(attention_mask)
                    
                    # Compute real rewards using the same reward function as training
                    rewards = self._compute_rewards(prompt_id, responses, example)
                    
                    # Compute advantages following DRGRPO: A = R - mean(R) per prompt
                    advantages = rewards - rewards.mean()  # Center advantages (GRPO style)
                    all_advantages.append(advantages)
                    
                    # Track max response length for probe loss normalization
                    response_lengths = [len(self._tokenizer.encode(resp, add_special_tokens=False)) for resp in responses]
                    max_len = max(response_lengths) if response_lengths else 200
                    max_lengths.append(max_len)
        
        # Pad sequences to same length and stack
        max_seq_len = max(seq.shape[1] for seq in all_sequences)
        padded_sequences = []
        padded_masks = []
        
        for b in range(B):
            seq = all_sequences[b]  # [G, seq_len]
            mask = all_attention_masks[b]  # [G, seq_len]
            
            # Pad to max length
            if seq.shape[1] < max_seq_len:
                pad_len = max_seq_len - seq.shape[1]
                seq = F.pad(seq, (0, pad_len), value=self._tokenizer.pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
                
            padded_sequences.append(seq)
            padded_masks.append(mask)
        
        # Stack everything
        sequences = torch.stack(padded_sequences, dim=0)  # [B, G, max_len]
        attention_masks = torch.stack(padded_masks, dim=0)  # [B, G, max_len]
        advantages = torch.stack(all_advantages, dim=0)  # [B, G]
        
        batch_data = {
            'sequences': sequences,  # [B, G, max_len] - for teacher forcing with gradients
            'prompt_lens': all_prompt_lens,  # [B] - prompt lengths
            'advantages': advantages,  # [B, G] - advantages A_{n,g}
            'max_lengths': max_lengths,  # [B] - L_max per prompt
            'attention_masks': attention_masks,  # [B, G, max_len]
            'prompt_ids': prompt_ids,  # [B] - original dataset indices for each prompt
            'num_prompts': B,
            'num_responses_per_prompt': G
        }
        
        self.logger.info(f"Successfully sampled batch: {B} prompts × {G} responses")
        return batch_data
        
    # =====================================================================
    # MICROBATCHED HELPER METHODS - Added for memory optimization
    # =====================================================================
    
    def _teacher_force_logprobs(self, prompt_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute sequence log-probabilities with gradients enabled via teacher forcing.
        
        CRITICAL: This ensures S has requires_grad=True for gradient computation.
        
        Args:
            prompt_batch: Batch data (can be microbatch or single unit)
            
        Returns:
            Dictionary with:
                - 'S': [batch_size, G] sequence log-probabilities with gradients
                - 'sequences': sequences used
                - 'advantages': advantages 
                - 'max_lengths': max lengths per prompt
        """
        sequences = prompt_batch['sequences']  # [batch_size, G, max_len]  
        prompt_lens = prompt_batch['prompt_lens']  # [batch_size]
        attention_masks = prompt_batch['attention_masks']  # [batch_size, G, max_len]
        advantages = prompt_batch['advantages']  # [batch_size, G]
        max_lengths = prompt_batch['max_lengths']  # [batch_size]
        
        batch_size, G, max_len = sequences.shape
        
        # Ensure model is in training mode for gradients
        was_training = self.model.training
        self.model.train()
        
        try:
            # Compute log-probabilities for all sequences in batch
            S_batch = []  # [batch_size, G]
            
            for b in range(batch_size):
                seq_logprobs = self._compute_sequence_logprobs(sequences[b], prompt_lens[b])
                S_batch.append(seq_logprobs)
                
            # Stack into tensor with gradients enabled
            S = torch.stack(S_batch, dim=0)  # [batch_size, G]
            
        finally:
            # Restore training state
            self.model.train(was_training)
        
        # 🔍 DEBUG: Check S tensor gradient requirements
        self.logger.info(f"🔍 [S-TENSOR] S.requires_grad={S.requires_grad}, S.grad_fn={S.grad_fn}")
        
        return {
            'S': S,  # [batch_size, G] with requires_grad=True
            'sequences': sequences,
            'advantages': advantages, 
            'max_lengths': max_lengths,
            'attention_masks': attention_masks
        }
        
        
    def _get_stop_processor(self):
        """Get stopping processor for generation."""
        from evals.utils_io import StopAfterAnswer
        from transformers import LogitsProcessorList
        return LogitsProcessorList([StopAfterAnswer(self._tokenizer)])
        
    def _compute_sequence_logprobs(self, sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """
        Compute sequence log probabilities via teacher forcing with G-microbatching.
        
        Args:
            sequences: [G, total_len] generated sequences 
            prompt_len: Length of the prompt portion
            
        Returns:
            [G] sequence log probabilities
        """
        G, total_len = sequences.shape
        
        # ✅ G-MICROBATCHING: Process sequences in smaller chunks to reduce memory
        teacher_force_microbatch_size = self.config.get('memory_config', {}).get('teacher_force_microbatch_size', 2)
        self.logger.debug(f"Using teacher_force_microbatch_size={teacher_force_microbatch_size} for gradient computation")
        
        all_token_log_probs = []
        
        # Process G sequences in microbatches
        for g_start in range(0, G, teacher_force_microbatch_size):
            g_end = min(g_start + teacher_force_microbatch_size, G)
            micro_sequences = sequences[g_start:g_end]  # [g_micro, total_len]
            
            # Teacher forcing forward pass on microbatch
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(micro_sequences).logits  # [g_micro, total_len, vocab_size]
                
            # Apply temperature if configured
            temp = self.config['generation'].get('temperature', 1.0)
            if temp != 1.0:
                logits = logits / temp
                
            # Convert to log probabilities in float32 for numerical stability
            log_probs = F.log_softmax(logits.float(), dim=-1)  # [g_micro, total_len, vocab_size]
            
            # Extract log probs of actual tokens (causal shift)
            target_ids = micro_sequences[:, 1:].unsqueeze(-1)  # [g_micro, total_len-1, 1]
            token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [g_micro, total_len-1]
            
            all_token_log_probs.append(token_log_probs)
            
            # Free intermediate tensors to reduce memory pressure
            del logits, log_probs, target_ids, token_log_probs
            torch.cuda.empty_cache()
        
        # Concatenate all microbatch results
        token_log_probs = torch.cat(all_token_log_probs, dim=0)  # [G, total_len-1]
        
        # Sum over generation tokens only (excluding prompt)
        gen_start = prompt_len - 1  # -1 for causal shift
        gen_token_log_probs = token_log_probs[:, gen_start:]  # [G, gen_len]
        
        # Create mask for real tokens (not padding)
        gen_ids = sequences[:, prompt_len:]  # [G, gen_len] 
        if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
            gen_mask = (gen_ids != self._tokenizer.pad_token_id).float()
        else:
            gen_mask = torch.ones_like(gen_ids).float()
            
        # Apply mask and sum
        if gen_mask.shape != gen_token_log_probs.shape:
            # Handle shape mismatch
            min_len = min(gen_mask.shape[1], gen_token_log_probs.shape[1])
            gen_mask = gen_mask[:, :min_len]
            gen_token_log_probs = gen_token_log_probs[:, :min_len]
            
        seq_log_probs = (gen_token_log_probs * gen_mask).sum(dim=1)  # [G]
        
        return seq_log_probs
        
    def _trim_at_stop_tag(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Trim token sequence at first occurrence of stop tag."""
        from evals.utils_io import TAG_STOP
        
        # Get stop tag token ids
        stop_ids = self._tokenizer.encode(TAG_STOP, add_special_tokens=False)
        if len(stop_ids) == 0:
            return token_ids
            
        # Find first occurrence of stop sequence
        token_ids_list = token_ids.tolist()
        for i in range(len(token_ids_list) - len(stop_ids) + 1):
            if token_ids_list[i:i+len(stop_ids)] == stop_ids:
                # Include the stop tag itself
                return token_ids[:i+len(stop_ids)]
                
        return token_ids  # No stop tag found
        
    def _compute_rewards(self, prompt_id: int, responses: List[str], example: Any) -> torch.Tensor:
        """
        Compute rewards for responses using the same reward function as training.
        
        This uses the tag_pref reward function which checks for proper <answer> formatting
        and mathematical correctness.
        
        Args:
            prompt_id: Original dataset index for this prompt
            responses: List of G generated responses
            example: Dataset example object containing gold answer
            
        Returns:
            [G] tensor of rewards
        """
        from rl_training.rewards.tag_pref import reward_fn, PROMPT2GOLD
        
        try:
            # Get gold answer directly from example object
            gold_answer = example.answer
            
            # Set up the PROMPT2GOLD mapping temporarily
            original_mapping = PROMPT2GOLD.copy()
            PROMPT2GOLD[prompt_id] = gold_answer
            
            # Compute rewards
            rewards = reward_fn(prompt_id, responses)
            
            # Restore original mapping
            PROMPT2GOLD.clear()
            PROMPT2GOLD.update(original_mapping)
            
            return rewards.to(self.device)
                
        except Exception as e:
            self.logger.warning(f"Error computing rewards for prompt_id {prompt_id}: {e}, using zero rewards")
            return torch.zeros(len(responses), device=self.device)
        
                
    def _get_learning_rate(self, optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        """Extract learning rate, prioritizing config over optimizer."""
        # Check config first - this allows manual override
        if 'learning_rate' in self.config:
            lr = self.config['learning_rate']
            self.logger.info(f"🔍 [LR] Using learning rate from config: {lr:.2e}")
            return lr
        elif optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"🔍 [LR] Extracted learning rate from optimizer: {lr:.2e}")
            return lr
        elif hasattr(self.model, 'optimizer'):
            lr = self.model.optimizer.param_groups[0]['lr']
            self.logger.info(f"🔍 [LR] Extracted learning rate from model.optimizer: {lr:.2e}")
            return lr
        else:
            # Final fallback
            lr = 1e-6
            self.logger.warning(f"🔍 [LR] Using default fallback learning rate: {lr:.2e}")
            return lr
    
    # ========================================================================
    # STAGE 1: Mixed E/U Batch Probe - Buffer Utilities
    # ========================================================================
    
    def zeros_like_params(self, dtype: torch.dtype = torch.float32, device: str = "cpu") -> Dict[int, torch.Tensor]:
        """
        Create zero buffers matching model parameters.
        
        Args:
            dtype: Data type for buffers (default float32 for stability)
            device: Device for buffers ("cpu" or "cuda")
            
        Returns:
            Dict mapping param id to zero tensor with same shape as parameter
        """
        param_buffers = {}
        for param in self.model.parameters():
            if param.requires_grad:
                param_buffers[id(param)] = torch.zeros_like(param, dtype=dtype, device=device)
        return param_buffers
    
    def add_into_param_buffer(self, buf: Dict[int, torch.Tensor], scale: float = 1.0) -> None:
        """
        Add current param.grad into buffer with optional scaling.
        
        Args:
            buf: Parameter buffer dict from zeros_like_params()
            scale: Scaling factor for gradients
        """
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param_id = id(param)
                if param_id in buf:
                    # Convert gradient to buffer's dtype and device
                    grad = param.grad.detach()
                    if buf[param_id].device != grad.device or buf[param_id].dtype != grad.dtype:
                        grad = grad.to(buf[param_id].device, dtype=buf[param_id].dtype)
                    buf[param_id].add_(grad, alpha=scale)
    
    def dot_param_buffers(self, buf_a: Dict[int, torch.Tensor], buf_b: Dict[int, torch.Tensor]) -> float:
        """
        Compute dot product between two parameter buffers.
        
        Args:
            buf_a: First parameter buffer
            buf_b: Second parameter buffer
            
        Returns:
            Scalar dot product as float
        """
        total_dot = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                param_id = id(param)
                if param_id in buf_a and param_id in buf_b:
                    dot_contrib = (buf_a[param_id] * buf_b[param_id]).sum().item()
                    total_dot += dot_contrib
        return total_dot
    
    # ========================================================================
    # STAGE 1: Mixed E/U Batch Probe - Loss Builders
    # ========================================================================
    
    def build_LX_from_S(self, S_dict: Dict[str, Any], weighting_mode: str = "dr_grpo") -> torch.Tensor:
        """
        Build X-loss: L_X = mean_over_responses(detach(S_w - S_w_LOO) * S)
        
        Args:
            S_dict: Dict with 'S' [batch_size, G], 'max_lengths' [batch_size], etc.
            weighting_mode: "dr_grpo" (1/L_max) or "per_token_avg" (1/L_eff)
            
        Returns:
            Scalar loss tensor with requires_grad=True
        """
        S = S_dict['S']  # [batch_size, G] - sequence log-probabilities
        max_lengths = S_dict['max_lengths']  # [batch_size] - max length per prompt
        batch_size, G = S.shape
        
        total_loss = 0.0
        total_count = 0
        
        for b in range(batch_size):
            S_b = S[b]  # [G] - responses for prompt b
            L_max_b = max_lengths[b]
            
            # Compute weighted scores S_w based on weighting mode
            with torch.no_grad():  # Weights are θ-independent
                if weighting_mode == "dr_grpo":
                    # S_w = S / L_max (DR-GRPO style)
                    S_w_b = S_b / L_max_b
                elif weighting_mode == "per_token_avg":
                    # S_w = S / L_eff - need generation lengths
                    if 'gen_lengths' in S_dict:
                        gen_lengths_b = S_dict['gen_lengths'][b]  # [G]
                        S_w_b = S_b / gen_lengths_b.clamp(min=1.0)
                    else:
                        # Fallback to dr_grpo if gen_lengths not available
                        S_w_b = S_b / L_max_b
                else:
                    raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
                
                # Compute leave-one-out baselines S_w_LOO
                S_w_LOO_b = torch.zeros_like(S_w_b)
                for g in range(G):
                    # LOO mean excluding response g
                    mask = torch.ones(G, dtype=torch.bool, device=S_w_b.device)
                    mask[g] = False
                    if mask.sum() > 0:
                        S_w_LOO_b[g] = S_w_b[mask].mean()
                    else:
                        S_w_LOO_b[g] = 0.0  # Edge case: G=1
                
                # Detached coefficient
                coeff_b = (S_w_b - S_w_LOO_b).detach()
            
            # L_X for this prompt: mean_g(coeff * S)
            prompt_loss = (coeff_b * S_b).mean()
            total_loss = total_loss + prompt_loss
            total_count += 1
        
        # Return average loss across prompts
        return total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def build_LY_from_S(self, S_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Build Y-loss: L_Y = mean_over_responses((A / L_max) * S)
        
        Args:
            S_dict: Dict with 'S' [batch_size, G], 'advantages' [batch_size, G], 'max_lengths' [batch_size]
            
        Returns:
            Scalar loss tensor with requires_grad=True
        """
        S = S_dict['S']  # [batch_size, G]
        advantages = S_dict['advantages']  # [batch_size, G]
        max_lengths = S_dict['max_lengths']  # [batch_size]
        batch_size, G = S.shape
        
        total_loss = 0.0
        total_count = 0
        
        for b in range(batch_size):
            S_b = S[b]  # [G]
            A_b = advantages[b]  # [G]
            L_max_b = max_lengths[b]  # scalar
            
            # L_Y for this prompt: mean_g((A / L_max) * S)
            weights_b = A_b / L_max_b  # θ-independent weights
            prompt_loss = (weights_b * S_b).mean()
            total_loss = total_loss + prompt_loss
            total_count += 1
        
        # Return average loss across prompts
        return total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def iter_microbatches(self, batch_dict: Dict[str, Any], size: int):
        """
        Iterate over microbatches by slicing along the prompt dimension.
        
        Args:
            batch_dict: Batch dictionary with tensors/lists indexed by prompt
            size: Microbatch size (number of prompts per microbatch)
            
        Yields:
            Microbatch dictionaries with sliced tensors
        """
        # Determine batch size from one of the batch dimensions
        if 'sequences' in batch_dict:
            batch_size = len(batch_dict['sequences'])
        elif 'advantages' in batch_dict and hasattr(batch_dict['advantages'], 'shape'):
            batch_size = batch_dict['advantages'].shape[0]
        elif 'max_lengths' in batch_dict:
            batch_size = len(batch_dict['max_lengths'])
        else:
            raise ValueError("Cannot determine batch size from batch_dict")
        
        # Iterate over microbatches
        for start_idx in range(0, batch_size, size):
            end_idx = min(start_idx + size, batch_size)
            microbatch = {}
            
            # Slice each key in the batch
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    # Slice tensor along first dimension
                    microbatch[key] = value[start_idx:end_idx]
                elif isinstance(value, (list, tuple)):
                    # Slice list/tuple
                    microbatch[key] = value[start_idx:end_idx]
                else:
                    # Keep scalar values as-is
                    microbatch[key] = value
            
            yield microbatch
    
    # ========================================================================
    # STAGE 1: Mixed E/U Batch Probe - Accumulation Methods  
    # ========================================================================
    
    def accumulate_sum_X(self, E_batch: Dict[str, Any], mb_size_prompts: int, weighting_mode: str = "dr_grpo") -> tuple:
        """
        Phase 1: Accumulate ΣX (raw gradients) from evaluation batch E.
        
        Args:
            E_batch: Evaluation batch dictionary
            mb_size_prompts: Microbatch size (number of prompts per microbatch)
            weighting_mode: Weighting mode for X-loss
            
        Returns:
            (sum_X_buf, B_local): Parameter sum buffer and local batch size
        """
        # Get trainable parameters  
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize sum buffer on CPU in fp32
        sum_X_buf = self.zeros_like_params(dtype=torch.float32, device='cpu')
        
        # Count prompts processed
        B_local = 0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.debug(f"Starting X accumulation with {mb_size_prompts} prompts per microbatch")
        
        # Process microbatches
        with no_sync_context():  # Prevent DDP gradient averaging
            for microbatch in self.iter_microbatches(E_batch, mb_size_prompts):
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                
                # Forward pass with teacher forcing
                S_dict = self._teacher_force_logprobs(microbatch)
                
                # Build X-loss with detached LOO coefficient
                L_X_mb = self.build_LX_from_S(S_dict, weighting_mode)
                
                # Backward pass - populates param.grad with raw X gradients  
                L_X_mb.backward()
                
                # Accumulate gradients into sum buffer
                self.add_into_param_buffer(sum_X_buf)
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    B_local += len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    B_local += microbatch['advantages'].shape[0]
                else:
                    B_local += len(microbatch.get('max_lengths', []))
                
                # Clear gradients for next microbatch
                self.model.zero_grad(set_to_none=True)
        
        self.logger.info(f"X accumulation complete: {B_local} prompts processed")
        return sum_X_buf, B_local
    
    def accumulate_sum_Y(self, U_batch: Dict[str, Any], mb_size_prompts: int, 
                        adam_preconditioner) -> tuple:
        """
        Phase 2: Accumulate ΣY (preconditioned gradients) from update batch U.
        
        Args:
            U_batch: Update batch dictionary
            mb_size_prompts: Microbatch size (number of prompts per microbatch)
            adam_preconditioner: Preconditioner to apply after backward()
            
        Returns:
            (sum_Y_buf, B_local): Parameter sum buffer and local batch size
        """
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize sum buffer on CPU in fp32
        sum_Y_buf = self.zeros_like_params(dtype=torch.float32, device='cpu')
        
        # Count prompts processed
        B_local = 0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.debug(f"Starting Y accumulation with {mb_size_prompts} prompts per microbatch")
        
        # Process microbatches
        with no_sync_context():  # Prevent DDP gradient averaging
            for microbatch in self.iter_microbatches(U_batch, mb_size_prompts):
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                
                # Forward pass with teacher forcing
                S_dict = self._teacher_force_logprobs(microbatch)
                
                # Build Y-loss with advantage weighting
                L_Y_mb = self.build_LY_from_S(S_dict)
                
                # Backward pass - populates param.grad with raw ∇J
                L_Y_mb.backward()
                
                # Apply preconditioner in-place: grad ← P(grad) 
                for param in params:
                    if param.grad is not None:
                        preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                        param.grad.copy_(preconditioned_grad)
                
                # Accumulate preconditioned gradients into sum buffer  
                self.add_into_param_buffer(sum_Y_buf)
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    B_local += len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    B_local += microbatch['advantages'].shape[0]
                else:
                    B_local += len(microbatch.get('max_lengths', []))
                
                # Clear gradients for next microbatch
                self.model.zero_grad(set_to_none=True)
        
        self.logger.info(f"Y accumulation complete: {B_local} prompts processed")
        return sum_Y_buf, B_local
    
    # ========================================================================
    # STAGE 2: Variance Computation - Helper Methods
    # ========================================================================
    
    def dot_param_grad_minus_mean_with(self, direction_buf: Dict[int, torch.Tensor], 
                                     mean_buf: Dict[int, torch.Tensor]) -> float:
        """
        Compute dot product of (current param.grad - mean_buf) with direction_buf.
        
        This is used for scalar projections in variance computation:
        proj = (X_u - μ_X) · μ_Y  or  (Y_u - μ_Y) · μ_X
        
        Args:
            direction_buf: Direction buffer (e.g., μ_Y for X variance)
            mean_buf: Mean buffer to subtract (e.g., μ_X for X variance) 
            
        Returns:
            Scalar projection as float
        """
        total_proj = 0.0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param_id = id(param)
                if param_id in direction_buf and param_id in mean_buf:
                    # Convert grad to CPU fp32 for computation
                    grad = param.grad.detach().to('cpu', dtype=torch.float32)
                    
                    # Compute (grad - mean) dot direction
                    diff = grad - mean_buf[param_id]
                    dot_contrib = (diff * direction_buf[param_id]).sum().item()
                    total_proj += dot_contrib
                    
        return total_proj
    
    # ========================================================================
    # STAGE 2: Variance Computation - V_X and V_Y Methods
    # ========================================================================
    
    def compute_VX(self, E_batch: Dict[str, Any], muX_buf: Dict[int, torch.Tensor], 
                   muY_buf: Dict[int, torch.Tensor], mb_size_prompts: int, 
                   weighting_mode: str = "dr_grpo") -> float:
        """
        Compute V_X variance component via per-unit scalar projections.
        
        V_X = [1/(B_E(B_E-1))] Σ_n [(X_n - μ_X) · μ_Y]²
        
        Args:
            E_batch: Evaluation batch (same as used for X accumulation)
            muX_buf: Mean X buffer (I^X from Stage 1)
            muY_buf: Mean Y buffer (I^Y from Stage 1)  
            mb_size_prompts: Microbatch size for processing
            weighting_mode: Weighting mode for X-loss
            
        Returns:
            V_X_local as float (for all-reduce in driver)
        """
        sum_sq_proj_X = 0.0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.debug("Computing V_X via per-unit scalar projections")
        
        # Process each prompt unit individually (not microbatched for variance)
        with no_sync_context():  # Prevent DDP gradient averaging
            for unit in self._iter_units(E_batch):
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                
                # Forward pass with teacher forcing for single unit
                S_dict = self._teacher_force_logprobs(unit)
                
                # Build X-loss for this unit
                L_X_u = self.build_LX_from_S(S_dict, weighting_mode)
                
                # Backward pass - get raw X_u in param.grad
                L_X_u.backward()
                
                # Compute projection: (X_u - μ_X) · μ_Y
                proj = self.dot_param_grad_minus_mean_with(muY_buf, muX_buf)
                
                # Accumulate squared projection
                sum_sq_proj_X += proj * proj
                
                # Clear gradients for next unit
                self.model.zero_grad(set_to_none=True)
        
        # Note: Global B_E and division by B_E(B_E-1) handled in driver
        # This returns the local contribution
        self.logger.info(f"V_X computation complete: sum_sq_proj_X = {sum_sq_proj_X:.6f}")
        return sum_sq_proj_X
    
    def compute_VY(self, U_batch: Dict[str, Any], muX_buf: Dict[int, torch.Tensor],
                   muY_buf: Dict[int, torch.Tensor], mb_size_prompts: int,
                   adam_preconditioner) -> float:
        """
        Compute V_Y variance component via per-unit scalar projections.
        
        V_Y = [1/(B_U(B_U-1))] Σ_p [(Y_p - μ_Y) · μ_X]²
        
        Args:
            U_batch: Update batch (same as used for Y accumulation)  
            muX_buf: Mean X buffer (I^X from Stage 1)
            muY_buf: Mean Y buffer (I^Y from Stage 1)
            mb_size_prompts: Microbatch size for processing
            adam_preconditioner: Preconditioner to apply after backward()
            
        Returns:
            V_Y_local as float (for all-reduce in driver)
        """
        sum_sq_proj_Y = 0.0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.debug("Computing V_Y via per-unit scalar projections")
        
        # Process each prompt unit individually 
        with no_sync_context():  # Prevent DDP gradient averaging
            for unit in self._iter_units(U_batch):
                # Clear gradients
                self.model.zero_grad(set_to_none=True)
                
                # Forward pass with teacher forcing for single unit
                S_dict = self._teacher_force_logprobs(unit)
                
                # Build Y-loss for this unit
                L_Y_u = self.build_LY_from_S(S_dict)
                
                # Backward pass - get raw ∇J in param.grad
                L_Y_u.backward()
                
                # Apply preconditioner in-place: grad ← P(grad)
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                        param.grad.copy_(preconditioned_grad)
                
                # Compute projection: (Y_u - μ_Y) · μ_X  
                proj = self.dot_param_grad_minus_mean_with(muX_buf, muY_buf)
                
                # Accumulate squared projection
                sum_sq_proj_Y += proj * proj
                
                # Clear gradients for next unit
                self.model.zero_grad(set_to_none=True)
        
        # Note: Global B_U and division by B_U(B_U-1) handled in driver
        # This returns the local contribution
        self.logger.info(f"V_Y computation complete: sum_sq_proj_Y = {sum_sq_proj_Y:.6f}")
        return sum_sq_proj_Y
    
    def _iter_units(self, batch_dict: Dict[str, Any]):
        """
        Iterate over individual prompt units (single prompts).
        
        Args:
            batch_dict: Batch dictionary with tensors/lists indexed by prompt
            
        Yields:
            Unit dictionaries with single prompt data
        """
        # Determine batch size
        if 'sequences' in batch_dict:
            batch_size = len(batch_dict['sequences'])
        elif 'advantages' in batch_dict and hasattr(batch_dict['advantages'], 'shape'):
            batch_size = batch_dict['advantages'].shape[0]
        elif 'max_lengths' in batch_dict:
            batch_size = len(batch_dict['max_lengths'])
        else:
            raise ValueError("Cannot determine batch size from batch_dict")
        
        # Iterate over individual units (prompts)
        for idx in range(batch_size):
            unit = {}
            
            # Extract single prompt data
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    # Extract single prompt: [idx] -> [1, ...] to maintain batch dimension
                    unit[key] = value[idx:idx+1]
                elif isinstance(value, (list, tuple)):
                    # Extract single element but keep as list for compatibility
                    unit[key] = [value[idx]]
                else:
                    # Keep scalar values as-is
                    unit[key] = value
            
            yield unit