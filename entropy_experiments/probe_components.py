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
        self.M = config['probe_config']['M']  # number of blocks if mode="blocks"
        
        self.logger.info(f"ProbeComponents initialized: mode={self.mode}, B={self.B}, G={self.G}")
        
    def sample_batch(self, B: int, G: int) -> Dict[str, Any]:
        """
        Sample batch of B prompts, each with G responses.
        
        CRITICAL CHANGE: This no longer computes log-probs here (P0 fix).
        Instead, it returns tokenized sequences that will be used to compute
        log-probs with gradients enabled during the probe passes.
        
        Args:
            B: Number of prompts
            G: Number of responses per prompt
            
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
        
        # Sample B prompts randomly, keeping track of dataset indices
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
        all_sequences = []
        all_prompt_lens = []
        all_advantages = []
        max_lengths = []
        all_attention_masks = []
        
        for b, prompt in enumerate(prompts):
            # Generate G responses for this prompt
            with torch.inference_mode():
                # Tokenize prompt
                prompt_enc = self._tokenizer([prompt], padding=True, return_tensors="pt").to(self.device)
                prompt_len = prompt_enc.input_ids.shape[1]
                
                # Generate
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    gen_out = self.model.generate(
                        **prompt_enc,
                        generation_config=gen_cfg,
                        logits_processor=self._get_stop_processor(),
                        return_dict_in_generate=True
                    )
                    
                # Extract sequences 
                sequences = gen_out.sequences  # [G, total_len]
                
                # Decode generated text for advantage computation
                responses = []
                for g in range(G):
                    gen_ids = sequences[g, prompt_len:]
                    # Trim at stop tag if present
                    gen_ids = self._trim_at_stop_tag(gen_ids)
                    response_text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
                    responses.append(response_text)
                
                # Store sequences and metadata for probe passes
                all_sequences.append(sequences)  # [G, total_len]
                all_prompt_lens.append(prompt_len)
                
                # Create attention masks
                attention_mask = (sequences != self._tokenizer.pad_token_id).long()
                all_attention_masks.append(attention_mask)
                
                # Compute real rewards using the same reward function as training
                prompt_id = prompt_ids[b]  # Get the dataset index for this prompt
                rewards = self._compute_rewards(prompt_id, responses, sampled_examples[b])
                
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
        
    def compute_delta_h1(self, batch_data: Dict[str, Any], 
                        adam_preconditioner: 'AdamPreconditioner',
                        u_statistics: 'UStatisticsCalculator', 
                        distributed_helpers: Optional['DistributedHelpers'] = None) -> Dict[str, Any]:
        """
        Compute first-order entropy change prediction δH₁.
        
        This is the main computation pipeline implementing the U-statistic approach
        described in offline_entropy_probe_strategy.txt.
        
        Args:
            batch_data: Sampled batch from sample_batch()
            adam_preconditioner: Component for applying P^{1/2}
            u_statistics: Component for U-statistic computation
            distributed_helpers: Component for distributed coordination
            
        Returns:
            Dictionary with δH₁ results and diagnostics
        """
        start_time = time.time()
        
        if self.mode == "exact":
            return self._compute_delta_h1_exact(
                batch_data, adam_preconditioner, u_statistics, distributed_helpers
            )
        elif self.mode == "blocks":
            return self._compute_delta_h1_blocks(
                batch_data, adam_preconditioner, u_statistics, distributed_helpers  
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    # =====================================================================
    # MICROBATCHED HELPER METHODS - Added for memory optimization
    # =====================================================================
    
    def _iter_prompt_microbatches(self, batch_data: Dict[str, Any], microbatch_size: int):
        """
        Iterate over microbatches of prompts for memory-efficient processing.
        
        Args:
            batch_data: Full batch data from sample_batch()
            microbatch_size: Number of prompts per microbatch
            
        Yields:
            Dictionary with sliced batch data for this microbatch
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B]
        advantages = batch_data['advantages']  # [B, G] 
        max_lengths = batch_data['max_lengths']  # [B]
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        prompt_ids = batch_data['prompt_ids']  # [B]
        
        B = sequences.shape[0]
        
        for start_idx in range(0, B, microbatch_size):
            end_idx = min(start_idx + microbatch_size, B)
            
            microbatch = {
                'sequences': sequences[start_idx:end_idx],
                'prompt_lens': prompt_lens[start_idx:end_idx],
                'advantages': advantages[start_idx:end_idx],
                'max_lengths': max_lengths[start_idx:end_idx], 
                'attention_masks': attention_masks[start_idx:end_idx],
                'prompt_ids': prompt_ids[start_idx:end_idx],
                'num_prompts': end_idx - start_idx
            }
            
            yield microbatch
    
    def _iter_prompts_as_units(self, batch_data: Dict[str, Any]):
        """
        Iterate over individual prompts as units for diagonal computation.
        
        Args:
            batch_data: Full batch data from sample_batch()
            
        Yields:
            Dictionary with data for single prompt (unit)
        """
        sequences = batch_data['sequences']  # [B, G, max_len]
        prompt_lens = batch_data['prompt_lens']  # [B]
        advantages = batch_data['advantages']  # [B, G]
        max_lengths = batch_data['max_lengths']  # [B]
        attention_masks = batch_data['attention_masks']  # [B, G, max_len]
        prompt_ids = batch_data['prompt_ids']  # [B]
        
        B = sequences.shape[0]
        
        for b in range(B):
            unit = {
                'sequences': sequences[b:b+1],  # Keep batch dimension
                'prompt_lens': prompt_lens[b:b+1],
                'advantages': advantages[b:b+1],
                'max_lengths': max_lengths[b:b+1],
                'attention_masks': attention_masks[b:b+1],
                'prompt_ids': prompt_ids[b:b+1],
                'num_prompts': 1
            }
            
            yield unit, b  # Return unit and original index
    
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
        
        return {
            'S': S,  # [batch_size, G] with requires_grad=True
            'sequences': sequences,
            'advantages': advantages, 
            'max_lengths': max_lengths,
            'attention_masks': attention_masks
        }
    
    def _build_probe_loss_X_from_S(self, S_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Build L_X probe loss from pre-computed sequence log-probabilities.
        
        L_X = mean_b mean_g ((S_{b,g} - LOO_mean_{b,g}) * S_{b,g})
        
        Args:
            S_dict: Dictionary from _teacher_force_logprobs with 'S' key
            
        Returns:
            Scalar probe loss tensor with gradients enabled
        """
        S = S_dict['S']  # [batch_size, G] 
        batch_size, G = S.shape
        
        total_loss = 0.0
        total_count = 0
        
        for b in range(batch_size):
            S_b = S[b]  # [G] sequence log-probs for this prompt
            
            # Compute leave-one-out means
            loo_means = torch.zeros_like(S_b)
            for g in range(G):
                if G > 1:
                    mask = torch.ones(G, dtype=torch.bool, device=self.device)
                    mask[g] = False
                    loo_means[g] = S_b[mask].mean()
                else:
                    loo_means[g] = S_b[g]  # If G=1, use itself
            
            # L_X for this prompt: mean_g((S - S_loo) * S)
            prompt_loss = ((S_b - loo_means) * S_b).mean()
            total_loss = total_loss + prompt_loss  # Keep as tensor
            total_count += 1
            
        return total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _build_probe_loss_Y_from_S(self, S_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Build L_Y probe loss from pre-computed sequence log-probabilities.
        
        L_Y = mean_b mean_g ((A_{b,g} / L_max_b) * S_{b,g})
        
        Args:
            S_dict: Dictionary from _teacher_force_logprobs
            
        Returns:
            Scalar probe loss tensor with gradients enabled
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
            prompt_loss = (A_b / L_max_b * S_b).mean()
            total_loss = total_loss + prompt_loss  # Keep as tensor
            total_count += 1
            
        return total_loss / total_count if total_count > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
            
    def _compute_delta_h1_exact(self, batch_data: Dict[str, Any],
                               adam_preconditioner: 'AdamPreconditioner', 
                               u_statistics: 'UStatisticsCalculator',
                               distributed_helpers: Optional['DistributedHelpers']) -> Dict[str, Any]:
        """
        Compute δH₁ using exact per-prompt U-statistic with microbatched processing.
        
        NEW MEMORY-EFFICIENT IMPLEMENTATION:
        - Uses microbatched gradient computation to bound VRAM by microbatch size
        - Never materializes computation graph spanning all prompts
        - Computes U_cross = (B/(B-1)) * bars_dot - (1/(B(B-1))) * diag_sum
        """
        start_time = time.time()
        self.logger.info("Computing δH₁ using microbatched exact per-prompt method")
        
        # Extract dimensions
        B = batch_data['sequences'].shape[0]
        G = batch_data['sequences'].shape[1]
        
        # Get microbatch size from config (default to 4 if not specified)
        microbatch_size = self.config.get('memory_config', {}).get('microbatch_size', 4)
        self.logger.info(f"Using microbatch_size={microbatch_size} for memory efficiency")
        
        # Get model parameters for gradient accumulation
        params = list(self.model.parameters())
        
        # =========================================================================
        # PHASE 1: X-PASS - Accumulate ΣX via microbatches
        # =========================================================================
        self.logger.debug("Phase 1: X-pass - accumulating ΣX via microbatches")
        
        # Allocate ΣX buffer (param-sized, same dtype/device as params)
        X_sum = {}
        for param in params:
            X_sum[id(param)] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
        # Accumulate X gradients over microbatches
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # 1a. Forward with grad to get S for this microbatch
            S_dict = self._teacher_force_logprobs(microbatch)
            L_X_mb = self._build_probe_loss_X_from_S(S_dict)
            
            # 1b. Backward immediately (microbatch)
            self.model.zero_grad()
            L_X_mb.backward()
            
            # 1c. Apply preconditioner and accumulate into ΣX
            for param in params:
                if param.grad is not None:
                    preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                    X_sum[id(param)].add_(preconditioned_grad)
            
            # Free activations immediately
            self.model.zero_grad()
        
        # =========================================================================
        # PHASE 2: Y-PASS - Accumulate ΣY via microbatches (Option A)
        # =========================================================================
        self.logger.debug("Phase 2: Y-pass - accumulating ΣY via microbatches")
        
        # Allocate ΣY buffer  
        Y_sum = {}
        for param in params:
            Y_sum[id(param)] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
        # Accumulate Y gradients over microbatches
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # Fresh forward with grad
            S_dict = self._teacher_force_logprobs(microbatch)
            L_Y_mb = self._build_probe_loss_Y_from_S(S_dict)
            
            # Get Y gradients via autograd.grad
            y_grads = torch.autograd.grad(L_Y_mb, params, allow_unused=True)
            
            # Apply preconditioner and accumulate into ΣY
            for param, y_grad in zip(params, y_grads):
                if y_grad is not None:
                    preconditioned_grad = adam_preconditioner.apply_preconditioner(y_grad, param)
                    Y_sum[id(param)].add_(preconditioned_grad)
        
        # Compute bars_dot = (ΣX · ΣY) / B²
        bars_dot = 0.0
        for param in params:
            param_id = id(param)
            if param_id in X_sum and param_id in Y_sum:
                bars_dot += (X_sum[param_id] * Y_sum[param_id]).sum().item()
        bars_dot /= (B * B)
        
        # =========================================================================
        # PHASE 3: DIAGONAL LOOP - Compute Σ diag and row-means r_u
        # =========================================================================
        self.logger.debug("Phase 3: Diagonal loop - computing diag terms and row means")
        
        diag_sum = 0.0
        r_values = []
        
        for unit, unit_idx in self._iter_prompts_as_units(batch_data):
            # Get X_u gradients
            self.model.zero_grad()
            S_u = self._teacher_force_logprobs(unit)
            L_X_u = self._build_probe_loss_X_from_S(S_u)
            L_X_u.backward()
            
            # Get Y_u gradients
            y_grads_u = torch.autograd.grad(
                self._build_probe_loss_Y_from_S(S_u), 
                params, 
                allow_unused=True
            )
            
            # Compute required dot products
            dot_Xu_Yu = 0.0      # X̃_u · Ỹ_u 
            dot_Xu_SumY = 0.0    # X̃_u · ΣY
            dot_Yu_SumX = 0.0    # Ỹ_u · ΣX
            
            for param, y_grad in zip(params, y_grads_u):
                param_id = id(param)
                
                # Get preconditioned X_u
                X_u_grad = param.grad
                if X_u_grad is not None:
                    X_u_preconditioned = adam_preconditioner.apply_preconditioner(X_u_grad, param)
                else:
                    continue
                    
                # Get preconditioned Y_u  
                if y_grad is not None:
                    Y_u_preconditioned = adam_preconditioner.apply_preconditioner(y_grad, param)
                else:
                    continue
                
                # Compute dot products
                dot_Xu_Yu += (X_u_preconditioned * Y_u_preconditioned).sum().item()
                
                if param_id in Y_sum:
                    dot_Xu_SumY += (X_u_preconditioned * Y_sum[param_id]).sum().item()
                if param_id in X_sum:
                    dot_Yu_SumX += (Y_u_preconditioned * X_sum[param_id]).sum().item()
            
            # Update diagonal sum
            diag_sum += dot_Xu_Yu
            
            # Compute row mean: r_u = 0.5 * [X̃_u·Ȳ_{-u} + Ỹ_u·X̄_{-u}]
            # where Ȳ_{-u} = (ΣY - Ỹ_u)/(B-1) and X̄_{-u} = (ΣX - X̃_u)/(B-1)
            if B > 1:
                XdotYbar_minus = (dot_Xu_SumY - dot_Xu_Yu) / (B - 1)
                YdotXbar_minus = (dot_Yu_SumX - dot_Xu_Yu) / (B - 1) 
                r_u = 0.5 * (XdotYbar_minus + YdotXbar_minus)
            else:
                r_u = 0.0
                
            r_values.append(r_u)
            
            # Free activations
            self.model.zero_grad()
        
        # Apply distributed reduction if needed
        if distributed_helpers:
            bars_dot, diag_sum = distributed_helpers.reduce_scalars(bars_dot, diag_sum)
            r_values = distributed_helpers.reduce_statistics(torch.tensor(r_values, device=self.device))
            
        # =========================================================================
        # PHASE 4: ASSEMBLY - Compute U_cross and δH₁
        # =========================================================================
        self.logger.debug("Phase 4: Assembly - computing final U_cross and δH₁")
        
        # Compute U-statistic: U_cross = (B/(B-1)) * bars_dot - (1/(B(B-1))) * diag_sum
        if B > 1:
            U_cross = (B / (B - 1)) * bars_dot - (1.0 / (B * (B - 1))) * diag_sum
        else:
            U_cross = 0.0
            
        learning_rate = self._get_learning_rate()
        delta_h1 = -learning_rate * U_cross
        
        # Compute variance estimates from r_values
        variance_results = u_statistics.compute_variance_estimates(
            U_cross=U_cross, U=B, mode="exact", prompt_contributions=r_values
        )
        
        compute_time = time.time() - start_time
        self.logger.info(f"Microbatched exact computation completed in {compute_time:.2f}s")
        
        return {
            "U_cross": U_cross,
            "deltaH1": delta_h1,
            "bars_dot": bars_dot,
            "diag_sum": diag_sum,
            "row_means": r_values,
            "timing": {"compute_time": compute_time},
            "diagnostics": {"B": B, "G": G, "microbatch_size": microbatch_size, "mode": "exact_microbatched"},
            **variance_results
        }
        
    def _compute_delta_h1_blocks(self, batch_data: Dict[str, Any],
                                adam_preconditioner: 'AdamPreconditioner',
                                u_statistics: 'UStatisticsCalculator', 
                                distributed_helpers: Optional['DistributedHelpers']) -> Dict[str, Any]:
        """
        Compute δH₁ using block-level U-statistic with microbatched processing.
        
        NEW IMPLEMENTATION:
        - Treats each microbatch as one block (unit)
        - Uses fresh teacher-forced logprobs with gradients instead of precomputed ones
        - Applies the same X-pass, Y-pass, diagonal pattern but with microbatches as blocks
        - Formula: U_cross_blocks = (M/(M-1)) * bars_dot_blocks - (1/(M*(M-1))) * diag_blocks
        """
        start_time = time.time()
        self.logger.info("Computing δH₁ using microbatched block-level method")
        
        # Extract dimensions
        B = batch_data['sequences'].shape[0] 
        G = batch_data['sequences'].shape[1]
        
        # Get microbatch size from config (default to 4 if not specified)
        microbatch_size = self.config.get('memory_config', {}).get('microbatch_size', 4)
        
        # Calculate number of blocks (microbatches)
        M = (B + microbatch_size - 1) // microbatch_size  # Ceiling division
        self.logger.info(f"Using {M} blocks (microbatches) with max_size={microbatch_size}")
        
        # Get model parameters for gradient accumulation
        params = list(self.model.parameters())
        
        # =========================================================================
        # PHASE 1: X-PASS - Accumulate ΣX_blocks via microbatches  
        # =========================================================================
        self.logger.debug("Phase 1: X-pass - accumulating ΣX_blocks via microbatches")
        
        # Allocate ΣX_blocks buffer
        X_sum_blocks = {}
        for param in params:
            X_sum_blocks[id(param)] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
        # Accumulate X gradients over microbatches (blocks)
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # Fresh forward with grad to get S for this block
            S_dict = self._teacher_force_logprobs(microbatch)
            L_X_block = self._build_probe_loss_X_from_S(S_dict)
            
            # Backward immediately (block-level)
            self.model.zero_grad()
            L_X_block.backward()
            
            # Apply preconditioner and accumulate into ΣX_blocks
            for param in params:
                if param.grad is not None:
                    preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                    X_sum_blocks[id(param)].add_(preconditioned_grad)
            
            # Free activations immediately
            self.model.zero_grad()
        
        # =========================================================================
        # PHASE 2: Y-PASS - Accumulate ΣY_blocks via microbatches
        # =========================================================================
        self.logger.debug("Phase 2: Y-pass - accumulating ΣY_blocks via microbatches")
        
        # Allocate ΣY_blocks buffer
        Y_sum_blocks = {}
        for param in params:
            Y_sum_blocks[id(param)] = torch.zeros_like(param, memory_format=torch.preserve_format)
            
        # Accumulate Y gradients over microbatches (blocks)
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # Fresh forward with grad
            S_dict = self._teacher_force_logprobs(microbatch)
            L_Y_block = self._build_probe_loss_Y_from_S(S_dict)
            
            # Get Y gradients via autograd.grad
            y_grads = torch.autograd.grad(L_Y_block, params, allow_unused=True)
            
            # Apply preconditioner and accumulate into ΣY_blocks
            for param, y_grad in zip(params, y_grads):
                if y_grad is not None:
                    preconditioned_grad = adam_preconditioner.apply_preconditioner(y_grad, param)
                    Y_sum_blocks[id(param)].add_(preconditioned_grad)
        
        # Compute bars_dot_blocks = (ΣX_blocks · ΣY_blocks) / M²
        bars_dot_blocks = 0.0
        for param in params:
            param_id = id(param)
            if param_id in X_sum_blocks and param_id in Y_sum_blocks:
                bars_dot_blocks += (X_sum_blocks[param_id] * Y_sum_blocks[param_id]).sum().item()
        bars_dot_blocks /= (M * M)
        
        # =========================================================================
        # PHASE 3: DIAGONAL LOOP - Compute Σ diag_blocks and row-means r_b
        # =========================================================================
        self.logger.debug("Phase 3: Diagonal loop - computing block diagonal terms and row means")
        
        diag_sum_blocks = 0.0
        r_values_blocks = []
        
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # Get X_b gradients for this block
            self.model.zero_grad()
            S_b = self._teacher_force_logprobs(microbatch)
            L_X_b = self._build_probe_loss_X_from_S(S_b)
            L_X_b.backward()
            
            # Get Y_b gradients for this block
            y_grads_b = torch.autograd.grad(
                self._build_probe_loss_Y_from_S(S_b), 
                params, 
                allow_unused=True
            )
            
            # Compute required dot products for this block
            dot_Xb_Yb = 0.0      # X̃_b · Ỹ_b
            dot_Xb_SumY = 0.0    # X̃_b · ΣY_blocks  
            dot_Yb_SumX = 0.0    # Ỹ_b · ΣX_blocks
            
            for param, y_grad in zip(params, y_grads_b):
                param_id = id(param)
                
                # Get preconditioned X_b
                X_b_grad = param.grad
                if X_b_grad is not None:
                    X_b_preconditioned = adam_preconditioner.apply_preconditioner(X_b_grad, param)
                else:
                    continue
                    
                # Get preconditioned Y_b
                if y_grad is not None:
                    Y_b_preconditioned = adam_preconditioner.apply_preconditioner(y_grad, param)
                else:
                    continue
                
                # Compute dot products
                dot_Xb_Yb += (X_b_preconditioned * Y_b_preconditioned).sum().item()
                
                if param_id in Y_sum_blocks:
                    dot_Xb_SumY += (X_b_preconditioned * Y_sum_blocks[param_id]).sum().item()
                if param_id in X_sum_blocks:
                    dot_Yb_SumX += (Y_b_preconditioned * X_sum_blocks[param_id]).sum().item()
            
            # Update diagonal sum
            diag_sum_blocks += dot_Xb_Yb
            
            # Compute row mean for this block: r_b = 0.5 * [X̃_b·Ȳ_{-b} + Ỹ_b·X̄_{-b}]
            # where Ȳ_{-b} = (ΣY_blocks - Ỹ_b)/(M-1) and X̄_{-b} = (ΣX_blocks - X̃_b)/(M-1)
            if M > 1:
                XdotYbar_minus = (dot_Xb_SumY - dot_Xb_Yb) / (M - 1)
                YdotXbar_minus = (dot_Yb_SumX - dot_Xb_Yb) / (M - 1)
                r_b = 0.5 * (XdotYbar_minus + YdotXbar_minus)
            else:
                r_b = 0.0
                
            r_values_blocks.append(r_b)
            
            # Free activations
            self.model.zero_grad()
        
        # Apply distributed reduction if needed
        if distributed_helpers:
            bars_dot_blocks, diag_sum_blocks = distributed_helpers.reduce_scalars(
                bars_dot_blocks, diag_sum_blocks
            )
            r_values_blocks = distributed_helpers.reduce_statistics(
                torch.tensor(r_values_blocks, device=self.device)
            )
            M = distributed_helpers.reduce_scalar(M)
            
        # =========================================================================
        # PHASE 4: ASSEMBLY - Compute U_cross_blocks and δH₁
        # =========================================================================
        self.logger.debug("Phase 4: Assembly - computing final U_cross_blocks and δH₁")
        
        # Compute block-level U-statistic: 
        # U_cross_blocks = (M/(M-1)) * bars_dot_blocks - (1/(M*(M-1))) * diag_sum_blocks
        if M > 1:
            U_cross_blocks = (M / (M - 1)) * bars_dot_blocks - (1.0 / (M * (M - 1))) * diag_sum_blocks
        else:
            U_cross_blocks = 0.0
            
        learning_rate = self._get_learning_rate()
        delta_h1_blocks = -learning_rate * U_cross_blocks
        
        # Compute variance estimates from block row means
        variance_results = u_statistics.compute_variance_estimates(
            U_cross=U_cross_blocks, U=M, mode="blocks", prompt_contributions=r_values_blocks
        )
        
        compute_time = time.time() - start_time
        self.logger.info(f"Microbatched block computation completed in {compute_time:.2f}s")
        
        return {
            "U_cross": U_cross_blocks,
            "deltaH1": delta_h1_blocks,
            "bars_dot": bars_dot_blocks,
            "diag_sum": diag_sum_blocks,
            "row_means": r_values_blocks,
            "timing": {"compute_time": compute_time},
            "diagnostics": {"B": B, "G": G, "M": M, "microbatch_size": microbatch_size, "mode": "blocks_microbatched"},
            **variance_results
        }
        
    def _compute_prompt_contribution(self, prompt_idx: int,
                                   logprobs: torch.Tensor, advantages: torch.Tensor,
                                   max_lengths: List[int], 
                                   adam_preconditioner: 'AdamPreconditioner') -> float:
        """
        Compute (X̃_n · Ỹ_n) for a single prompt using autograd.
        
        This implements the core "scalar probe loss" strategy to avoid
        storing parameter-sized vectors.
        """
        n = prompt_idx
        
        # Extract data for this prompt
        S_n = logprobs[n, :]  # [G] 
        A_n = advantages[n, :]  # [G]
        L_max_n = max_lengths[n]
        
        # Compute leave-one-out baselines
        S_n_loo = torch.zeros_like(S_n)
        for g in range(len(S_n)):
            mask = torch.ones_like(S_n, dtype=torch.bool)
            mask[g] = False
            S_n_loo[g] = S_n[mask].mean()
            
        # Build scalar probe losses
        # L_X^(n) = (1/G) * Σ_g (S_{n,g} - S̄_{n,-g}) * S_{n,g}
        L_X_n = ((S_n - S_n_loo) * S_n).mean()
        
        # L_Y^(n) = (1/(L_max * G)) * Σ_g A_{n,g} * S_{n,g}  
        L_Y_n = (A_n * S_n).mean() / L_max_n
        
        # Use autograd.grad to extract gradients
        # First backward: L_X -> param.grad (with retain_graph=True)
        self.model.zero_grad()
        L_X_n.backward(retain_graph=True)
        
        # Apply preconditioner to get X̃_n
        x_tilde_params = []
        for param in self.model.parameters():
            if param.grad is not None:
                x_tilde_param = adam_preconditioner.apply_preconditioner(param.grad)
                x_tilde_params.append(x_tilde_param)
            else:
                x_tilde_params.append(None)
                
        # Second gradient extraction: L_Y (without affecting param.grad)
        y_grads = torch.autograd.grad(
            L_Y_n, self.model.parameters(), 
            retain_graph=False, allow_unused=True
        )
        
        # Apply preconditioner to get Ỹ_n and compute dot product
        dot_product = 0.0
        for x_tilde, y_grad in zip(x_tilde_params, y_grads):
            if x_tilde is not None and y_grad is not None:
                y_tilde = adam_preconditioner.apply_preconditioner(y_grad)
                dot_product += (x_tilde * y_tilde).sum().item()
                
        return dot_product
        
    # REMOVED: _build_total_probe_loss
    # This method created massive computation graphs spanning all prompts, which is exactly
    # what the microbatched approach avoids. Replaced by _build_probe_loss_X_from_S and 
    # _build_probe_loss_Y_from_S which work on microbatch-sized chunks.
        
    def _build_prompt_probe_loss(self, sequences: torch.Tensor, prompt_len: int,
                                advantages: torch.Tensor, max_length: int, 
                                attention_mask: torch.Tensor, loss_type: str) -> torch.Tensor:
        """
        Build scalar probe loss for a single prompt.
        
        CRITICAL: Computes fresh log-probs S with gradients enabled.
        
        Args:
            sequences: [G, max_len] sequences for this prompt
            prompt_len: prompt length
            advantages: [G] advantages for this prompt 
            max_length: L_max for this prompt
            attention_mask: [G, max_len] attention mask
            loss_type: "X" or "Y"
            
        Returns:
            Scalar loss tensor with gradients enabled
        """
        G, max_len = sequences.shape
        
        # STEP 1: Compute sequence log-probs with gradients enabled
        # Must use model in training mode and enable autocast/gradients
        was_training = self.model.training
        self.model.train()  # Ensure training mode for gradients
        
        try:
            # Teacher forcing pass with gradients enabled
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(sequences).logits  # [G, max_len, vocab_size]
            
            # Apply temperature if configured  
            temp = self.config['generation'].get('temperature', 1.0)
            if temp != 1.0:
                logits = logits / temp
                
            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # [G, max_len, vocab_size]
            
            # Extract log probs of actual tokens (causal shift)
            target_ids = sequences[:, 1:].unsqueeze(-1)  # [G, max_len-1, 1]
            token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [G, max_len-1]
            
            # Sum over generation tokens only (excluding prompt)
            gen_start = prompt_len - 1  # -1 for causal shift
            gen_token_log_probs = token_log_probs[:, gen_start:]  # [G, gen_len]
            
            # Create mask for real generation tokens  
            gen_mask = attention_mask[:, prompt_len:].float()  # [G, gen_len]
            if gen_mask.shape[1] > gen_token_log_probs.shape[1]:
                gen_mask = gen_mask[:, :gen_token_log_probs.shape[1]]
            elif gen_mask.shape[1] < gen_token_log_probs.shape[1]:
                gen_token_log_probs = gen_token_log_probs[:, :gen_mask.shape[1]]
            
            # Compute sequence log probabilities S_{n,g}
            seq_log_probs = (gen_token_log_probs * gen_mask).sum(dim=1)  # [G]
            
        finally:
            # Restore original training state
            self.model.train(was_training)
        
        # STEP 2: Build probe loss based on type
        if loss_type == "X":
            # L_X = mean_g((S_{n,g} - loo_mean(S_{n,·})) * S_{n,g})
            # Compute leave-one-out means for each response
            loo_means = torch.zeros_like(seq_log_probs)
            for g in range(G):
                if G > 1:
                    mask = torch.ones(G, dtype=torch.bool, device=self.device)
                    mask[g] = False  
                    loo_means[g] = seq_log_probs[mask].mean()
                else:
                    loo_means[g] = seq_log_probs[g]  # If G=1, use itself
                    
            probe_loss = ((seq_log_probs - loo_means) * seq_log_probs).mean()
            
        elif loss_type == "Y":
            # L_Y = mean_g((A_{n,g} / L_max) * S_{n,g})
            probe_loss = (advantages / max_length * seq_log_probs).mean()
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
            
        return probe_loss
        
    def _get_stop_processor(self):
        """Get stopping processor for generation."""
        from evals.utils_io import StopAfterAnswer
        from transformers import LogitsProcessorList
        return LogitsProcessorList([StopAfterAnswer(self._tokenizer)])
        
    def _compute_sequence_logprobs(self, sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """
        Compute sequence log probabilities via teacher forcing.
        
        Args:
            sequences: [G, total_len] generated sequences 
            prompt_len: Length of the prompt portion
            
        Returns:
            [G] sequence log probabilities
        """
        G, total_len = sequences.shape
        
        # Teacher forcing forward pass
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(sequences).logits  # [G, total_len, vocab_size]
            
        # Apply temperature if configured
        temp = self.config['generation'].get('temperature', 1.0)
        if temp != 1.0:
            logits = logits / temp
            
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [G, total_len, vocab_size]
        
        # Extract log probs of actual tokens (causal shift)
        target_ids = sequences[:, 1:].unsqueeze(-1)  # [G, total_len-1, 1]
        token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [G, total_len-1]
        
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
        
    def _compute_block_pair_contribution(self, b_start: int, b_end: int, c_start: int, c_end: int,
                                        logprobs: torch.Tensor, advantages: torch.Tensor,
                                        max_lengths: List[int],
                                        adam_preconditioner: 'AdamPreconditioner') -> float:
        """
        Compute contribution from block pair (b, c) where b ≠ c.
        
        This implements the "recompute X then Y" strategy from Section VII.B.
        """
        # Build L_X for block b
        L_X_b = self._build_block_probe_loss(
            b_start, b_end, logprobs, advantages, max_lengths, loss_type="X"
        )
        
        # Build L_Y for block c  
        L_Y_c = self._build_block_probe_loss(
            c_start, c_end, logprobs, advantages, max_lengths, loss_type="Y"
        )
        
        # Backward L_X_b to get X̃_b in param.grad
        self.model.zero_grad()
        L_X_b.backward()
        
        # Apply preconditioner to param.grad
        x_tilde_params = []
        for param in self.model.parameters():
            if param.grad is not None:
                x_tilde_param = adam_preconditioner.apply_preconditioner(param.grad)
                x_tilde_params.append(x_tilde_param)
            else:
                x_tilde_params.append(None)
                
        # Get gradients of L_Y_c without affecting param.grad
        y_grads = torch.autograd.grad(
            L_Y_c, self.model.parameters(),
            retain_graph=False, allow_unused=True
        )
        
        # Apply preconditioner and compute dot product
        dot_product = 0.0
        for x_tilde, y_grad in zip(x_tilde_params, y_grads):
            if x_tilde is not None and y_grad is not None:
                y_tilde = adam_preconditioner.apply_preconditioner(y_grad)
                dot_product += (x_tilde * y_tilde).sum().item()
                
        return dot_product
        
    def _compute_block_diagonal_contribution(self, block_start: int, block_end: int,
                                           logprobs: torch.Tensor, advantages: torch.Tensor, 
                                           max_lengths: List[int],
                                           adam_preconditioner: 'AdamPreconditioner') -> float:
        """Compute diagonal contribution Σ (X̃_b · Ỹ_b) for a single block."""
        # This would implement the diagonal computation for within-block terms
        # Simplified for now
        return 0.0
        
    def _build_block_probe_loss(self, start_idx: int, end_idx: int,
                               logprobs: torch.Tensor, advantages: torch.Tensor,
                               max_lengths: List[int], loss_type: str) -> torch.Tensor:
        """
        Build scalar probe loss for a block of prompts.
        
        Args:
            start_idx, end_idx: Block boundaries
            logprobs: [B, G] log probabilities 
            advantages: [B, G] advantages
            max_lengths: [B] max lengths per prompt
            loss_type: "X" for entropy loss, "Y" for objective loss
            
        Returns:
            Scalar loss tensor
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for n in range(start_idx, end_idx):
            S_n = logprobs[n, :]
            A_n = advantages[n, :]
            L_max_n = max_lengths[n]
            G = len(S_n)
            
            if loss_type == "X":
                # Compute leave-one-out baseline
                S_n_loo = torch.zeros_like(S_n)
                for g in range(G):
                    mask = torch.ones_like(S_n, dtype=torch.bool)
                    mask[g] = False
                    S_n_loo[g] = S_n[mask].mean()
                    
                # L_X contribution: (S_{n,g} - S̄_{n,-g}) * S_{n,g}
                loss_n = ((S_n - S_n_loo) * S_n).sum()
                weight_n = G
                
            elif loss_type == "Y":
                # L_Y contribution: (A_{n,g} / L_max) * S_{n,g}
                loss_n = (A_n * S_n).sum() / L_max_n
                weight_n = G
                
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
                
            total_loss += loss_n
            total_weight += weight_n
            
        # Return normalized scalar loss
        return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=self.device)
        
    def _get_learning_rate(self) -> float:
        """Extract learning rate from optimizer for δH₁ computation."""
        # This is a simplified version - you may need to adapt based on your optimizer
        if hasattr(self.model, 'optimizer'):
            return self.model.optimizer.param_groups[0]['lr']
        else:
            # Use from config as fallback
            return self.config.get('learning_rate', 1e-6)  # Default fallback