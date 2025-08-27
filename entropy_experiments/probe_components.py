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
    
    # =====================================================================
    # DELETED LEGACY METHODS (Old U-statistic approach)
    # =====================================================================
    # - _compute_delta_h1_exact: Old exact per-prompt U-statistic implementation (~240 lines)
    # - _compute_delta_h1_blocks: Old block-level U-statistic implementation (~190 lines)  
    # - _build_prompt_probe_loss: Used by old exact computation
    # - _build_block_probe_loss: Used by old blocks method
    # - _compute_block_diagonal_contribution: Helper for block approach
    # Total deleted: ~500 lines of legacy U-statistic code
    # =====================================================================
        """
        start_time = time.time()
        self.logger.info("Computing δH₁ using microbatched exact per-prompt method")
        
        # Extract dimensions
        B = batch_data['sequences'].shape[0]
        G = batch_data['sequences'].shape[1]
        
        # Get microbatch size from config (default to 4 if not specified)
        microbatch_size = self.config.get('memory_config', {}).get('microbatch_size', 4)
        self.logger.info(f"Using microbatch_size={microbatch_size} for memory efficiency")
        
        # Get model parameters that require gradients (LoRA parameters only)
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_trainable_params = sum(p.numel() for p in params)
        self.logger.info(f"Using {total_trainable_params:,} trainable parameters ({len(params)} tensors) for gradient computation")
        
        # =========================================================================
        # PHASE 1: X-PASS - Accumulate ΣX via microbatches
        # =========================================================================
        self.logger.debug("Phase 1: X-pass - accumulating ΣX via microbatches")
        
        # ✅ CPU ACCUMULATORS: Allocate ΣX buffer on CPU to save GPU memory
        X_sum = {}
        for param in params:
            X_sum[id(param)] = torch.zeros_like(param, device='cpu', dtype=torch.float32)
            
        # Accumulate X gradients over microbatches
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # 1a. Forward with grad to get S for this microbatch
            S_dict = self._teacher_force_logprobs(microbatch)
            L_X_mb = self._build_probe_loss_X_from_S(S_dict)
            
            # 1b. Backward immediately (microbatch)
            self.model.zero_grad()
            L_X_mb.backward()
            
            # 1c. Apply preconditioner and accumulate into ΣX (CPU)
            for param in params:
                if param.grad is not None:
                    # Move gradient to CPU in float32 for accumulation
                    grad_cpu = param.grad.detach().to('cpu', dtype=torch.float32)
                    X_sum[id(param)].add_(grad_cpu)
        
        # 🔍 X-PASS GRADIENT FLOW DEBUG (before clearing)
        non_none_x = sum(1 for p in params if p.grad is not None)
        if non_none_x > 0:
            x_grad_norms = [p.grad.norm().item() for p in params if p.grad is not None]
            x_grad_mean = sum(x_grad_norms) / len(x_grad_norms)
            x_grad_max = max(x_grad_norms)
            self.logger.info(f"🔍 [X-PASS] Gradients: {non_none_x}/{len(params)} params, mean_norm={x_grad_mean:.2e}, max_norm={x_grad_max:.2e}")
        else:
            self.logger.warning(f"🔍 [X-PASS] WARNING: No gradients found! {non_none_x}/{len(params)}")
            
        # ✅ Enhanced memory cleanup
        self.model.zero_grad(set_to_none=True)  # More aggressive cleanup
        torch.cuda.empty_cache()
        
        # =========================================================================
        # PHASE 2: Y-PASS - Accumulate ΣY via microbatches (Option A)
        # =========================================================================
        self.logger.debug("Phase 2: Y-pass - accumulating ΣY via microbatches")
        
        # ✅ CPU ACCUMULATORS: Allocate ΣY buffer on CPU to save GPU memory  
        Y_sum = {}
        for param in params:
            Y_sum[id(param)] = torch.zeros_like(param, device='cpu', dtype=torch.float32)
            
        # 🔍 SAVE X GRADIENTS before Y-pass (since backward() overwrites param.grad)
        X_gradients_saved = {}
        for param in params:
            if param.grad is not None:
                X_gradients_saved[id(param)] = param.grad.clone()
            
        # Accumulate Y gradients over microbatches
        first_microbatch = True
        for microbatch in self._iter_prompt_microbatches(batch_data, microbatch_size):
            # Fresh forward with grad
            S_dict = self._teacher_force_logprobs(microbatch)
            L_Y_mb = self._build_probe_loss_Y_from_S(S_dict)
            
            # 🔍 DEBUG: Check L_Y_mb gradient requirements
            self.logger.info(f"🔍 [L_Y] requires_grad={L_Y_mb.requires_grad}, grad_fn={L_Y_mb.grad_fn}")
            
            # 🔍 Clear gradients before Y computation
            for param in params:
                param.grad = None
                        
            # Use backward() instead of autograd.grad (more reliable)
            L_Y_mb.backward()
            
            # Apply preconditioner and accumulate Y gradients into ΣY (CPU)
            for param in params:
                if param.grad is not None:
                    preconditioned_grad = adam_preconditioner.apply_preconditioner(param.grad, param)
                    # Move gradient to CPU in float32 for accumulation
                    grad_cpu = preconditioned_grad.detach().to('cpu', dtype=torch.float32)
                    Y_sum[id(param)].add_(grad_cpu)
        
        # 🔍 Y-PASS GRADIENT FLOW DEBUG  
        y_non_none = sum(1 for g in y_grads if g is not None)
        if y_non_none > 0:
            y_grad_norms = [g.norm().item() for g in y_grads if g is not None]
            y_grad_mean = sum(y_grad_norms) / len(y_grad_norms)
            y_grad_max = max(y_grad_norms)
            self.logger.info(f"🔍 [Y-PASS] Gradients: {y_non_none}/{len(params)} params, mean_norm={y_grad_mean:.2e}, max_norm={y_grad_max:.2e}")
        else:
            self.logger.warning(f"🔍 [Y-PASS] WARNING: No gradients found! {y_non_none}/{len(params)}")
        
        # ✅ Enhanced memory cleanup after Y-pass
        torch.cuda.empty_cache()
        
        # 🔍 X_SUM/Y_SUM MAGNITUDE DEBUG
        x_sum_norms = [X_sum[id(p)].norm().item() for p in params if id(p) in X_sum]
        y_sum_norms = [Y_sum[id(p)].norm().item() for p in params if id(p) in Y_sum]
        if x_sum_norms and y_sum_norms:
            x_sum_mean = sum(x_sum_norms) / len(x_sum_norms)
            y_sum_mean = sum(y_sum_norms) / len(y_sum_norms)
            x_sum_max = max(x_sum_norms)
            y_sum_max = max(y_sum_norms)
            self.logger.info(f"🔍 [SUMS] X_sum mean_norm={x_sum_mean:.2e}, max_norm={x_sum_max:.2e}")
            self.logger.info(f"🔍 [SUMS] Y_sum mean_norm={y_sum_mean:.2e}, max_norm={y_sum_max:.2e}")
        
        # Compute bars_dot = (ΣX · ΣY) / B²
        bars_dot = 0.0
        for param in params:
            param_id = id(param)
            if param_id in X_sum and param_id in Y_sum:
                dot_product = (X_sum[param_id] * Y_sum[param_id]).sum().item()
                bars_dot += dot_product
        bars_dot /= (B * B)
        
        # =========================================================================
        # PHASE 3: DIAGONAL LOOP - Compute Σ diag and row-means r_u
        # =========================================================================
        self.logger.debug("Phase 3: Diagonal loop - computing diag terms and row means")
        
        diag_sum = 0.0
        r_values = []
        
        for unit, unit_idx in self._iter_prompts_as_units(batch_data):
            # X forward
            self.model.zero_grad()
            S_uX = self._teacher_force_logprobs(unit)
            L_X_u = self._build_probe_loss_X_from_S(S_uX)
            L_X_u.backward()                                   # frees graph

            # Y forward (fresh)
            S_uY = self._teacher_force_logprobs(unit)
            L_Y_u = self._build_probe_loss_Y_from_S(S_uY)
            y_grads_u = torch.autograd.grad(L_Y_u, params, allow_unused=True)

            dot_Xu_Yu = dot_Xu_SumY = dot_Yu_SumX = 0.0
            for param, y_grad in zip(params, y_grads_u):
                pid = id(param)
                X_u = param.grad
                if X_u is None:
                    continue
                Y_u = adam_preconditioner.apply_preconditioner(y_grad, param) if y_grad is not None else None
                if Y_u is None:
                    continue
                dot_Xu_Yu   += (X_u * Y_u).sum().item()
                dot_Xu_SumY += (X_u * Y_sum[pid]).sum().item()
                dot_Yu_SumX += (Y_u * X_sum[pid]).sum().item()
            
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
            
            # ✅ Enhanced memory cleanup in diagonal loop
            self.model.zero_grad(set_to_none=True)
            if unit_idx % 2 == 0:  # Cleanup every 2 prompts to balance performance vs memory
                torch.cuda.empty_cache()
        
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
            
        learning_rate = self._get_learning_rate(optimizer)
        delta_h1 = -learning_rate * U_cross
        
        # 🔍 HIGH-PRECISION DEBUG LOGGING
        self.logger.info(f"🔍 [ASSEMBLY] U_cross={U_cross:.10f}")
        self.logger.info(f"🔍 [ASSEMBLY] learning_rate={learning_rate:.2e}")
        self.logger.info(f"🔍 [ASSEMBLY] delta_h1={delta_h1:.10f}")
        self.logger.info(f"🔍 [ASSEMBLY] bars_dot={bars_dot:.10f}")
        self.logger.info(f"🔍 [ASSEMBLY] diag_sum={diag_sum:.10f}")
        
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
        
        # Get model parameters that require gradients (LoRA parameters only)
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_trainable_params = sum(p.numel() for p in params)
        self.logger.info(f"Using {total_trainable_params:,} trainable parameters ({len(params)} tensors) for gradient computation")
        
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
                    X_sum_blocks[id(param)].add_(param.grad)
            
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
            # X_b forward/backward
            self.model.zero_grad()
            S_bX = self._teacher_force_logprobs(microbatch)
            L_X_b = self._build_probe_loss_X_from_S(S_bX)
            L_X_b.backward()

            # Y_b forward/grad (fresh graph)
            S_bY = self._teacher_force_logprobs(microbatch)
            L_Y_b = self._build_probe_loss_Y_from_S(S_bY)
            y_grads_b = torch.autograd.grad(L_Y_b, params, allow_unused=True)

            dot_Xb_Yb = 0.0
            dot_Xb_SumY = 0.0
            dot_Yb_SumX = 0.0

            for param, y_grad in zip(params, y_grads_b):
                pid = id(param)
                X_b = param.grad
                if X_b is None:
                    continue
                Y_b = adam_preconditioner.apply_preconditioner(y_grad, param) if y_grad is not None else None
                if Y_b is None:
                    continue

                dot_Xb_Yb   += (X_b * Y_b).sum().item()
                dot_Xb_SumY += (X_b * Y_sum_blocks[pid]).sum().item()
                dot_Yb_SumX += (Y_b * X_sum_blocks[pid]).sum().item()

            # accumulate diagonal
            diag_sum_blocks += dot_Xb_Yb

            # r_b = 0.5 [X_b · Ȳ_{−b} + Y_b · X̄_{−b}]
            if M > 1:
                XdotYbar_minus = (dot_Xb_SumY - dot_Xb_Yb) / (M - 1)
                YdotXbar_minus = (dot_Yb_SumX - dot_Xb_Yb) / (M - 1)
                r_b = 0.5 * (XdotYbar_minus + YdotXbar_minus)
            else:
                r_b = 0.0

            r_values_blocks.append(r_b)
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
            with torch.no_grad():
                loo_means = torch.zeros_like(seq_log_probs)
                for g in range(G):
                    if G > 1:
                        mask = torch.ones(G, dtype=torch.bool, device=self.device)
                        mask[g] = False  
                        loo_means[g] = seq_log_probs[mask].mean()
                    else:
                        loo_means[g] = seq_log_probs[g]  # If G=1, use itself
            
            coeff = (seq_log_probs - loo_means).detach()
            probe_loss = (coeff * seq_log_probs).mean()
            
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
                with torch.no_grad():
                    S_n_loo = torch.zeros_like(S_n)
                    for g in range(G):
                        mask = torch.ones_like(S_n, dtype=torch.bool)
                        mask[g] = False
                        S_n_loo[g] = S_n[mask].mean()
                    
                # L_X contribution: (S_{n,g} - S̄_{n,-g}) * S_{n,g}
                coeff = (S_n - S_n_loo).detach()
                loss_n = (coeff * S_n).sum()
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
        
    def _get_learning_rate(self, optimizer: Optional[torch.optim.Optimizer] = None) -> float:
        """Extract learning rate from optimizer for δH₁ computation."""
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
            self.logger.info(f"🔍 [LR] Extracted learning rate from optimizer: {lr:.2e}")
            return lr
        elif hasattr(self.model, 'optimizer'):
            lr = self.model.optimizer.param_groups[0]['lr']
            self.logger.info(f"🔍 [LR] Extracted learning rate from model.optimizer: {lr:.2e}")
            return lr
        else:
            # Use from config as fallback
            lr = self.config.get('learning_rate', 1e-6)  # Default fallback
            self.logger.warning(f"🔍 [LR] Using fallback learning rate: {lr:.2e} (may be incorrect!)")
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