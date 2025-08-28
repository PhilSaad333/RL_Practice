"""
In this file we compute an estimate of the variance of our estimator for \delta H_1, conditioned on the update batch U.

We reuse some of the computation from \delta H_1, but we also need to do a bunch of new computation, so we have moved 
this to a separate file for organization. 

The variance estimator we use (haven't implemented another option, the jackknife), is given by a sum over prompts 
\hat{V} = \sum_p \hat{V}_p, where \hat{V}_p must be computed using some graidents. Naively we could just compute an
auxiliary loss for each prompt and backprop through that, but in order to parallelize this efficiently we need to use the
"alpha trick" (suggested by chat gpt, I don't know who originally came up with this). This ended up being a massive headache,
and I couldn't get it to work until I turned off gradient checkpointing and used normal LoRA instead of QLoRA. But perhaps
now that it is working we can try to reintroduce those things later.

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


class ConditionalVarianceEstimator:
    """
    Conditional variance estimation methods for entropy probe.
    
    This class handles the α-trick optimization and conditional variance computation
    that was previously in ProbeComponents but is now separated for organization.
    """
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], logger: logging.Logger, tokenizer):
        """Initialize ConditionalVarianceEstimator with shared resources from ProbeComponents."""
        self.model = model
        self.config = config
        self.logger = logger
        self._tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # AMP settings from config
        self.use_amp = config['memory_config']['amp']
        self.amp_dtype = getattr(torch, config['memory_config']['dtype'])

        # --- Canonical Parameter Registry (same pattern as ProbeComponents) ---
        self._peft = self.model.module if hasattr(self.model, "module") else self.model

        self._trainable_named = [
            (n, p) for (n, p) in self._peft.named_parameters() if p.requires_grad
        ]
        self._trainable_params = [p for _, p in self._trainable_named]
        self._trainable_ids    = {id(p) for p in self._trainable_params}

        self._lora_named = [
            (n, p) for (n, p) in self._trainable_named
            if ("lora_a" in n.lower()) or ("lora_b" in n.lower()) or n.endswith("lm_head.weight")
        ]
        self._lora_params = [p for _, p in self._lora_named]

        if hasattr(self, "logger"):
            self.logger.debug(f"[REGISTRY] CV: trainable={len(self._trainable_params)} lora={len(self._lora_params)}")

    def _assert_grad_context(self, where: str):
        """Assert that we're in a valid gradient context for probe computations."""
        # Inference mode cannot be turned back on locally; never enter it for probe passes.
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                f"{where}: torch.inference_mode() is enabled. "
                "This disables autograd globally; rerun probe outside inference_mode."
            )
        if not torch.is_grad_enabled():
            raise RuntimeError(
                f"{where}: torch.no_grad() is active. "
                "Enable grad (remove no_grad) for probe computations."
            )
            
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

    def build_LX_vector_from_S(self, S_dict: Dict[str, Any], weighting_mode: str = "dr_grpo") -> torch.Tensor:
        """
        Vectorized per-prompt X-losses for α-trick conditional variance.
        
        Args:
            S_dict: Dictionary containing:
                - S: [k, G] tensor of per-response sequence log-probs (requires_grad=True)
                - max_lengths: List of length k with L_max per prompt (DR-GRPO style)
            weighting_mode: Loss weighting method
            
        Returns:
            L_vec: [k] tensor, one scalar loss per prompt (requires_grad=True)
            Convention: ∇_θ L_vec[b] = ∇_θ H for prompt b (minus sign included)
        """
        S = S_dict["S"]  # [k, G], requires_grad=True
        assert S.dim() == 2, f"S must be [k, G], got shape {S.shape}"
        
        if weighting_mode == "dr_grpo":
            # Divide each response score by L_max(prompt): broadcast [k, 1]
            if isinstance(S_dict["max_lengths"], list):
                Lmax = torch.tensor(S_dict["max_lengths"], device=S.device, dtype=S.dtype).view(-1, 1)
            else:
                Lmax = S_dict["max_lengths"].to(device=S.device, dtype=S.dtype).view(-1, 1)
            S_w = S / Lmax  # [k, G]
        elif weighting_mode == "per_token_avg":
            # Expect gen_lengths [k, G] with per-response effective token counts
            genL = S_dict["gen_lengths"].to(device=S.device, dtype=S.dtype)
            S_w = S / genL.clamp_min(1.0)
        else:
            raise ValueError(f"Unknown weighting_mode: {weighting_mode}")
            
        # Leave-One-Out mean for each response within its prompt: [k, G]
        G = S.size(1)
        sum_Sw = S_w.sum(dim=1, keepdim=True)           # [k, 1]
        S_w_LOO = (sum_Sw - S_w) / max(G - 1, 1)       # [k, G]
        
        # Coefficient with stop-grad; MINUS sign so ∇ L_X = ∇ H
        coeff = (S_w - S_w_LOO).detach()                # [k, G]
        L_vec = -(coeff * S).mean(dim=1)                # [k]
        
        return L_vec
    
    def _batched_teacher_force_logprobs(self, prompt_batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute sequence log-probabilities for multiple prompts in batches with proper padding.
        
        This method processes multiple prompts together for efficiency while maintaining
        per-prompt accuracy. Uses left-padding + right-padding strategy to align 
        prompt-generation boundaries.
        
        Args:
            prompt_batches: List of single-prompt batch dicts (from _iter_units)
            
        Returns:
            List of S_dict results (same as _teacher_force_logprobs output)
        """
        results = []
        
        # Get batch size from config
        batch_size = self.config.get("probe_rework", {}).get("conditional_variance_batch_size", 6)
        
        # Process prompts in groups of batch_size
        for i in range(0, len(prompt_batches), batch_size):
            batch_group = prompt_batches[i:i + batch_size]
            
            # Convert single-prompt batches to multi-prompt batch with aligned padding
            aligned_batch = self._align_prompt_batch_with_padding(batch_group)
            
            # Process the aligned batch efficiently
            batch_S_dict = self._compute_aligned_batch_logprobs(aligned_batch)
            
            # Split results back to per-prompt format
            per_prompt_results = self._split_batch_results(batch_S_dict, batch_group)
            results.extend(per_prompt_results)
            
        return results
        
    def _align_prompt_batch_with_padding(self, prompt_batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Align multiple single-prompt batches using left-padding + right-padding strategy.
        
        Layout: [left_pad][prompt][generation][right_pad]
        - All prompts end at the same index (max_prompt_len) 
        - All generations start at the same index (max_prompt_len)
        - Longest prompt has no left padding
        - Longest generation has no right padding
        
        Args:
            prompt_batches: List of single-prompt batch dicts
            
        Returns:
            Aligned batch dict with shape [num_prompts, G, total_aligned_len]
        """
        num_prompts = len(prompt_batches)
        if num_prompts == 0:
            raise ValueError("Empty prompt batch list")
            
        # Extract info from each prompt
        all_sequences = []
        all_prompt_lens = []
        all_advantages = []
        all_max_lengths = []
        
        for prompt_batch in prompt_batches:
            # Each prompt_batch has shape [1, G, seq_len] - extract the single prompt
            sequences = prompt_batch['sequences'][0]  # [G, seq_len]  
            prompt_len = prompt_batch['prompt_lens'][0]  # scalar
            advantages = prompt_batch['advantages'][0]  # [G]
            max_length = prompt_batch['max_lengths'][0]  # scalar
            
            all_sequences.append(sequences)
            all_prompt_lens.append(prompt_len)
            all_advantages.append(advantages)
            all_max_lengths.append(max_length)
            
        # Determine padding requirements
        max_prompt_len = max(all_prompt_lens)
        G = all_sequences[0].shape[0]  # Number of generations per prompt
        
        # Calculate generation lengths for each prompt 
        gen_lengths = []
        for i, seq in enumerate(all_sequences):
            gen_len = seq.shape[1] - all_prompt_lens[i]  # total - prompt = generation
            gen_lengths.append(gen_len)
        max_gen_len = max(gen_lengths)
        
        total_aligned_len = max_prompt_len + max_gen_len
        
        # Create aligned sequences with proper padding
        aligned_sequences = []
        aligned_prompt_lens = []  # All will be max_prompt_len after alignment
        aligned_advantages = []
        aligned_max_lengths = []
        
        for i, seq in enumerate(all_sequences):
            prompt_len = all_prompt_lens[i]
            gen_len = gen_lengths[i]
            
            # Split original sequence: [prompt_tokens][gen_tokens] 
            prompt_tokens = seq[:, :prompt_len]  # [G, prompt_len]
            gen_tokens = seq[:, prompt_len:]     # [G, gen_len]
            
            # Apply padding: [left_pad][prompt][gen][right_pad]
            left_pad_len = max_prompt_len - prompt_len
            right_pad_len = max_gen_len - gen_len
            
            # Create padded sequence for each generation
            padded_seqs = []
            for g in range(G):
                # Left padding
                if left_pad_len > 0:
                    left_pad = torch.full((left_pad_len,), self._tokenizer.pad_token_id, 
                                        dtype=seq.dtype, device=seq.device)
                else:
                    left_pad = torch.empty(0, dtype=seq.dtype, device=seq.device)
                
                # Right padding  
                if right_pad_len > 0:
                    right_pad = torch.full((right_pad_len,), self._tokenizer.pad_token_id,
                                         dtype=seq.dtype, device=seq.device) 
                else:
                    right_pad = torch.empty(0, dtype=seq.dtype, device=seq.device)
                    
                # Concatenate: [left_pad][prompt][gen][right_pad]
                padded_seq = torch.cat([left_pad, prompt_tokens[g], gen_tokens[g], right_pad], dim=0)
                padded_seqs.append(padded_seq)
                
            aligned_seq = torch.stack(padded_seqs, dim=0)  # [G, total_aligned_len]
            aligned_sequences.append(aligned_seq)
            aligned_prompt_lens.append(max_prompt_len)  # All prompts now end at same index
            aligned_advantages.append(all_advantages[i])
            aligned_max_lengths.append(all_max_lengths[i])
            
        # Stack into final batch format
        batch_sequences = torch.stack(aligned_sequences, dim=0)  # [num_prompts, G, total_aligned_len]
        
        # Create attention masks for aligned sequences
        attention_masks = []
        for seq in aligned_sequences:
            if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                mask = (seq != self._tokenizer.pad_token_id).long()
            else:
                mask = torch.ones_like(seq, dtype=torch.long)
            attention_masks.append(mask)
        batch_attention_masks = torch.stack(attention_masks, dim=0)  # [num_prompts, G, total_aligned_len]
        
        return {
            'sequences': batch_sequences,
            'prompt_lens': aligned_prompt_lens,  # All are max_prompt_len 
            'attention_masks': batch_attention_masks,
            'advantages': torch.stack(aligned_advantages, dim=0),
            'max_lengths': aligned_max_lengths,
            'total_aligned_len': total_aligned_len,
            'max_prompt_len': max_prompt_len,
            'max_gen_len': max_gen_len,
            'original_prompt_lens': all_prompt_lens,  # Keep originals for reference
            'gen_lengths': gen_lengths
        }
        
    def _compute_aligned_batch_logprobs(self, aligned_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute per-response sequence log-probs S for a batch of prompts with aligned shapes.
        Returns dict with:
          'S': [num_prompts, G], requires_grad=True (sum over generated tokens per response)
          'sequences', 'advantages', 'max_lengths' (and optionally 'gen_lengths')
        Assumes: aligned_batch has fields:
          sequences: [num_prompts, G, total_len] (left-padded), dtype long
          attention_masks: [num_prompts, G, total_len], 1=real token, 0=pad
          max_prompt_len: int
          advantages: [num_prompts, G] (if present in your pipeline)
          max_lengths: list[int] or tensor [num_prompts]
        """
        self._assert_grad_context("_compute_aligned_batch_logprobs")
        sequences = aligned_batch['sequences']
        attention_masks = aligned_batch['attention_masks']
        max_prompt_len = aligned_batch['max_prompt_len']
        num_prompts, G, total_len = sequences.shape

        was_training = self._peft.training  # use the same PEFT instance used for forward
        self._peft.eval()                   # turn off dropout/noise

        try:
            # IMPORTANT: ensure graph tracking regardless of outer context
            with torch.set_grad_enabled(True):
                flat_sequences = sequences.view(num_prompts * G, total_len)
                flat_attention_masks = attention_masks.view(num_prompts * G, total_len)

                # Use canonical PEFT model and ensure adapter is active BEFORE forward
                peft_model = self._peft
                if hasattr(peft_model, "set_adapter"):
                    peft_model.set_adapter("default")
                
                # Forward with attention_mask; AMP as in training if you use it
                use_amp = getattr(self, "use_amp", False)
                amp_dtype = getattr(self, "amp_dtype", torch.float16)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = peft_model(input_ids=flat_sequences, attention_mask=flat_attention_masks)
                    logits = outputs.logits  # [B, T, V]

                # Assert logits are connected to autograd graph
                if not logits.requires_grad or logits.grad_fn is None:
                    raise RuntimeError(
                        "TF logits are not connected to autograd graph "
                        "(likely due to surrounding inference_mode/no_grad)."
                    )

                # Apply temperature (keeping consistent with existing behavior)
                temp = self.config['generation'].get('temperature', 1.0)
                if temp != 1.0:
                    logits = logits / temp

                # Token-level log-probs for next-token targets, with proper masking
                log_probs = F.log_softmax(logits.float(), dim=-1)   # [B, T, V]
                target_ids = flat_sequences[:, 1:].unsqueeze(-1)    # shift left by 1
                token_log_probs = log_probs[:, :-1].gather(2, target_ids).squeeze(-1)  # [B, T-1]

                # Reshape back to [num_prompts, G, T-1]
                token_log_probs = token_log_probs.view(num_prompts, G, total_len - 1)

                # Generated-region slice (exclude prompt tokens); index is (max_prompt_len - 1)
                gen_start = int(max_prompt_len) - 1
                gen_token_log_probs = token_log_probs[:, :, gen_start:]  # [num_prompts, G, L_gen]

                # Build generation mask from sequences; 1 for non-pad tokens in generated region
                gen_sequences = sequences[:, :, max_prompt_len:]  # [num_prompts, G, L_gen]
                if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                    gen_masks = (gen_sequences != self._tokenizer.pad_token_id).float()
                else:
                    gen_masks = torch.ones_like(gen_sequences, dtype=torch.float32)

                # Align shapes defensively
                if gen_masks.shape != gen_token_log_probs.shape:
                    min_len = min(gen_masks.shape[2], gen_token_log_probs.shape[2])
                    gen_masks = gen_masks[:, :, :min_len]
                    gen_token_log_probs = gen_token_log_probs[:, :, :min_len]

                # Sequence log-prob per response (sum over generated tokens)
                S_batch = (gen_token_log_probs * gen_masks).sum(dim=2)  # [num_prompts, G]
                
                # Assert S_batch is connected to autograd graph
                if not S_batch.requires_grad or S_batch.grad_fn is None:
                    raise RuntimeError("S_batch lost its grad_fn; check TF pipeline and masking.")
                
                # 🔧 DIAGNOSTIC: Quick on/off adapter check for this batch computation
                if num_prompts >= 1:  # Only test if we have data
                    test_seq = flat_sequences[0:1]  # [1, total_len]
                    test_mask = flat_attention_masks[0:1] if flat_attention_masks is not None else None
                    
                    with torch.no_grad():
                        logits_on = peft_model(test_seq, attention_mask=test_mask).logits
                        with peft_model.disable_adapter():
                            logits_off = peft_model(test_seq, attention_mask=test_mask).logits
                    
                    delta = (logits_on - logits_off).abs().max().item()
                    self.logger.debug(f"🔧 [ALIGNED-BATCH-DEBUG] LoRA on/off delta: {delta:.2e}")
                    if delta < 1e-7:
                        raise RuntimeError("🔧 LoRA appears inactive in aligned batch computation: on/off logits nearly identical.")
                
                # Use canonical registry for LoRA params  
                params_for_probe = self._lora_params
                
                if len(params_for_probe) == 0:
                    raise RuntimeError("TF probe: no trainable LoRA params found on PEFT model (adapters missing/merged/disabled).")

                # Test gradient flow from S to LoRA params
                probe = S_batch[0].sum()  # scalar
                g_lora = torch.autograd.grad(probe, params_for_probe, retain_graph=True, allow_unused=True)
                non_none_count = sum(g is not None for g in g_lora)
                
                # Enhanced diagnostic logging
                self.logger.debug(f"🔧 [ALIGNED-BATCH-DEBUG] S→LoRA probe: {non_none_count}/{len(params_for_probe)} params got gradients")
                
                if not any(g is not None for g in g_lora):
                    raise RuntimeError("🔧 TF probe: S does not backprop to LoRA params in aligned batch computation. "
                                       "Forward pass not using adapters despite activation checks.")
                # --- end comprehensive probe ---

        finally:
            self._peft.train(was_training)

        out = {
            'S': S_batch,
            'sequences': sequences,
            'advantages': aligned_batch.get('advantages', None),
            'max_lengths': aligned_batch['max_lengths'],
        }
        if 'gen_lengths' in aligned_batch:
            out['gen_lengths'] = aligned_batch['gen_lengths']
        return out
        
    def _split_batch_results(self, batch_S_dict: Dict[str, Any], 
                           original_batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split batched results back to per-prompt format compatible with existing code.
        
        Args:
            batch_S_dict: Results from _compute_aligned_batch_logprobs
            original_batches: Original single-prompt batches for reference
            
        Returns:
            List of per-prompt S_dict results (same format as _teacher_force_logprobs)
        """
        S_batch = batch_S_dict['S']  # [num_prompts, G]
        
        # Runtime assertion to ensure S has gradients (critical for α-trick)
        assert S_batch.requires_grad, "S must require grad in TF path for α-trick to work"
        
        results = []
        for i, original_batch in enumerate(original_batches):
            # Extract results for prompt i
            S_single = S_batch[i:i+1]  # [1, G] - maintain batch dimension
            
            # Create result dict matching _teacher_force_logprobs output format
            result = {
                'S': S_single,
                'sequences': original_batch['sequences'],  # Use original sequences
                'advantages': original_batch['advantages'], 
                'max_lengths': original_batch['max_lengths']
            }
            results.append(result)
            
        return results


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
        self._assert_grad_context("_teacher_force_logprobs")
        sequences = prompt_batch['sequences']  # [batch_size, G, max_len]  
        prompt_lens = prompt_batch['prompt_lens']  # [batch_size]
        attention_masks = prompt_batch['attention_masks']  # [batch_size, G, max_len]
        advantages = prompt_batch['advantages']  # [batch_size, G]
        max_lengths = prompt_batch['max_lengths']  # [batch_size]
        
        batch_size, G, max_len = sequences.shape
        
        # 🔧 FIX 1: Ensure PEFT wrapper and active adapter (from fix.txt §1)
        # Use canonical PEFT module from registry 
        peft_model = self._peft
        
        # Make sure an adapter is active and trainable
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")  # or your adapter name
        
        # Sanity: we are not inside a disable_adapter() context
        active = getattr(peft_model, "active_adapter", None)
        if active is None:
            raise RuntimeError("No active adapter on PEFT model.")
            
        # 🔧 FIX 2: Add invariant logging (from fix.txt §6) - DEBUG SECTION
        self.logger.debug(f"🔧 [ADAPTER-DEBUG] active_adapter={active} training={peft_model.training}")
        trainable = [n for n,p in peft_model.named_parameters() if p.requires_grad]
        self.logger.debug(f"🔧 [ADAPTER-DEBUG] #trainable={len(trainable)} sample={trainable[:8]}")
        
        # 🔧 FIX 3: Prove LoRA is active at runtime (from fix.txt §2) - DEBUG SECTION  
        if batch_size >= 1:  # Only test on first prompt to avoid overhead
            seqs_small = sequences[0:1]  # [1, G, max_len]
            masks_small = attention_masks[0:1] if attention_masks is not None else None
            
            with torch.no_grad():
                logits_on = peft_model(seqs_small.view(-1, max_len), 
                                     attention_mask=masks_small.view(-1, max_len) if masks_small is not None else None).logits
                with peft_model.disable_adapter():
                    logits_off = peft_model(seqs_small.view(-1, max_len),
                                          attention_mask=masks_small.view(-1, max_len) if masks_small is not None else None).logits
            
            delta = (logits_on - logits_off).abs().max().item()
            self.logger.debug(f"🔧 [ADAPTER-DEBUG] LoRA on/off delta: {delta:.2e}")
            if delta < 1e-7:
                raise RuntimeError("LoRA appears inactive in TF: on/off logits nearly identical.")
        
        # Use deterministic forward passes (disable dropout/noise)  
        was_training = self._peft.training  # use the same PEFT instance used for forward
        self._peft.eval()                   # turn off dropout/noise
        
        try:
            # Compute log-probabilities for all sequences in batch
            S_batch = []  # [batch_size, G]
            
            for b in range(batch_size):
                seq_logprobs = self._compute_sequence_logprobs(sequences[b], prompt_lens[b])
                S_batch.append(seq_logprobs)
                
            # Stack into tensor with gradients enabled
            S = torch.stack(S_batch, dim=0)  # [batch_size, G]
            
            # 🔧 FIX 4: Assert S has gradient path (from fix.txt §6) - DEBUG SECTION
            assert S.requires_grad and S.grad_fn is not None, "S lost grad path (inference/no_grad or detach?)"
            self.logger.debug(f"🔧 [ADAPTER-DEBUG] S.requires_grad={S.requires_grad} S.grad_fn={S.grad_fn}")
            
        finally:
            # Restore training state
            self._peft.train(was_training)
        
        return {
            'S': S,  # [batch_size, G] with requires_grad=True
            'sequences': sequences,
            'advantages': advantages, 
            'max_lengths': max_lengths,
            'attention_masks': attention_masks
        }



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
        
        # G-MICROBATCHING: Process sequences in smaller chunks to reduce memory
        teacher_force_microbatch_size = self.config.get('memory_config', {}).get('teacher_force_microbatch_size', 2)
        self.logger.debug(f"Using teacher_force_microbatch_size={teacher_force_microbatch_size} for gradient computation")
        
        all_token_log_probs = []
        
        # Process G sequences in microbatches
        for g_start in range(0, G, teacher_force_microbatch_size):
            g_end = min(g_start + teacher_force_microbatch_size, G)
            micro_sequences = sequences[g_start:g_end]  # [g_micro, total_len]
            
            # Create attention mask for teacher forcing (padding only - causality handled by model architecture)
            if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                attention_mask = (micro_sequences != self._tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(micro_sequences, dtype=torch.long)
            
            # Teacher forcing forward pass on microbatch with attention mask
            # Note: Causal attention is handled automatically by the model architecture
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(micro_sequences, attention_mask=attention_mask).logits  # [g_micro, total_len, vocab_size]
                
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




    def compute_conditional_variance_over_E(self, E_batch: Dict[str, Any], 
                                          muY_buf: Dict[int, torch.Tensor],
                                          mb_size_prompts: int,
                                          weighting_mode: str) -> Tuple[float, float, int]:
        """
        Compute conditional variance Var_E(δH₁ | U) using scalar projections.
        
        This implements the new variance estimator that fixes U and computes variance
        over E only. Returns variance, standard error, and batch count.
        
        From variance_estimator_change.txt:
        - s_n := μ_Y^T X_n (scalar projection for each prompt n in E)
        - sample_var_s = (Σs_n² - B_E·s̄²) / (B_E-1)  
        - Var_E(δH₁|U) = η²·sample_var_s/B_E
        - SE_E(δH₁|U) = η·√(sample_var_s/B_E)
        
        Args:
            E_batch: Evaluation batch (used for X gradients)
            muY_buf: Fixed mean Y buffer (from U batch) 
            mb_size_prompts: Microbatch size for processing
            weighting_mode: Weighting mode for loss computation
            
        Returns:
            (sample_var_s, scalar_mean, B_E_local): Statistics for global computation
        """
        sum_s_local = 0.0
        sum_s2_local = 0.0
        B_E_local = 0
        
        # Check if model is wrapped in DDP for no_sync context
        is_ddp = hasattr(self.model, 'no_sync')
        no_sync_context = self.model.no_sync if is_ddp else nullcontext
        
        self.logger.debug("Computing conditional variance over E using scalar projections")
        
        # Use α-trick conditional variance for efficient computation
        cv_batch_size = self.config.get('probe_rework', {}).get('conditional_variance_batch_size', 8)
        
        # Convert E_batch to list of single-prompt units  
        E_units = list(self._iter_units(E_batch))
        
        # Call the α-trick method to compute all scalar projections efficiently
        sum_s_local, sum_s2_local, B_E_local = self.compute_conditional_variance_over_E_alpha(
            E_units=E_units,
            muY_buf=muY_buf,
            weighting_mode=weighting_mode,
            cv_batch_size=cv_batch_size
        )
        
        self.logger.debug(f"Conditional variance over E: B_E_local={B_E_local}, sum_s={sum_s_local:.6f}")
        return sum_s_local, sum_s2_local, B_E_local
    
    def compute_conditional_variance_over_E_alpha(
        self,
        E_units,
        muY_buf: dict,
        weighting_mode: str = "dr_grpo",
        cv_batch_size: int = 8,
    ):
        """
        Fast E-only conditional variance via α-trick.
        Returns (sum_s_local, sum_s2_local, B_E_local) as Python floats/ints.
        Caller assembles Var_E = η^2 * [ (Σ s^2 − B s̄^2) / (B − 1) ] / B, SE_E = η * sqrt( ... ).
        """
        self._assert_grad_context("compute_conditional_variance_over_E_alpha")
        
        # Note: QLoRA gradient flow diagnostics are now integrated into the main computation paths
                
        from contextlib import nullcontext
        device = next(self.model.parameters()).device
        sum_s_local = 0.0
        sum_s2_local = 0.0
        B_E_local = 0

        logger = getattr(self, "logger", None)
        def _dbg(msg):
            if logger is not None and hasattr(logger, "debug"):
                logger.debug(msg)

        no_sync_ctx = self.model.no_sync if hasattr(self.model, "no_sync") else nullcontext

        for i in range(0, len(E_units), cv_batch_size):
            group = E_units[i : i + cv_batch_size]
            if not group:
                continue

            # 1) One TF forward for this group; stack into a single S_dict
            batch_S_list = self._batched_teacher_force_logprobs(group)
            S_cat = torch.cat([d["S"].view(1, -1) for d in batch_S_list], dim=0)  # [k, G]

            if not S_cat.requires_grad:
                raise RuntimeError("α-trick: S.requires_grad=False — TF path detached from parameters.")

            S_dict = {"S": S_cat}
            if "max_lengths" in batch_S_list[0]:
                S_dict["max_lengths"] = [d["max_lengths"] for d in batch_S_list]
            if "gen_lengths" in batch_S_list[0]:
                S_dict["gen_lengths"] = torch.stack([d["gen_lengths"] for d in batch_S_list], dim=0)

            with no_sync_ctx():
                # 2) Per-prompt losses and α selector
                L_vec = self.build_LX_vector_from_S(S_dict, weighting_mode=weighting_mode)   # [k]
                alpha = torch.ones_like(L_vec, requires_grad=True)                           # [k]
                L = (alpha * L_vec).sum()
                _dbg(f"α-trick: group L={L.item():.6f}, L.requires_grad={L.requires_grad}")
                
                # --- Diagnostic 1: Check that dL/dS is non-zero ---
                dL_dS = torch.autograd.grad(L, S_dict["S"], retain_graph=True, allow_unused=False)[0]
                if (dL_dS.abs().sum() == 0):
                    raise RuntimeError("α-trick: dL/dS is exactly zero; check L_vec construction / coefficients")
                _dbg(f"α-trick: dL/dS.abs().sum()={dL_dS.abs().sum().item():.6f} (should be > 0)")
                
                # New (use canonical registry):
                params_for_test = self._lora_params
                if len(params_for_test) == 0:
                    raise RuntimeError("α‑trick: no trainable LoRA (or lm_head) params in registry.")

                _dbg(f"🔧 [PARAM-DEBUG] Using {len(params_for_test)} LoRA params from canonical registry")

                g_test = torch.autograd.grad(L, params_for_test, retain_graph=True, allow_unused=True)
                if not any(g is not None for g in g_test):
                    raise RuntimeError("α-trick: L does not backprop into LoRA params. "
                                       "If μY was built on the PEFT wrapper, make sure TF also uses the same wrapper.")
                # --- end diagnostic ---
                _dbg(f"🔧 [VJP-DEBUG] Direct L→LoRA test passed: {sum(g is not None for g in g_test)}/{len(g_test)} params got gradients")

                # 3) Robust VJP-based approach via S instead of direct L→params
                S = S_dict["S"]                     # [k, G], requires_grad=True
                k, G = S.shape

                # Rebuild the same coefficients used inside L_vec
                if weighting_mode == "dr_grpo":
                    Lmax = (torch.tensor(S_dict["max_lengths"], device=S.device, dtype=S.dtype)
                            if isinstance(S_dict["max_lengths"], list)
                            else S_dict["max_lengths"].to(device=S.device, dtype=S.dtype))
                    Lmax = Lmax.view(-1, 1)         # [k, 1]
                    S_w = S / Lmax                  # [k, G]
                elif weighting_mode == "per_token_avg":
                    genL = S_dict["gen_lengths"].to(device=S.device, dtype=S.dtype)
                    S_w = S / genL.clamp_min(1.0)
                else:
                    raise ValueError(f"Unknown weighting_mode: {weighting_mode}")

                sum_Sw = S_w.sum(dim=1, keepdim=True)        # [k, 1]
                S_w_LOO = (sum_Sw - S_w) / max(G - 1, 1)     # [k, G]

                # dL/dS = -(alpha/G) * (S_w - S_w_LOO)   (alpha is broadcast over g)
                W = - (alpha.view(-1, 1) / float(G)) * (S_w - S_w_LOO)  # [k, G]

                params = params_for_test  # canonical list

                g_list = torch.autograd.grad(
                    outputs=S, inputs=params, grad_outputs=W,
                    create_graph=True, allow_unused=True, retain_graph=False
                )
                non_none = sum(int(gi is not None) for gi in g_list)
                _dbg(f"[VJP] S→params non‑None grads = {non_none}/{len(g_list)}")
                if non_none == 0:
                    raise RuntimeError("α‑trick VJP: S→params produced no grads; check TF using adapters.")

                h = torch.zeros((), device=S.device, dtype=S.dtype)
                for p, gi in zip(params, g_list):
                    if gi is None:
                        continue
                    mu = muY_buf.get(id(p))
                    if mu is None:
                        continue  # strict: only contract over known μ_Y entries
                    if mu.device != gi.device or mu.dtype != gi.dtype:
                        mu = mu.to(device=gi.device, dtype=gi.dtype)
                    h = h + (gi * mu).sum()

                # 5) Second reverse: s = ∂h/∂α  → [k]
                s = torch.autograd.grad(h, alpha, allow_unused=False, retain_graph=False)[0].detach()  # [k]

            # 6) Accumulate scalars
            sum_s_local  += float(s.sum().item())
            sum_s2_local += float((s * s).sum().item())
            B_E_local    += int(s.numel())

            # 7) Clear grads to keep memory bounded
            self.model.zero_grad(set_to_none=True)

        return sum_s_local, sum_s2_local, B_E_local

    def _test_minimal_gradient_flow(self, batch_ids: torch.Tensor, batch_mask: torch.Tensor) -> None:
        """
        Minimal gradient flow test from fix.txt to diagnose parameter trainability.
        
        This implements the diagnostic test from fix.txt section C to verify:
        1. Model forward produces gradients
        2. LoRA parameters receive gradients from a simple loss
        
        Args:
            batch_ids: Input token IDs [batch_size, seq_len]
            batch_mask: Attention mask [batch_size, seq_len]
        """
        self.logger.info("Testing minimal gradient flow...")
        
        # Use deterministic forward passes (disable dropout/noise)
        self._peft.eval()
        
        # 1) Tiny TF forward on the PEFT model
        outs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = outs.logits  # must have grad_fn
        
        # Assert logits are tracking gradients
        if not (logits.requires_grad and logits.grad_fn is not None):
            raise RuntimeError("Minimal test: logits not tracking grad - model may be frozen or in eval mode")
        
        # 2) Build S and a vectorized loss L = <alpha, L_vec(S)>
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_lp = logprobs[:, :-1].gather(2, batch_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        S = token_lp.sum(dim=1).view(-1, 1)  # [k, 1]
        S.retain_grad()
        
        alpha = torch.ones(S.shape[0], device=S.device, requires_grad=True)
        L = (-(alpha.view(-1, 1) * S).mean())  # minus sign; simple cross-entropy variant
        
        # 3) Use canonical LoRA params from registry
        lora_params = self._lora_params
        
        if len(lora_params) == 0:
            raise RuntimeError("Minimal test: No lora_A/B parameters found (adapters missing/merged?)")
        
        self.logger.info(f"Found {len(lora_params)} LoRA parameters for gradient test")
        
        g_lora = torch.autograd.grad(L, lora_params, allow_unused=True, retain_graph=True)
        non_none_lora = sum(int(g is not None) for g in g_lora)
        
        self.logger.info(f"Minimal test results: {non_none_lora}/{len(lora_params)} LoRA params received gradients")
        
        if non_none_lora == 0:
            raise RuntimeError("Minimal test FAILED: Forward isn't using the same adapter tensors you're differentiating w.r.t. (wrong model object or adapters disabled)")
        
        self.logger.info("Minimal test PASSED: LoRA gradient flow is working correctly for QLoRA setup")
