"""
Probe Components

Core computational components for the computation of \delta H_1 and related quantities.
The computation of the variance estimates is handled in conditional_variance.py (moved from here).

The main strategy is to compute the quantities Xbar and Ybar described in my notes. Each of these are a
sum over prompts from the E and U batches respectively, and are computed as the gradients of auxiliarly
loss functions L_X and L_Y. In the end, \delta H_1 is simply \eta Xbar \cdot Ybar, where \eta is the learning rate.

Pretty straightforward, but I seemed to be having some issues with numerical precision, so I added some safeguards.

Another complicating factor is that we need to incorporate the preconditioning factors from the adam optimizer, since Ybar
is supposed to be the actual update direction, not just the raw gradient.

For now we only compute the first order term \delta H_1, but depending on how things go we might want to add the second order
term (eg. if I find in tests I am running that the entropy change deviates from linear in learning rate at learning rates of order the ones
I use in training)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import nullcontext
import os
import logging
from collections import defaultdict
import time
import math
from .conditional_variance import ConditionalVarianceEstimator
from sequence_processing.sequence_processor import SequenceProcessor, GenerationConfig, BatchedSequences


class BaselineState:
    """
    Maintains per-position residual means μ_j for the baseline b_j = H_j + μ_j.
    G_j - H_j is the target; we track an EMA per absolute step index j (0-based).
    """
    def __init__(self, ema_decay: float = 0.9, device: torch.device = torch.device("cpu")):
        self.ema_decay = float(ema_decay)
        self.mu = torch.zeros(0, device=device)  # shape [J_max]; grows as needed

    def ensure_len(self, J_needed: int):
        if self.mu.numel() < J_needed:
            pad = torch.zeros(J_needed - self.mu.numel(), device=self.mu.device)
            self.mu = torch.cat([self.mu, pad], dim=0)

    @torch.no_grad()
    def update_from_batch(self, residuals: torch.Tensor, lengths: torch.Tensor):
        """
        residuals: (B_total, T_max) padded with 0 where j >= length
        lengths:   (B_total,) number of valid steps per row
        """
        J_max = residuals.size(1)
        self.ensure_len(J_max)

        # Compute per-position batch means of residuals over valid rows
        # mask: 1 for j < length
        idx = torch.arange(J_max, device=lengths.device).unsqueeze(0)  # [1, J_max]
        mask = (idx < lengths.unsqueeze(1)).float()                    # [B, J_max]
        denom = mask.sum(dim=0).clamp_min(1.0)                         # [J_max]
        mean_resid = (residuals * mask).sum(dim=0) / denom             # [J_max]

        # EMA update μ_j ← (1-α) μ_j + α mean_resid_j
        α = self.ema_decay
        self.mu[:J_max].mul_(1.0 - α).add_(α * mean_resid)

    def get_mu_vector(self, J: int) -> torch.Tensor:
        self.ensure_len(J)
        return self.mu[:J].detach()  # [J], no grad


class ProbeComponents:
    """
    Core computational components for entropy probe analysis.
    
    This class handles the main gradient computation pipeline while avoiding
    storing parameter-sized vectors through clever use of scalar probe losses.
    """
    
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
            
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], logger: logging.Logger):
        self.model = model
        self.config = config  
        self.logger = logger
        self.device = next(model.parameters()).device
        
        # AMP settings to match training
        self.use_amp = config['memory_config']['amp']
        self.amp_dtype = getattr(torch, config['memory_config']['dtype'])
        
        # Sampling parameters (B_E passed as parameter, no instance variable)
        self.G = config['batch_config']['G']  # responses per prompt
        self.microbatch_size = config['memory_config']['microbatch_size']
        
        # Initialize tokenizer for conditional variance estimator
        if not hasattr(self, '_tokenizer'):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B", 
                trust_remote_code=True
            )
            self._tokenizer.padding_side = "left"
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
        # --- SequenceProcessor (Phase 1 sampling only) ---
        sp_cfg = self.config.get('generation', {})
        self._sp_gen_config = GenerationConfig(
            do_sample=True,
            num_return_sequences=self.G,
            max_new_tokens=sp_cfg.get('max_new_tokens', 256),
            temperature=sp_cfg.get('temperature', 1.0),
            top_p=sp_cfg.get('top_p', 1.0),
            gen_batch_size=sp_cfg.get('gen_batch_size', 8),
            tf_batch_size=sp_cfg.get('tf_batch_size', 64)
        )
        self._sequence_processor = SequenceProcessor(
            model=self.model,
            tokenizer=self._tokenizer,
            config=self._sp_gen_config
        )
            
        # Initialize conditional variance estimator
        self.variance_estimator = ConditionalVarianceEstimator(model, config, logger, self._tokenizer)
        
        # --- Canonical Parameter Registry (single source of truth) ---
        self._peft = self.model.module if hasattr(self.model, "module") else self.model

        # Trainable parameters from the same module that performs forward passes
        self._trainable_named = [
            (n, p) for (n, p) in self._peft.named_parameters() if p.requires_grad
        ]
        self._trainable_params = [p for _, p in self._trainable_named]
        self._trainable_ids    = {id(p) for p in self._trainable_params}

        # LoRA-only (plus lm_head if trainable) convenience view
        self._lora_named = [
            (n, p) for (n, p) in self._trainable_named
            if ("lora_a" in n.lower()) or ("lora_b" in n.lower()) or n.endswith("lm_head.weight")
        ]
        self._lora_params = [p for _, p in self._lora_named]

        self.logger.debug(f"[REGISTRY] trainable_tensors={len(self._trainable_params)} "
                          f"lora_tensors={len(self._lora_params)}")
        
        # --- Phase 3: Initialize BaselineState for RB-residual estimator ---
        bl_cfg = self.config.get('estimator', {}).get('baseline', {})
        self._baseline_state_x = BaselineState(
            ema_decay=bl_cfg.get('ema_decay', 0.9),
            device=self.device
        )
        
        self.logger.info(f"ProbeComponents initialized with SequenceProcessor: G={self.G}")
        
        # Log Phase 3 configuration
        x_estimator_mode = self.config.get('estimator', {}).get('x_estimator_mode', 'naive')
        baseline_mode = bl_cfg.get('mode', 'residual_mu')
        self.logger.info(f"Phase 3 config: x_estimator_mode={x_estimator_mode}, baseline_mode={baseline_mode}")
        
    def _to_batched_sequences_from_probe(self, batch: Dict[str, Any]) -> BatchedSequences:
        """
        Convert probe batch dict -> BatchedSequences required by SequenceProcessor.teacher_force_logprobs.
        """
        sequences = batch['sequences']            # [B, G, T]
        attention_masks = batch['attention_masks']# [B, G, T]
        prompt_lens = batch['prompt_lens']        # List[B], left-padded prompt length
        B, G, T = sequences.shape

        # gen_len[b][g] = sum(attn_mask[b,g]) - prompt_len[b]
        sums = attention_masks.long().sum(dim=-1)              # [B, G]
        gen_lens: List[List[int]] = []
        for b in range(B):
            gl = (sums[b].cpu().tolist())
            pl = int(prompt_lens[b])
            gen_lens.append([max(0, int(x) - pl) for x in gl])

        return BatchedSequences(
            sequences=sequences,
            attention_masks=attention_masks,
            prompt_lens=prompt_lens,
            gen_lens=gen_lens,
            responses_text=None  # not needed for TF
        )

    def _build_X_loss_rb_residual(
        self,
        logprob_results,  # LogprobResults from SequenceProcessor
        prompt_lens: List[int],
        normalize_by_length: bool = True,
    ) -> torch.Tensor:
        """
        Construct the scalar loss whose gradient equals the RB entropy gradient:
            sum_j (G_j - b_j) * ∇ log π(Y_j|prefix) + sum_k ∇ H_k(prefix)
        where b_j = H_j + μ_j (μ_j from BaselineState).
        Uses:
          - logprob_results.logprobs[b][g]: torch [T]
          - logprob_results.rb_entropies_torch[b][g]: torch [T]
        """
        assert logprob_results.rb_entropies_torch is not None, "rb_requires_grad=True + compute_rb=True required"

        B = len(logprob_results.logprobs)
        total_loss = torch.zeros((), device=next(self.model.parameters()).device)

        # Determine max per-seq gen length to pad positionwise residuals for EMA update
        lengths = []
        rb_list, lp_list = [], []
        for b in range(B):
            for g in range(len(logprob_results.logprobs[b])):
                lp = logprob_results.logprobs[b][g]              # [T]
                rb = logprob_results.rb_entropies_torch[b][g]    # [T]
                if lp is None or rb is None or lp.numel() == 0:
                    continue
                L = rb.numel()
                lengths.append(L)
                rb_list.append(rb)
                lp_list.append(lp)
        if len(lengths) == 0:
            return total_loss
        T_max = int(max(lengths))

        # Pad per-sample tensors to T_max for baseline update bookkeeping
        padded_residuals = []
        length_tensor = []
        for rb in rb_list:
            L = rb.numel()
            G = torch.cumsum(torch.flip(rb, dims=[0]), dim=0)    # [L], reverse cumsum
            G = torch.flip(G, dims=[0])
            resid = G - rb                                       # (G_j - H_j)
            if L < T_max:
                pad = torch.zeros(T_max - L, device=rb.device)
                resid = torch.cat([resid, pad], dim=0)
            padded_residuals.append(resid.unsqueeze(0))
            length_tensor.append(L)
        padded_residuals = torch.cat(padded_residuals, dim=0)    # [N_seq, T_max]
        length_tensor = torch.tensor(length_tensor, device=padded_residuals.device, dtype=torch.long)  # [N_seq]

        # Update EMA baseline μ_j using current batch residuals
        self._baseline_state_x.update_from_batch(padded_residuals, length_tensor)
        mu_vec = self._baseline_state_x.get_mu_vector(T_max)     # [T_max], detached

        # Assemble losses per sequence
        idx_seq = 0
        for b in range(B):
            for g in range(len(logprob_results.logprobs[b])):
                token_lp = logprob_results.logprobs[b][g]             # [T]
                Hk = logprob_results.rb_entropies_torch[b][g]         # [T]
                if token_lp is None or Hk is None or token_lp.numel() == 0:
                    continue
                L = Hk.numel()

                # Returns-to-go G_j from RB entropies
                G = torch.cumsum(torch.flip(Hk, dims=[0]), dim=0)
                G = torch.flip(G, dims=[0])                           # [L]

                # Baseline b_j = H_j + μ_j (positionwise; slice μ to length L)
                bl_mode = (self.config.get('estimator', {}).get('baseline', {}) or {}).get('mode', 'residual_mu')
                if bl_mode == 'residual_mu':
                    mu = mu_vec[:L]                                   # [L]
                    b_j = Hk.detach() + mu                            # detach baseline
                elif bl_mode == 'none':
                    b_j = Hk.detach()  # μ=0
                else:
                    # "constant" or "regression" can be added later; for now treat as residual_mu
                    mu = mu_vec[:L]
                    b_j = Hk.detach() + mu

                # Advantages for the score term
                adv = (G.detach() - b_j).detach()                     # [L]

                # --- Score term with correct sign: grad = Σ adv * ∇ log π ---
                score_loss = torch.dot(adv, token_lp)                 # scalar

                # --- Pathwise term: Σ H_k (no detach, to get ∂H) ---
                pathwise_loss = Hk.sum()

                if normalize_by_length and L > 0:
                    loss = (score_loss + pathwise_loss) / float(L)
                else:
                    loss = score_loss + pathwise_loss

                total_loss = total_loss + loss
                idx_seq += 1

        return total_loss
        
    def sample_batch(self, B: int, G: int, indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Sample batch of B prompts, each with G responses.
        
       
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
            
        # Generate responses using SequenceProcessor
        all_sequences = []
        all_prompt_lens = []
        all_advantages = []
        max_lengths = []
        all_attention_masks = []

        rollout_batch_size = self.config.get('batch_config', {}).get('rollout_batch_size', 8)
        self.logger.info(f"Using SequenceProcessor with rollout_batch_size={rollout_batch_size}")

        for batch_start in range(0, len(prompts), rollout_batch_size):
            batch_end = min(batch_start + rollout_batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_examples = sampled_examples[batch_start:batch_end]
            batch_prompt_ids = prompt_ids[batch_start:batch_end]

            # Generate G responses per prompt using SequenceProcessor
            sequences_pack = self._sequence_processor.generate_batched(
                prompts=batch_prompts, G=G, gen_batch_size=self._sp_gen_config.gen_batch_size
            )

            B_b = len(batch_prompts)
            sequences_b = sequences_pack.sequences            # [B_b, G, T]
            attn_b      = sequences_pack.attention_masks      # [B_b, G, T]
            prompt_lens_b = sequences_pack.prompt_lens        # [B_b]
            gen_lens_b    = sequences_pack.gen_lens           # List[List[int]]
            responses_b   = sequences_pack.responses_text     # List[List[str]]

            # Compute rewards/advantages for each prompt
            for local_b in range(B_b):
                example   = batch_examples[local_b]
                prompt_id = batch_prompt_ids[local_b]
                responses = responses_b[local_b]              # List[str], length G

                # Compute advantages: A = R - mean(R) per prompt (DR-GRPO style)
                rewards = self._compute_rewards(prompt_id, responses, example)
                advantages = rewards - rewards.mean()
                all_advantages.append(advantages)

                # Track L_max per prompt for probe loss normalization
                L_max = max(gen_lens_b[local_b]) if len(gen_lens_b[local_b]) > 0 else 1
                max_lengths.append(L_max)

            # Store sequences and masks for probe teacher forcing passes
            all_sequences.extend([sequences_b[i] for i in range(B_b)])
            all_attention_masks.extend([attn_b[i] for i in range(B_b)])
            all_prompt_lens.extend(prompt_lens_b)

        # Pad sequences to same max length across all prompts
        max_seq_len = max(seq.shape[1] for seq in all_sequences)
        padded_sequences, padded_masks = [], []
        for b in range(len(all_sequences)):
            seq = all_sequences[b]  # [G, seq_len]
            mask = all_attention_masks[b]  # [G, seq_len]
            if seq.shape[1] < max_seq_len:
                pad_len = max_seq_len - seq.shape[1]
                seq  = F.pad(seq,  (0, pad_len), value=self._tokenizer.pad_token_id)
                mask = F.pad(mask, (0, pad_len), value=0)
            padded_sequences.append(seq)
            padded_masks.append(mask)

        sequences = torch.stack(padded_sequences, dim=0)   # [B, G, max_len]
        attention_masks = torch.stack(padded_masks, dim=0) # [B, G, max_len]
        advantages = torch.stack(all_advantages, dim=0)    # [B, G]

        # Sanity checks (temporary; remove once stable)
        B_chk, G_chk, T_chk = sequences.shape
        assert B_chk == B and G_chk == G
        assert len(all_prompt_lens) == B_chk
        assert len(max_lengths) == B_chk
        assert attention_masks.shape == (B_chk, G_chk, T_chk)
        self.logger.info(f"[SP→Probe] batch shapes OK: sequences={tuple(sequences.shape)}")

        batch_data = {
            'sequences': sequences,            # [B, G, max_len] - for teacher forcing with gradients
            'prompt_lens': all_prompt_lens,    # [B] - prompt lengths
            'advantages': advantages,          # [B, G] - advantages A_{n,g}
            'max_lengths': max_lengths,        # [B] - L_max per prompt
            'attention_masks': attention_masks,# [B, G, max_len]
            'prompt_ids': prompt_ids,          # [B] - original dataset indices for each prompt
            'num_prompts': B,
            'num_responses_per_prompt': G
        }
        
        self.logger.info(f"Successfully sampled batch: {B} prompts × {G} responses")
        return batch_data

    def _get_dataset(self, split: str = "test"):
        """Get dataset for sampling."""
        from rlp_datasets import DATASET_REGISTRY
        dataset = DATASET_REGISTRY[self.config['batch_config']['dataset_name']]
        return dataset(split)

    def _build_prompt_from_example(self, example):
        """Build prompt string from dataset example."""
        return example.question

    def sample_E_batch_with_replacement(self, E_total_sequences: int, G: int) -> Dict[str, Any]:
        """
        Sample E batch with replacement sampling for entropy estimation.
        
        Returns a probe-compatible batch dict with ~E_total_sequences total generations,
        using prompt sampling WITH replacement. Shapes:
          sequences:       [B_E, G, max_len]
          attention_masks: [B_E, G, max_len]
          prompt_lens:     List[B_E]
          advantages:      [B_E, G]  (centered per prompt)
          max_lengths:     List[B_E] (max gen_len per prompt)
          prompt_ids:      List[B_E]
        """
        import numpy as np
        
        ds = self._get_dataset(split=self.config['batch_config']['split'])
        N = len(ds)
        G = int(G)

        # How many prompts do we need if each yields G generations?
        B_E = int(np.ceil(E_total_sequences / G))
        
        self.logger.info(f"E-batch replacement sampling: {E_total_sequences} total sequences, {B_E} prompts × {G} responses")

        # Sample indices WITH replacement
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility during testing
        idx = rng.integers(low=0, high=N, size=B_E, endpoint=False).tolist()

        prompts, examples, prompt_ids = [], [], []
        for i in idx:
            ex = ds[i]
            prompts.append(self._build_prompt_from_example(ex))
            examples.append(ex)
            prompt_ids.append(i)

        # Use rollout batching like Phase 1
        rollout_bs = self.config.get('batch_config', {}).get('rollout_batch_size', 8)
        self.logger.info(f"E-batch using rollout_batch_size={rollout_bs}")

        all_sequences, all_masks = [], []
        all_prompt_lens, all_advantages, max_lengths = [], [], []

        for s in range(0, B_E, rollout_bs):
            e = min(s + rollout_bs, B_E)
            pack = self._sequence_processor.generate_batched(
                prompts=prompts[s:e], G=G, gen_batch_size=self._sp_gen_config.gen_batch_size
            )
            B_b = e - s
            seq_b, mask_b = pack.sequences, pack.attention_masks
            prompt_lens_b, gen_lens_b, responses_b = pack.prompt_lens, pack.gen_lens, pack.responses_text

            # Compute rewards -> advantages per prompt (unchanged from Phase 1)
            for b_local in range(B_b):
                ex = examples[s + b_local]
                pid = prompt_ids[s + b_local]
                resp = responses_b[b_local]
                rewards = self._compute_rewards(pid, resp, ex)
                adv = rewards - rewards.mean()

                all_advantages.append(adv)
                L_max = max(gen_lens_b[b_local]) if len(gen_lens_b[b_local]) > 0 else 1
                max_lengths.append(L_max)

            all_sequences.extend([seq_b[i] for i in range(B_b)])
            all_masks.extend([mask_b[i] for i in range(B_b)])
            all_prompt_lens.extend(prompt_lens_b)

        # Pad to common max length
        max_T = max(seq.shape[1] for seq in all_sequences)
        pad_id = self._tokenizer.pad_token_id
        padded_seq, padded_mask = [], []
        for b in range(len(all_sequences)):
            seq = all_sequences[b]
            msk = all_masks[b]
            if seq.shape[1] < max_T:
                pad_len = max_T - seq.shape[1]
                seq = F.pad(seq, (0, pad_len), value=pad_id)
                msk = F.pad(msk, (0, pad_len), value=0)
            padded_seq.append(seq)
            padded_mask.append(msk)

        sequences = torch.stack(padded_seq, dim=0)
        attention_masks = torch.stack(padded_mask, dim=0)
        advantages = torch.stack(all_advantages, dim=0)   # [B_E, G]

        batch_data = {
            'sequences': sequences,
            'attention_masks': attention_masks,
            'prompt_lens': all_prompt_lens,
            'advantages': advantages,
            'max_lengths': max_lengths,
            'prompt_ids': prompt_ids,
            'num_prompts': len(all_prompt_lens),
            'num_responses_per_prompt': G,
        }

        self.logger.info(f"E-batch replacement complete: {len(prompt_ids)} prompts, duplicates: {len(prompt_ids) - len(set(prompt_ids))}")
        return batch_data

    def sample_U_batch_distinct(self, B_U: int, G: int) -> Dict[str, Any]:
        """
        Sample U batch with distinct sampling (without replacement) for gradient updates.
        
        Returns a probe-compatible batch dict with B_U distinct prompts, each with G responses.
        Mirrors the current U batch approach but routed through SequenceProcessor.
        """
        import numpy as np
        
        ds = self._get_dataset(split=self.config['batch_config']['split'])
        N = len(ds)
        G = int(G)
        B_U = int(B_U)
        
        self.logger.info(f"U-batch distinct sampling: {B_U} distinct prompts × {G} responses")
        
        # Sample indices WITHOUT replacement  
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility during testing
        if B_U > N:
            self.logger.warning(f"Requested {B_U} distinct prompts but dataset only has {N} examples. Using all available.")
            B_U = N
        
        idx = rng.choice(N, size=B_U, replace=False).tolist()

        prompts, examples, prompt_ids = [], [], []
        for i in idx:
            ex = ds[i]
            prompts.append(self._build_prompt_from_example(ex))
            examples.append(ex)
            prompt_ids.append(i)

        # Use rollout batching like Phase 1
        rollout_bs = self.config.get('batch_config', {}).get('rollout_batch_size', 8)
        self.logger.info(f"U-batch using rollout_batch_size={rollout_bs}")

        all_sequences, all_masks = [], []
        all_prompt_lens, all_advantages, max_lengths = [], [], []

        for s in range(0, B_U, rollout_bs):
            e = min(s + rollout_bs, B_U)
            pack = self._sequence_processor.generate_batched(
                prompts=prompts[s:e], G=G, gen_batch_size=self._sp_gen_config.gen_batch_size
            )
            B_b = e - s
            seq_b, mask_b = pack.sequences, pack.attention_masks
            prompt_lens_b, gen_lens_b, responses_b = pack.prompt_lens, pack.gen_lens, pack.responses_text

            # Compute rewards -> advantages per prompt (unchanged from Phase 1)
            for b_local in range(B_b):
                ex = examples[s + b_local]
                pid = prompt_ids[s + b_local]
                resp = responses_b[b_local]
                rewards = self._compute_rewards(pid, resp, ex)
                adv = rewards - rewards.mean()

                all_advantages.append(adv)
                L_max = max(gen_lens_b[b_local]) if len(gen_lens_b[b_local]) > 0 else 1
                max_lengths.append(L_max)

            all_sequences.extend([seq_b[i] for i in range(B_b)])
            all_masks.extend([mask_b[i] for i in range(B_b)])
            all_prompt_lens.extend(prompt_lens_b)

        # Pad to common max length
        max_T = max(seq.shape[1] for seq in all_sequences)
        pad_id = self._tokenizer.pad_token_id
        padded_seq, padded_mask = [], []
        for b in range(len(all_sequences)):
            seq = all_sequences[b]
            msk = all_masks[b]
            if seq.shape[1] < max_T:
                pad_len = max_T - seq.shape[1]
                seq = F.pad(seq, (0, pad_len), value=pad_id)
                msk = F.pad(msk, (0, pad_len), value=0)
            padded_seq.append(seq)
            padded_mask.append(msk)

        sequences = torch.stack(padded_seq, dim=0)
        attention_masks = torch.stack(padded_mask, dim=0)
        advantages = torch.stack(all_advantages, dim=0)   # [B_U, G]

        batch_data = {
            'sequences': sequences,
            'attention_masks': attention_masks,
            'prompt_lens': all_prompt_lens,
            'advantages': advantages,
            'max_lengths': max_lengths,
            'prompt_ids': prompt_ids,
            'num_prompts': len(all_prompt_lens),
            'num_responses_per_prompt': G,
        }

        self.logger.info(f"U-batch distinct complete: {len(prompt_ids)} unique prompts (no duplicates)")
        return batch_data
        
    def _get_stop_processor(self):
        """Get stopping processor for generation."""
        from evals.utils_io import StopAfterAnswer
        from transformers import LogitsProcessorList
        return LogitsProcessorList([StopAfterAnswer(self._tokenizer)])
        
        
        
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
            lr = float(self.config['learning_rate'])  # Ensure it's a float
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
        Create zero buffers matching the canonical trainable parameters.
        """
        param_buffers = {}
        for p in self._trainable_params:
            param_buffers[id(p)] = torch.zeros_like(p, dtype=dtype, device=device)
        return param_buffers
    
    def add_into_param_buffer(self, buf: Dict[int, torch.Tensor], scale: float = 1.0) -> None:
        """
        Add current param.grad into buffer with optional scaling.
        Scaling is applied in the buffer's dtype/device (typically CPU fp32) to
        avoid bf16/fp16 roundoff when converting average->sum.
        """
        for p in self._trainable_params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in buf:
                continue
            g = p.grad.detach()
            want = buf[pid]
            if want.device != g.device or want.dtype != g.dtype:
                g = g.to(want.device, dtype=want.dtype)
            want.add_(g, alpha=float(scale))
    
    def scale_param_gradients(self, scale_factor: float):
        """
        (Optional) Keep for other call sites, but the probe accumulation
        should no longer depend on this; we scale inside add_into_param_buffer
        in fp32/CPU.
        """
        for p in self._trainable_params:
            if p.grad is not None:
                p.grad.data.mul_(float(scale_factor))
    
    def dot_param_buffers(self, buf_a: Dict[int, torch.Tensor], buf_b: Dict[int, torch.Tensor]) -> float:
        """
        Compute ⟨buf_a, buf_b⟩ with high-precision accumulation.
        Buffers can remain fp32; we upcast on-the-fly for the product/sum.
        """
        total = torch.zeros((), dtype=torch.float64)
        for p in self._trainable_params:
            pid = id(p)
            ta = buf_a.get(pid); tb = buf_b.get(pid)
            if ta is None or tb is None:
                continue
            # High-precision multiply-and-sum to reduce batch-size sensitivity
            total += (ta.double() * tb.double()).sum()
        return float(total.item())
    
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
            
            # L_X for this prompt: - mean_g(coeff * S)
            prompt_loss = -(coeff_b * S_b).mean()
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
        Phase 3: Accumulate ΣX (entropy gradients) from evaluation batch E.
        Supports both naive and RB-residual estimators.
        
        Args:
            E_batch: Evaluation batch dictionary
            mb_size_prompts: Microbatch size (number of prompts per microbatch)
            weighting_mode: Weighting mode for X-loss (used in naive mode)
            
        Returns:
            (sum_X_buf, B_local): Parameter sum buffer and local batch size
        """
        self._assert_grad_context("accumulate_sum_X")
        sum_X_buf = self.zeros_like_params(dtype=torch.float32, device='cpu')
        
        # Phase 3: Check estimator mode
        mode = (self.config.get('estimator', {}) or {}).get('x_estimator_mode', 'naive')
        normalize_by_length = bool(self.config.get('estimator', {}).get('rb_normalize_by_length', True))
        
        self.logger.debug(f"X accumulation mode: {mode}, normalize_by_length: {normalize_by_length}")
        
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
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    mb_prompt_count = len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    mb_prompt_count = microbatch['advantages'].shape[0]
                else:
                    mb_prompt_count = len(microbatch.get('max_lengths', []))
                
                # Phase 3: Branch on estimator mode
                if mode == 'rb_residual':
                    # New RB-residual path
                    # 1) Convert microbatch -> BatchedSequences for SP TF with grad
                    bs = self._to_batched_sequences_from_probe(microbatch)
                    
                    # 2) Run TF with grad + RB
                    logprob_results, _ = self._sequence_processor.teacher_force_logprobs(
                        sequences=bs, with_grad=True,
                        tf_batch_size=self._sp_gen_config.tf_batch_size,
                        compute_rb=True
                    )
                    
                    # 3) Build RB X loss
                    L_X_mb = self._build_X_loss_rb_residual(
                        logprob_results=logprob_results,
                        prompt_lens=bs.prompt_lens,
                        normalize_by_length=normalize_by_length
                    )
                    
                else:
                    # Fallback to existing naive path
                    # Forward pass with teacher forcing
                    S_dict = self._teacher_force_logprobs(microbatch)
                    
                    # Build X-loss with detached LOO coefficient
                    L_X_mb = self.build_LX_from_S(S_dict, weighting_mode)
                
                # Backward pass - populates param.grad with X gradients  
                L_X_mb.backward()

                # Optional invariant: count trainable grads present
                present = sum(int(p.grad is not None and p.grad.detach().abs().sum() > 0) for p in self._trainable_params)
                self.logger.debug(f"[X-{mode}] nonzero param.grads = {present}/{len(self._trainable_params)}")

                # Scale in CPU/fp32 during accumulation to avoid bf16 multiply
                # Both paths return average over prompts, but we need sum
                self.add_into_param_buffer(sum_X_buf, scale=float(mb_prompt_count))
                
                B_local += mb_prompt_count
                
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
        self._assert_grad_context("accumulate_sum_Y")
        params = self._trainable_params   # canonical
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
                
                # Count prompts in this microbatch
                if 'sequences' in microbatch:
                    mb_prompt_count = len(microbatch['sequences'])
                elif hasattr(microbatch.get('advantages'), 'shape'):
                    mb_prompt_count = microbatch['advantages'].shape[0]
                else:
                    mb_prompt_count = len(microbatch.get('max_lengths', []))
                
                # Backward pass - populates param.grad with raw ∇J
                L_Y_mb.backward()

                # Optional invariant
                present = sum(int(p.grad is not None and p.grad.detach().abs().sum() > 0) for p in params)
                self.logger.debug(f"[Y] raw nonzero param.grads = {present}/{len(params)}")

                # Adam preconditioning in-place, with detailed debugging
                self.logger.debug(f"[Y] Starting Adam preconditioning for {len(params)} parameters")
                zero_exp_avg_sq_count = 0
                tiny_exp_avg_sq_count = 0
                zero_output_count = 0
                
                for p in params:
                    g = p.grad
                    if g is None:
                        continue
                        
                    # Debug statistics before preconditioning
                    g_norm = g.detach().norm().item()
                    
                    try:
                        # Check Adam state statistics
                        state = adam_preconditioner.optimizer.state.get(p, {})
                        exp_avg_sq = state.get('exp_avg_sq', None)
                        
                        if exp_avg_sq is not None:
                            exp_avg_sq_norm = exp_avg_sq.norm().item()
                            exp_avg_sq_max = exp_avg_sq.max().item()
                            exp_avg_sq_min = exp_avg_sq.min().item()
                            
                            if exp_avg_sq_max == 0.0:
                                zero_exp_avg_sq_count += 1
                            elif exp_avg_sq_max < 1e-10:
                                tiny_exp_avg_sq_count += 1
                                
                            self.logger.debug(f"[Y] param {id(p)}: g_norm={g_norm:.2e}, "
                                            f"exp_avg_sq range=[{exp_avg_sq_min:.2e}, {exp_avg_sq_max:.2e}], "
                                            f"exp_avg_sq_norm={exp_avg_sq_norm:.2e}")
                        else:
                            self.logger.debug(f"[Y] param {id(p)}: g_norm={g_norm:.2e}, no exp_avg_sq state")
                        
                        # Apply preconditioning in fp32
                        g32 = g.to(torch.float32)
                        gtilde = adam_preconditioner.apply_preconditioner(g32, p)
                        if gtilde is None:
                            gtilde = g32
                            
                        # Check output statistics
                        gtilde_norm = gtilde.norm().item()
                        if gtilde_norm == 0.0:
                            zero_output_count += 1
                            
                        self.logger.debug(f"[Y] param {id(p)}: gtilde_norm={gtilde_norm:.2e} "
                                        f"(ratio={gtilde_norm/max(g_norm, 1e-20):.2e})")
                        
                        # Copy back (will downcast if p.grad is bf16)
                        g.copy_(gtilde)
                        
                    except Exception as e:
                        self.logger.warning(f"[Y] preconditioner failed on {id(p)}: {e}; using raw grad")
                        
                self.logger.info(f"[Y] Adam preconditioning complete: {zero_exp_avg_sq_count} zero exp_avg_sq, "
                               f"{tiny_exp_avg_sq_count} tiny exp_avg_sq, {zero_output_count} zero outputs")
                
                # Accumulate with fp32 scaling (average -> sum)
                self.add_into_param_buffer(sum_Y_buf, scale=float(mb_prompt_count))
                
                B_local += mb_prompt_count
                
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
        
        # 🔧 FIX: Apply same PEFT adapter activation as conditional_variance.py
        # Resolve the PEFT-wrapped module regardless of DDP
        peft_model = self._peft
        
        # Make sure an adapter is active and trainable
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")  # or your adapter name
        
        # Sanity: we are not inside a disable_adapter() context
        active = getattr(peft_model, "active_adapter", None)
        if active is None:
            raise RuntimeError("No active adapter on PEFT model in main teacher forcing.")
            
        # 🔧 FIX: Add invariant logging for main probe phases
        self.logger.debug(f"🔧 [MAIN-TF-DEBUG] active_adapter={active} training={peft_model.training}")
        trainable = [n for n,p in peft_model.named_parameters() if p.requires_grad]
        self.logger.debug(f"🔧 [MAIN-TF-DEBUG] #trainable={len(trainable)} sample={trainable[:8]}")
        
        # 🔧 FIX: Prove LoRA is active at runtime in main phases too
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
            self.logger.debug(f"🔧 [MAIN-TF-DEBUG] LoRA on/off delta: {delta:.2e}")
            if delta < 1e-7:
                raise RuntimeError("LoRA appears inactive in main TF: on/off logits nearly identical.")
        
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
            
            # 🔧 FIX: Assert S has gradient path in main phases
            assert S.requires_grad and S.grad_fn is not None, "S lost grad path in main TF (inference/no_grad or detach?)"
            self.logger.debug(f"🔧 [MAIN-TF-DEBUG] S.requires_grad={S.requires_grad} S.grad_fn={S.grad_fn}")
            
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
            [G] tensor of log-probabilities with gradients enabled
        """
        # Use config for microbatch size
        teacher_force_microbatch_size = self.config.get('memory_config', {}).get('teacher_force_microbatch_size', 2)
        self.logger.debug(f"Using teacher_force_microbatch_size={teacher_force_microbatch_size} for gradient computation")
        
        G = sequences.shape[0]
        all_logprobs = []
        
        # Process in microbatches to avoid GPU memory issues
        for g_start in range(0, G, teacher_force_microbatch_size):
            g_end = min(g_start + teacher_force_microbatch_size, G)
            micro_sequences = sequences[g_start:g_end]  # [g_micro, total_len]
            
            # Create attention mask (avoid pad tokens)
            if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                attention_mask = (micro_sequences != self._tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(micro_sequences)
            
            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(micro_sequences, attention_mask=attention_mask).logits  # [g_micro, total_len, vocab_size]
            
            # Apply temperature scaling
            temp = self.config['generation'].get('temperature', 1.0)
            logits = logits / temp
            
            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)  # [g_micro, total_len, vocab_size]
            
            # Extract log probabilities for generated tokens (skip prompt)
            for g_idx in range(len(micro_sequences)):
                seq = micro_sequences[g_idx]  # [total_len]
                gen_ids = seq[prompt_len:]  # [gen_len]
                gen_logits_seq = log_probs[g_idx, prompt_len:prompt_len+len(gen_ids)]  # [gen_len, vocab_size]
                
                # Gather log probabilities for actual generated token IDs
                # gen_ids: [gen_len], gen_logits_seq: [gen_len, vocab_size] 
                seq_logprobs = gen_logits_seq.gather(1, gen_ids.unsqueeze(1)).squeeze(1)  # [gen_len]
                
                # Sum to get total sequence log probability
                # Apply attention mask to avoid including pad token probabilities
                if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                    gen_mask = (gen_ids != self._tokenizer.pad_token_id).float()
                    masked_logprobs = seq_logprobs * gen_mask
                    total_logprob = masked_logprobs.sum()
                else:
                    total_logprob = seq_logprobs.sum()
                    
                all_logprobs.append(total_logprob)
        
        # Stack all log probabilities into tensor with gradients
        return torch.stack(all_logprobs)  # [G]

    def compute_conditional_variance_over_E(self, E_batch, muY_buf,
    mb_size_prompts, weighting_mode):
        return self.variance_estimator.compute_conditional_variance_over_E(
            E_batch, muY_buf, mb_size_prompts, weighting_mode
        )