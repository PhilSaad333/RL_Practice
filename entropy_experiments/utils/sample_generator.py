"""Batch sampling utilities for Fisher-kernel experiments.

This module centralizes construction of richly annotated sequence batches used
for Fisher-kernel analysis and entropy diagnostics. It wraps the shared
SequenceProcessor while keeping the public API focused on high-level sampling
use cases (update batches, evaluation batches, custom prompts, etc.).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from entropy_experiments.utils.model_loader import (
    load_adam_optimizer_from_path,
    load_peft_for_probe,
)
from entropy_experiments.utils.sequence_processor import (
    SequenceProcessor,
    BatchedSequences,
    LogprobResults,
    DiagnosticsResults,
)

try:  # Optional helper for templating custom prompts
    from rlp_datasets.gsm8k_r1_template import prompt_template as gsm8k_prompt_template
except ImportError:  # pragma: no cover - fallback when datasets not installed
    gsm8k_prompt_template = None


@dataclass
class SequenceRecord:
    """Rich metadata for a single prompt/response pair."""

    sequence_id: str
    prompt_text: str
    response_text: str
    prompt_tokens: Sequence[int]
    response_tokens: Sequence[int]
    logprob_per_token: Optional[Sequence[float]] = None
    logq_per_token: Optional[Sequence[float]] = None
    entropy_per_token: Optional[Sequence[float]] = None
    total_logprob: Optional[float] = None
    total_logq: Optional[float] = None
    reward: Optional[float] = None
    advantage: Optional[float] = None
    global_prompt_id: Optional[int] = None
    dataset_split: Optional[str] = None
    prompt_batch_idx: Optional[int] = None
    completion_idx: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedBatch:
    """Container for a collection of SequenceRecord entries plus tensors."""

    batch_type: str
    sequences: List[SequenceRecord] = field(default_factory=list)
    full_sequence_tensor: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    prompt_lens: Optional[List[int]] = None
    gen_lens: Optional[List[List[int]]] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    sampling_metadata: Dict[str, Any] = field(default_factory=dict)

    def sequence_ids(self) -> List[str]:
        """Return the list of stable sequence identifiers."""

        return [record.sequence_id for record in self.sequences]

    def to_serializable(self) -> Dict[str, Any]:
        """Convert batch contents into JSON-friendly structures."""

        raise NotImplementedError("Serialization will be implemented in a later pass")


class SampleGenerator:
    """High-level interface for constructing batches for Fisher-kernel studies."""

    def __init__(self, config: Dict[str, Any], *, logger: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer: Optional[Any] = None
        self._sequence_processor: Optional[SequenceProcessor] = None
        self._optimizer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------
    def _lazy_load_resources(self) -> None:
        """Load model/tokenizer and initialize SequenceProcessor on demand."""

        if self._sequence_processor is not None:
            return

        checkpoint_cfg = (self.config.get("checkpoint") or {}).copy()
        adapter_path = checkpoint_cfg.get("checkpoint_path") or checkpoint_cfg.get("adapter_path")
        backbone = checkpoint_cfg.get("backbone")
        device_map = checkpoint_cfg.get("device_map") or self.config.get("device") or "cuda"

        if adapter_path is None:
            raise ValueError("SampleGenerator requires 'checkpoint.checkpoint_path' (or adapter_path) to load the model.")
        if backbone is None:
            raise ValueError("SampleGenerator requires 'checkpoint.backbone' to load the model architecture.")

        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load_peft_for_probe(
                base_id=backbone,
                adapter_path=adapter_path,
                device_map=device_map,
                force_fp32_runtime=True,
            )

        sp_config = self._build_sequence_processor_config()
        self._sequence_processor = SequenceProcessor(
            self._model,
            self._tokenizer,
            logger=self.logger,
            config=sp_config,
        )

        optimizer_path = checkpoint_cfg.get("optimizer_path")
        if optimizer_path and self._optimizer is None:
            self._optimizer = load_adam_optimizer_from_path(
                self._model,
                optimizer_path=optimizer_path,
            )

    def _build_sequence_processor_config(self) -> Dict[str, Any]:
        """Return the configuration payload passed to SequenceProcessor."""

        gen_cfg = (self.config.get("generation") or {}).copy()
        gen_cfg.setdefault("max_new_tokens", 256)
        gen_cfg.setdefault("temperature", 1.0)
        gen_cfg.setdefault("top_p", 1.0)

        precision_cfg = self._sequence_processor_precision_config()
        return {
            "generation": gen_cfg,
            "precision": precision_cfg,
        }

    def _sequence_processor_precision_config(self) -> Dict[str, Any]:
        """Translate precision section of the config for SequenceProcessor."""

        prec = (self.config.get("precision") or {}).copy()
        runtime_dtype = str(prec.get("runtime_dtype", "float32")).lower()
        entropy_dtype = str(prec.get("entropy_dtype", "float32")).lower()

        return {
            "allow_tf32": bool(prec.get("allow_tf32", False)),
            "matmul_precision": prec.get("matmul_precision", "high"),
            "func_override": {
                "autocast": False,
                "dtype": runtime_dtype,
                "cast_params": True,
            },
            "tf_nograd": {
                "autocast": False,
                "dtype": runtime_dtype,
                "cast_logits_fp32": True,
            },
            "entropy_fp64": (entropy_dtype == "float64"),
        }

    def _compute_rb_flag(self) -> bool:
        generation_cfg = (self.config.get("generation") or {})
        return bool(generation_cfg.get("compute_rb", False))

    def _generate_sequences(
        self,
        prompts: List[str],
        examples: Optional[List[Any]],
        completions_per_prompt: int,
        *,
        dataset_split: str,
        compute_rb: bool,
    ) -> Tuple[BatchedSequences, LogprobResults, DiagnosticsResults]:
        assert self._sequence_processor is not None
        processor = self._sequence_processor
        original_examples = getattr(processor, "_temp_examples", None)
        if examples:
            processor._temp_examples = examples  # type: ignore[attr-defined]
        try:
            return processor.generate_with_logprobs(
                prompts=prompts,
                G=completions_per_prompt,
                dataset_name=None,
                split=dataset_split,
                num_prompts=None,
                seed=None,
                with_grad=False,
                compute_rb=compute_rb,
            )
        finally:
            if examples:
                if original_examples is not None:
                    processor._temp_examples = original_examples  # type: ignore[attr-defined]
                elif hasattr(processor, "_temp_examples"):
                    delattr(processor, "_temp_examples")

    def _build_generated_batch(
        self,
        *,
        batch_type: str,
        sequence_id_prefix: str,
        prompts: List[str],
        prompt_metadata: Optional[List[Dict[str, Any]]],
        sequences: BatchedSequences,
        logprob_results: LogprobResults,
        diagnostics: DiagnosticsResults,
        completions_per_prompt: int,
        dataset_name: str,
        dataset_split: str,
        seed: Optional[int],
        sampling_extra: Dict[str, Any],
    ) -> GeneratedBatch:
        B = len(prompts)
        if prompt_metadata is not None and len(prompt_metadata) != B:
            raise ValueError("prompt_metadata length must match number of prompts")

        prompt_meta_list = [
            (meta.copy() if meta is not None else {})
            for meta in (prompt_metadata or [{} for _ in range(B)])
        ]

        rewards_data = getattr(logprob_results, "rewards", None) or []
        if rewards_data and len(rewards_data) == B:
            rewards_tensor = torch.tensor(rewards_data, dtype=torch.float32)
            if rewards_tensor.ndim == 1:
                rewards_tensor = rewards_tensor.unsqueeze(1)
        else:
            rewards_tensor = torch.zeros((B, completions_per_prompt), dtype=torch.float32)
        advantages_tensor = rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)

        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        sequences_tensor = sequences.sequences
        prompt_tokens_per_prompt: List[List[int]] = []
        for idx in range(B):
            prompt_len = sequences.prompt_lens[idx]
            slice_tensor = sequences_tensor[idx, 0, :prompt_len].detach().cpu().reshape(-1)
            token_slice = slice_tensor.tolist()
            if pad_token_id is not None:
                prompt_tokens = [int(tok) for tok in token_slice if tok != pad_token_id]
            else:
                prompt_tokens = [int(tok) for tok in token_slice]
            prompt_tokens_per_prompt.append(prompt_tokens)
        diagnostics_list = getattr(diagnostics, "diagnostics", None)
        rb_entropies = getattr(logprob_results, "rb_entropies", None)
        token_logqs = getattr(logprob_results, "token_logqs", None)
        sequence_logqs = getattr(logprob_results, "sequence_logqs", None)

        records: List[SequenceRecord] = []
        for prompt_idx, prompt_text in enumerate(prompts):
            prompt_meta = prompt_meta_list[prompt_idx]
            meta_global_id = prompt_meta.get("global_prompt_id")

            for completion_idx in range(completions_per_prompt):
                sequence_id = f"{sequence_id_prefix}-{prompt_idx:03d}-{completion_idx:02d}"
                prompt_len = sequences.prompt_lens[prompt_idx]
                gen_len = sequences.gen_lens[prompt_idx][completion_idx]
                full_tokens = sequences_tensor[prompt_idx, completion_idx].detach().cpu()
                response_tokens = full_tokens[prompt_len: prompt_len + gen_len].tolist()

                logprob_tensor = logprob_results.logprobs[prompt_idx][completion_idx]
                logprob_per_token = (
                    logprob_tensor.detach().cpu().tolist()
                    if torch.is_tensor(logprob_tensor)
                    else list(logprob_tensor)
                )

                if token_logqs:
                    logq_tensor = token_logqs[prompt_idx][completion_idx]
                    logq_per_token = (
                        logq_tensor.detach().cpu().tolist()
                        if torch.is_tensor(logq_tensor)
                        else list(logq_tensor)
                    )
                else:
                    logq_per_token = None

                entropy_array = logprob_results.entropies[prompt_idx][completion_idx]
                entropy_per_token = (
                    entropy_array.tolist()
                    if hasattr(entropy_array, "tolist")
                    else list(entropy_array)
                ) if entropy_array is not None else None

                rb_entry = None
                if rb_entropies:
                    rb_candidate = rb_entropies[prompt_idx][completion_idx]
                    if rb_candidate is not None:
                        rb_entry = (
                            rb_candidate.tolist()
                            if hasattr(rb_candidate, "tolist")
                            else list(rb_candidate)
                        )

                diag_summary = None
                if diagnostics_list:
                    diag_pack = diagnostics_list[prompt_idx][completion_idx]
                    if diag_pack is not None and getattr(diag_pack, "seq", None) is not None:
                        try:
                            diag_summary = asdict(diag_pack.seq)
                        except Exception:  # pragma: no cover - defensive
                            diag_summary = None

                total_logprob = None
                if logprob_results.sequence_logprobs:
                    total_logprob = float(logprob_results.sequence_logprobs[prompt_idx][completion_idx])

                total_logq = None
                if sequence_logqs:
                    total_logq = float(sequence_logqs[prompt_idx][completion_idx])

                extra = {
                    "prompt_meta": prompt_meta,
                    "rb_entropy_per_token": rb_entry,
                    "sequence_diagnostics": diag_summary,
                    "response_length": gen_len,
                }

                record = SequenceRecord(
                    sequence_id=sequence_id,
                    prompt_text=prompt_text,
                    response_text=sequences.responses_text[prompt_idx][completion_idx],
                    prompt_tokens=prompt_tokens_per_prompt[prompt_idx],
                    response_tokens=[int(tok) for tok in response_tokens],
                    logprob_per_token=logprob_per_token,
                    logq_per_token=logq_per_token,
                    entropy_per_token=entropy_per_token,
                    total_logprob=total_logprob,
                    total_logq=total_logq,
                    reward=float(rewards_tensor[prompt_idx, completion_idx].item()),
                    advantage=float(advantages_tensor[prompt_idx, completion_idx].item()),
                    global_prompt_id=(int(meta_global_id) if meta_global_id is not None else None),
                    dataset_split=dataset_split,
                    prompt_batch_idx=prompt_idx,
                    completion_idx=completion_idx,
                    extra=extra,
                )
                records.append(record)

        sampling_metadata = {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "num_prompts": B,
            "completions_per_prompt": completions_per_prompt,
            "seed": seed,
        }
        sampling_metadata.update(sampling_extra)

        generated_batch = GeneratedBatch(
            batch_type=batch_type,
            sequences=records,
            full_sequence_tensor=sequences.sequences.detach().cpu(),
            attention_mask=sequences.attention_masks.detach().cpu(),
            prompt_lens=list(sequences.prompt_lens),
            gen_lens=[list(row) for row in sequences.gen_lens],
            rewards=rewards_tensor.clone(),
            advantages=advantages_tensor.clone(),
            sampling_metadata=sampling_metadata,
        )
        return generated_batch

    def _filter_zero_advantage_prompts(
        self, batch: GeneratedBatch, *, tol: float = 0.0
    ) -> GeneratedBatch:
        advantages = batch.advantages
        if advantages is None:
            return batch
        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(1)
        max_vals = advantages.abs().max(dim=1).values
        keep_mask = max_vals > tol
        keep_indices = [idx for idx, flag in enumerate(keep_mask.tolist()) if flag]
        if len(keep_indices) == advantages.shape[0]:
            return batch
        if not keep_indices:
            if self.logger:
                self.logger.warning("All update prompts have zero advantage; retaining original batch")
            return batch

        prefix = batch.sequences[0].sequence_id.split('-')[0] if batch.sequences else 'SEQ'
        index_map = {old: new for new, old in enumerate(keep_indices)}

        new_sequences = []
        for record in batch.sequences:
            prompt_idx = record.prompt_batch_idx or 0
            if prompt_idx in index_map:
                new_idx = index_map[prompt_idx]
                new_seq_id = f"{prefix}-{new_idx:03d}-{record.completion_idx:02d}"
                new_record = replace(record, sequence_id=new_seq_id, prompt_batch_idx=new_idx)
                new_sequences.append(new_record)

        full_tensor = batch.full_sequence_tensor[keep_indices].clone() if batch.full_sequence_tensor is not None else None
        attn = batch.attention_mask[keep_indices].clone() if batch.attention_mask is not None else None
        prompt_lens = [batch.prompt_lens[idx] for idx in keep_indices] if batch.prompt_lens else None
        gen_lens = [batch.gen_lens[idx] for idx in keep_indices] if batch.gen_lens else None
        rewards = batch.rewards[keep_indices].clone() if batch.rewards is not None else None
        advantages_filtered = batch.advantages[keep_indices].clone() if batch.advantages is not None else None

        metadata = dict(batch.sampling_metadata)
        metadata["num_prompts"] = len(keep_indices)
        metadata["filtered_zero_adv_prompts"] = int(batch.advantages.shape[0] - len(keep_indices))

        return GeneratedBatch(
            batch_type=batch.batch_type,
            sequences=new_sequences,
            full_sequence_tensor=full_tensor,
            attention_mask=attn,
            prompt_lens=prompt_lens,
            gen_lens=gen_lens,
            rewards=rewards,
            advantages=advantages_filtered,
            sampling_metadata=metadata,
        )

    def _merge_update_batches(
        self,
        batches: List[GeneratedBatch],
        completions_per_prompt: int,
    ) -> GeneratedBatch:
        if not batches:
            raise ValueError("_merge_update_batches requires at least one batch")

        first = batches[0]
        dtype = first.full_sequence_tensor.dtype if first.full_sequence_tensor is not None else torch.long

        valid_batches = [b for b in batches if b.full_sequence_tensor is not None]
        if not valid_batches:
            return batches[0]

        total_prompts = sum(batch.full_sequence_tensor.shape[0] for batch in valid_batches)
        max_len = max(batch.full_sequence_tensor.shape[-1] for batch in valid_batches)

        final_sequences = torch.zeros((total_prompts, completions_per_prompt, max_len), dtype=dtype)
        attn_dtype = valid_batches[0].attention_mask.dtype if valid_batches[0].attention_mask is not None else torch.float32
        final_attention = torch.zeros((total_prompts, completions_per_prompt, max_len), dtype=attn_dtype)

        prompt_lens: List[int] = []
        gen_lens: List[List[int]] = []
        rewards_list: List[torch.Tensor] = []
        advantages_list: List[torch.Tensor] = []
        new_records: List[SequenceRecord] = []

        filtered_total = 0
        prefix = first.sequences[0].sequence_id.split('-')[0] if first.sequences else 'U'

        offset = 0
        for batch in valid_batches:
            seq_tensor = batch.full_sequence_tensor
            attn_tensor = batch.attention_mask
            assert seq_tensor is not None and attn_tensor is not None
            assert seq_tensor.shape[1] == completions_per_prompt

            curr_prompts = seq_tensor.shape[0]
            curr_len = seq_tensor.shape[-1]
            final_sequences[offset: offset + curr_prompts, :, :curr_len] = seq_tensor
            final_attention[offset: offset + curr_prompts, :, :curr_len] = attn_tensor

            if batch.prompt_lens:
                prompt_lens.extend(batch.prompt_lens)
            if batch.gen_lens:
                gen_lens.extend(batch.gen_lens)
            if batch.rewards is not None:
                rewards_list.append(batch.rewards)
            if batch.advantages is not None:
                advantages_list.append(batch.advantages)

            filtered_total += batch.sampling_metadata.get("filtered_zero_adv_prompts", 0)

            for record in batch.sequences:
                local_idx = record.prompt_batch_idx or 0
                new_idx = offset + local_idx
                new_seq_id = f"{prefix}-{new_idx:03d}-{record.completion_idx:02d}"
                new_records.append(replace(record, sequence_id=new_seq_id, prompt_batch_idx=new_idx))

            offset += curr_prompts

        rewards_tensor = torch.cat(rewards_list, dim=0) if rewards_list else None
        advantages_tensor = torch.cat(advantages_list, dim=0) if advantages_list else None

        sampling_metadata = dict(first.sampling_metadata)
        sampling_metadata["num_prompts"] = total_prompts
        sampling_metadata["completions_per_prompt"] = completions_per_prompt
        sampling_metadata["filtered_zero_adv_prompts"] = filtered_total

        return GeneratedBatch(
            batch_type=first.batch_type,
            sequences=new_records,
            full_sequence_tensor=final_sequences,
            attention_mask=final_attention,
            prompt_lens=prompt_lens,
            gen_lens=gen_lens,
            rewards=rewards_tensor,
            advantages=advantages_tensor,
            sampling_metadata=sampling_metadata,
        )

    # ------------------------------------------------------------------
    # Batch construction entry points
    # ------------------------------------------------------------------
    def generate_update_batch(
        self,
        batch_size_prompts: int,
        completions_per_prompt: int,
        *,
        dataset_split: str = "train",
        seed: Optional[int] = None,
        reward_cfg: Optional[Dict[str, Any]] = None,
        advantage_cfg: Optional[Dict[str, Any]] = None,
        filter_zero_advantage_prompts: Optional[bool] = None,
        advantage_filter_tol: Optional[float] = None,
    ) -> GeneratedBatch:
        """Sample an update batch mirroring the RL training loop."""

        del reward_cfg, advantage_cfg
        if batch_size_prompts <= 0:
            raise ValueError("batch_size_prompts must be positive")
        if completions_per_prompt <= 0:
            raise ValueError("completions_per_prompt must be positive")

        self._lazy_load_resources()
        assert self._sequence_processor is not None

        batch_cfg = (self.config.get("batch_config") or {})
        dataset_name = batch_cfg.get("dataset_name")
        if dataset_name is None:
            raise ValueError("SampleGenerator requires 'batch_config.dataset_name' to sample prompts.")

        if filter_zero_advantage_prompts is None:
            filter_zero_advantage_prompts = bool(batch_cfg.get("filter_zero_advantage_prompts", True))
        tol = float(advantage_filter_tol if advantage_filter_tol is not None else batch_cfg.get("advantage_filter_tol", 0.0))

        attempts = 0
        max_attempts = int(batch_cfg.get("max_resample_attempts", 10))
        remaining = batch_size_prompts
        accepted_batches: List[GeneratedBatch] = []
        last_batch: Optional[GeneratedBatch] = None

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            prompts_needed = remaining
            prompts, examples = self._sequence_processor.sample_prompts(
                dataset_name=dataset_name,
                split=dataset_split,
                num_prompts=prompts_needed,
                seed=seed if attempts == 1 else None,
            )
            prompt_metadata = [getattr(ex, "meta", {}).copy() for ex in examples] if examples else [{} for _ in prompts]

            sequences, logprob_results, diagnostics = self._generate_sequences(
                prompts,
                examples,
                completions_per_prompt,
                dataset_split=dataset_split,
                compute_rb=self._compute_rb_flag(),
            )

            sampling_extra = {"sampling_strategy": "without_replacement", "attempt": attempts}
            batch = self._build_generated_batch(
                batch_type="update",
                sequence_id_prefix="U",
                prompts=prompts,
                prompt_metadata=prompt_metadata,
                sequences=sequences,
                logprob_results=logprob_results,
                diagnostics=diagnostics,
                completions_per_prompt=completions_per_prompt,
                dataset_name=dataset_name,
                dataset_split=dataset_split,
                seed=seed if attempts == 1 else None,
                sampling_extra=sampling_extra,
            )
            last_batch = batch

            if filter_zero_advantage_prompts:
                filtered = self._filter_zero_advantage_prompts(batch, tol=tol)
                if filtered is not batch and self.logger:
                    before = batch.sampling_metadata.get("num_prompts", len(prompts))
                    after = filtered.sampling_metadata.get("num_prompts", len(prompts))
                    removed = before - after
                    if removed > 0:
                        self.logger.info(
                            "Resample attempt %d: filtered %d zero-advantage prompt group(s)",
                            attempts,
                            removed,
                        )
                batch = filtered

            kept = batch.sampling_metadata.get("num_prompts", len(batch.prompt_lens or []))
            if kept > 0:
                accepted_batches.append(batch)
                remaining -= kept
            elif self.logger:
                self.logger.debug(
                    "Resample attempt %d produced no usable prompts (kept=0)", attempts
                )

        if not accepted_batches:
            if self.logger and filter_zero_advantage_prompts:
                self.logger.warning(
                    "Unable to find non-zero advantage update prompts after %d attempts; returning last batch",
                    attempts,
                )
            return last_batch if last_batch is not None else batch

        if remaining > 0 and self.logger:
            self.logger.warning(
                "Reached max resample attempts (%d); using %d/%d prompts",
                max_attempts,
                batch_size_prompts - remaining,
                batch_size_prompts,
            )

        merged = self._merge_update_batches(accepted_batches, completions_per_prompt)
        return merged
    def generate_evaluation_batch(
        self,
        batch_size_prompts: int,
        *,
        completions_per_prompt: int = 1,
        with_replacement: bool = True,
        dataset_split: str = "train",
        seed: Optional[int] = None,
    ) -> GeneratedBatch:
        """Sample an evaluation batch for probing Fisher influence."""

        if batch_size_prompts <= 0:
            raise ValueError("batch_size_prompts must be positive")
        if completions_per_prompt <= 0:
            raise ValueError("completions_per_prompt must be positive")

        self._lazy_load_resources()
        assert self._sequence_processor is not None

        batch_cfg = (self.config.get("batch_config") or {})
        dataset_name = batch_cfg.get("dataset_name")
        if dataset_name is None:
            raise ValueError("SampleGenerator requires 'batch_config.dataset_name' to sample prompts.")

        if with_replacement:
            all_prompts, all_examples = self._sequence_processor.sample_prompts(
                dataset_name=dataset_name,
                split=dataset_split,
                num_prompts=None,
                seed=None,
            )
            if not all_prompts:
                raise ValueError(f"Dataset {dataset_name}:{dataset_split} is empty")
            rng = random.Random(seed)
            indices = [rng.randrange(len(all_prompts)) for _ in range(batch_size_prompts)]
            prompts = [all_prompts[idx] for idx in indices]
            if all_examples:
                examples = [all_examples[idx] for idx in indices]
                prompt_metadata = [ex.meta.copy() for ex in examples]
            else:
                examples = None
                prompt_metadata = [{} for _ in prompts]
            sampling_extra = {
                "sampling_strategy": "with_replacement",
                "indices": indices,
                "unique_prompts": len(set(indices)),
            }
        else:
            prompts, examples = self._sequence_processor.sample_prompts(
                dataset_name=dataset_name,
                split=dataset_split,
                num_prompts=batch_size_prompts,
                seed=seed,
            )
            prompt_metadata = [getattr(ex, "meta", {}).copy() for ex in examples] if examples else [{} for _ in prompts]
            sampling_extra = {"sampling_strategy": "without_replacement"}

        sequences, logprob_results, diagnostics = self._generate_sequences(
            prompts,
            examples,
            completions_per_prompt,
            dataset_split=dataset_split,
            compute_rb=self._compute_rb_flag(),
        )

        return self._build_generated_batch(
            batch_type="evaluation",
            sequence_id_prefix="E",
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            sequences=sequences,
            logprob_results=logprob_results,
            diagnostics=diagnostics,
            completions_per_prompt=completions_per_prompt,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            seed=seed,
            sampling_extra=sampling_extra,
        )

    def generate_from_prompt_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        dataset_split: str = "train",
        completions_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> GeneratedBatch:
        """Construct a batch by selecting prompts via global identifiers."""

        ids = [int(pid) for pid in prompt_ids]
        if not ids:
            raise ValueError("prompt_ids must be non-empty")
        if completions_per_prompt <= 0:
            raise ValueError("completions_per_prompt must be positive")

        self._lazy_load_resources()
        assert self._sequence_processor is not None

        batch_cfg = (self.config.get("batch_config") or {})
        dataset_name = batch_cfg.get("dataset_name")
        if dataset_name is None:
            raise ValueError("SampleGenerator requires 'batch_config.dataset_name' to sample prompts.")

        all_prompts, all_examples = self._sequence_processor.sample_prompts(
            dataset_name=dataset_name,
            split=dataset_split,
            num_prompts=None,
            seed=None,
        )
        id_to_entry: Dict[int, Tuple[str, Any]] = {}
        if all_examples:
            for prompt_text, example in zip(all_prompts, all_examples):
                meta = getattr(example, "meta", {})
                gid = meta.get("global_prompt_id")
                if gid is not None and int(gid) not in id_to_entry:
                    id_to_entry[int(gid)] = (prompt_text, example)
        selected_prompts: List[str] = []
        selected_examples: List[Any] = []
        for gid in ids:
            if gid not in id_to_entry:
                raise ValueError(f"Global prompt id {gid} not found in dataset {dataset_name}:{dataset_split}")
            prompt_text, example = id_to_entry[gid]
            selected_prompts.append(prompt_text)
            selected_examples.append(example)
        prompt_metadata = [ex.meta.copy() for ex in selected_examples]

        sequences, logprob_results, diagnostics = self._generate_sequences(
            selected_prompts,
            selected_examples,
            completions_per_prompt,
            dataset_split=dataset_split,
            compute_rb=self._compute_rb_flag(),
        )

        sampling_extra = {
            "sampling_strategy": "prompt_ids",
            "prompt_ids": ids,
            "seed": seed,
        }
        return self._build_generated_batch(
            batch_type="prompt_ids",
            sequence_id_prefix="PID",
            prompts=selected_prompts,
            prompt_metadata=prompt_metadata,
            sequences=sequences,
            logprob_results=logprob_results,
            diagnostics=diagnostics,
            completions_per_prompt=completions_per_prompt,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            seed=seed,
            sampling_extra=sampling_extra,
        )

    def generate_from_custom_prompts(
        self,
        prompts: Sequence[str],
        *,
        completions_per_prompt: int = 1,
        seed: Optional[int] = None,
        apply_template: bool = True,
        custom_metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> GeneratedBatch:
        """Construct a batch from raw text prompts (template-aware)."""

        if not prompts:
            raise ValueError("prompts must be non-empty")
        if completions_per_prompt <= 0:
            raise ValueError("completions_per_prompt must be positive")

        self._lazy_load_resources()

        processed_prompts: List[str] = []
        for prompt_text in prompts:
            if apply_template and gsm8k_prompt_template is not None:
                processed_prompts.append(gsm8k_prompt_template(prompt_text))
            else:
                processed_prompts.append(prompt_text)

        prompt_metadata = [
            (meta.copy() if meta is not None else {"custom_prompt_index": idx})
            for idx, meta in enumerate(custom_metadata or [{} for _ in processed_prompts])
        ]

        sequences, logprob_results, diagnostics = self._generate_sequences(
            processed_prompts,
            examples=None,
            completions_per_prompt=completions_per_prompt,
            dataset_split="custom",
            compute_rb=self._compute_rb_flag(),
        )

        sampling_extra = {
            "sampling_strategy": "custom_prompts",
            "apply_template": apply_template,
            "seed": seed,
        }
        return self._build_generated_batch(
            batch_type="custom_prompts",
            sequence_id_prefix="CUST",
            prompts=processed_prompts,
            prompt_metadata=prompt_metadata,
            sequences=sequences,
            logprob_results=logprob_results,
            diagnostics=diagnostics,
            completions_per_prompt=completions_per_prompt,
            dataset_name="custom",
            dataset_split="custom",
            seed=seed,
            sampling_extra=sampling_extra,
        )

    def build_custom_sequence(
        self,
        prompt: str,
        response: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        apply_template: bool = False,
    ) -> GeneratedBatch:
        """Wrap a single prompt/response pair with teacher-forced statistics."""

        self._lazy_load_resources()
        assert self._sequence_processor is not None and self._tokenizer is not None

        prompt_text = (
            gsm8k_prompt_template(prompt)
            if apply_template and gsm8k_prompt_template is not None
            else prompt
        )

        tokenizer = self._tokenizer
        device = next(self._model.parameters()).device if self._model is not None else torch.device("cpu")

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = prompt_ids["input_ids"].to(device)
        response_tokens = response_ids["input_ids"].to(device)
        full = torch.cat([prompt_tokens, response_tokens], dim=1)
        attention_mask = torch.ones_like(full, dtype=torch.bool)

        batched = BatchedSequences(
            sequences=full.unsqueeze(0).unsqueeze(0),
            prompt_lens=[prompt_tokens.size(1)],
            gen_lens=[[response_tokens.size(1)]],
            attention_masks=attention_mask.unsqueeze(0).unsqueeze(0),
            responses_text=[[response]],
        )

        logprob_results, diagnostics = self._sequence_processor.teacher_force_logprobs_with_diagnostics(
            batched,
            with_grad=False,
            compute_rb=self._compute_rb_flag(),
        )
        reward_value = 0.0
        if metadata and "reward" in metadata:
            reward_value = float(metadata["reward"])
        logprob_results.rewards = [[reward_value]]

        prompt_metadata = [metadata.copy() if metadata is not None else {"custom_prompt_index": 0, "provided_response": True}]

        sampling_extra = {
            "sampling_strategy": "custom_sequence",
            "provided_response": True,
        }
        return self._build_generated_batch(
            batch_type="custom_sequence",
            sequence_id_prefix="CSEQ",
            prompts=[prompt_text],
            prompt_metadata=prompt_metadata,
            sequences=batched,
            logprob_results=logprob_results,
            diagnostics=diagnostics,
            completions_per_prompt=1,
            dataset_name="custom",
            dataset_split="custom",
            seed=None,
            sampling_extra=sampling_extra,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_batch(self, batch: GeneratedBatch, path: Path) -> None:
        """Persist a GeneratedBatch to disk."""

        raise NotImplementedError

    def load_batch(self, path: Path) -> GeneratedBatch:
        """Load a GeneratedBatch from disk."""

        raise NotImplementedError
