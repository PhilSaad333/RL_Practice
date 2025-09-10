# SequenceProcessor (utils) — Detailed Documentation

This document explains the design, configuration, and usage of `entropy_experiments/utils/sequence_processor.py`. The module provides a unified interface for batched text generation, teacher-forced log‑prob computation, and sequence‑level diagnostics, with careful control over precision and parameter override workflows used by the entropy probe.

## Overview

- Purpose: consolidate patterns from rollout collection and training (e.g., GRPO) into a single, reusable component for
  - Prompt sampling from registered datasets
  - Batched text generation with stop conditions
  - Teacher‑forced scoring (with/without gradients)
  - RB entropy and distribution diagnostics aligned to the sampling policy
  - Parameter‑only functional overrides for continuity and what‑if analyses

- Core concepts:
  - B: number of prompts in a batch
  - G: responses per prompt
  - Left padding, single padded prompt length per batch
  - “Generated region” alignment: token at position `i` is scored with logits at `i‑1`

## Key Types

- `GenerationConfig` (dataclass)
  - `temperature: float = 1.0`
  - `top_p: float = 1.0`
  - `max_new_tokens: int = 200`
  - `do_sample: bool = True`
  - `num_return_sequences: int = 8`
  - `gen_batch_size: int = 32`
  - `tf_batch_size: int = 8`
  - `rb_requires_grad: bool = False` (when computing differentiable RB entropies)

- `BatchedSequences` (dataclass)
  - `sequences: Tensor [B, G, L_total]`
  - `prompt_lens: List[int]` (single padded prompt length repeated per prompt)
  - `gen_lens: List[List[int]]` (per prompt/response generation lengths)
  - `attention_masks: Tensor [B, G, L_total]`
  - `responses_text: List[List[str]]`

- `LogprobResults` (dataclass)
  - `logprobs: List[List[Tensor]]` per‑token log p for generated tokens
  - `entropies: List[List[np.ndarray]]` naive per‑token surprisal
  - `sequence_logprobs: List[List[float]]` sums of log p
  - `rb_entropies: List[List[np.ndarray]]` RB per‑token entropies under the sampling policy
  - `rewards: List[List[float]]` sequence rewards (via `tag_pref`)
  - `rb_entropies_torch: Optional[List[List[Tensor]]]` differentiable RB entropies, if enabled
  - `baseline_feats_torch: Optional[List[List[Tensor]]]` optional per‑step features for regression baselines
  - `token_logqs: Optional[List[List[Tensor]]]` per‑token log‑q under the sampling policy
  - `sequence_logqs: Optional[List[List[float]]]` sums of log‑q per sequence

- `DiagnosticsResults` (dataclass)
  - `diagnostics: List[List[DiagnosticsPack]]`

- `DiagnosticsPack` (dataclass)
  - `TokenStepDiagnostics`: per‑step RB entropy, head/tail mass, two‑point entropy, top‑1 prob, margin, collision, Rényi‑2, effective support, logit moments, optional EOS prob
  - `SequenceDiagnostics`: aggregates (sum/mean/min/max, early/late windows, surprisal stats, etc.)

## Precision and Determinism Profiles

`SequenceProcessor` accepts either a `GenerationConfig` or a dict with sections `{ generation: ..., precision: ... }`.

- Functional override profile (`_fo_cfg`):
  - `autocast` (default False)
  - `dtype` (default "float32")
  - `cast_params` (default True) — upcast effective parameters (θ + η·v) before `functional_call`

- Teacher‑forcing, no‑grad profile (`_tf_cfg`):
  - `autocast` (default False)
  - `dtype` (default "float32")
  - `cast_logits_fp32` (default True) — upcast logits for numerics

- Global knobs:
  - `allow_tf32` (default False), `matmul_precision` (default "high") via `apply_global_precision`
  - `entropy_fp64` (optional): compute softmax/log‑prob math in fp64 for improved numerical stability while keeping the forward in fp32

Notes:
- The module unwraps DDP for a canonical target (`self._mdl_target`) but keeps LoRA/PEFT wrappers intact.
- For probe stability, runtime params are assumed fp32 at model load; autocast is disabled in the key scoring paths.

## Parameter Override Helpers

- `_build_params_override(v_named, eta) -> Dict[str, Tensor]`
  - Builds a params‑only mapping for `torch.func.functional_call` against `self._mdl_target`.
  - Detaches base params/buffers; optionally casts params to `_fo_cfg.dtype` if `cast_params=True`.

- `_build_state_override(v_named, eta, *, param_dtype, include_buffers, buffer_dtype=None)`
  - Generalized builder for params + optional buffers; still detaches and can apply dtype casts.
  - Used internally to cache a params‑only zero‑eta mapping (`self._params_zero`).

- `_fc_logits_noautocast(input_ids, attention_mask, params_mapping)`
  - Runs a single forward via `functional_call` with autocast disabled and `use_cache=False`, returning logits.

## Generation

- `sample_prompts(dataset_name, split, num_prompts, seed)`
  - Uses `rlp_datasets.DATASET_REGISTRY[dataset_name](split=...)` and extracts `example.question` for prompts.

- `generate_batched(prompts, G, gen_batch_size=None) -> BatchedSequences`
  - Tokenizes with left padding; uses a padded prompt length shared across the batch.
  - Expands prompts to `[B*G, seq_len]` and generates with a `StopAfterAnswer` logits processor that halts on the `</answer>` tag.
  - Returns sequences reshaped to `[B, G, L_total]`, attention masks, per‑response gen lengths, and decoded responses.

- `generate_with_replacement_sampling(total_sequences, dataset_name, split='train', max_prompts_pool=None, seed=None, **kwargs)`
  - Samples prompt indices with replacement from a prompt pool; creates explicit prompts of length `total_sequences`.
  - Forces `G=1` for true independence and then calls `generate_with_logprobs`.

## Teacher‑Forced Scoring

- `teacher_force_logprobs(sequences, with_grad=False, tf_batch_size=None, compute_rb=False, return_baseline_features=False, params_override=None, buffers_override=None) -> LogprobResults`
  - Dispatches to `_teacher_force_no_grad` or `_teacher_force_with_grad`.
  - In no‑grad path:
    - Unified forward: always uses `_fc_logits_noautocast(..., mapping)` where `mapping` is `params_override` or a cached `self._params_zero` (params‑only, fp32).
    - Entropy/log‑prob math optionally in fp64 if `entropy_fp64=True`.
    - Computes per‑token log‑p for realized tokens, naive surprisal, optional RB entropies, and Stage‑2 per‑token log‑q (sampling policy).
    - Produces `DiagnosticsResults` using the same sampling policy (temperature + top‑p) via `DistributionDiagnostics`.

- `teacher_force_logprobs_with_diagnostics(...) -> (LogprobResults, DiagnosticsResults)`
  - Same as above but always returns the diagnostics alongside results.

- `generate_with_logprobs(prompts=None, G=8, dataset_name=None, split='train', num_prompts=None, seed=None, with_grad=False, gen_batch_size=None, tf_batch_size=None, compute_rb=False) -> (BatchedSequences, LogprobResults, DiagnosticsResults)`
  - End‑to‑end path: build prompts (explicit or sampled), generate sequences, score with teacher forcing (+ diagnostics), compute rewards with `tag_pref`, and return everything.

## RB Entropies and Sampling Measure

- `_rb_entropies_top_p(gen_logits, top_p, temperature) -> Tensor[T]`
  - Computes per‑step RB entropies under the SAME sampling policy used for generation:
    - temperature scaling
    - nucleus (top‑p) truncation and renormalization, or full softmax when `top_p=1.0`

- `_compute_logq_top_p(gen_logits, gen_tokens, top_p, temperature) -> Tensor[T]`
  - Produces per‑token log‑q for realized tokens under the sampling distribution, not the raw model distribution.
  - Used for correct importance sampling weights and diagnostics.

- `DistributionDiagnostics(top_p, temperature, eos_token_id)` and `compute_from_logits(...)`
  - Convenience utility to compute token‑level and sequence‑level diagnostics aligned with the sampling policy. Returns a `DiagnosticsPack` with both per‑step and aggregate stats.

## Rewards

- `_compute_rewards(prompts, sequences, examples) -> List[List[float]]`
  - Uses `rl_training.rewards.tag_pref.reward_fn` with a temporary `PROMPT2GOLD` mapping derived from dataset examples (if available). Falls back to zero rewards when gold answers are missing or reward computation fails.

## Shapes and Alignment Conventions

- `sequences.sequences`: `[B, G, L_total]` (prompt+generation with left padding)
- `prompt_lens[b]`: padded prompt length shared across the batch
- `gen_lens[b][g]`: generated token count for each response
- Teacher forcing aligns logits to targets as: logits at index `i‑1` score token at index `i` within the generated region (`[prompt_len, prompt_len+gen_len)`).

## Usage Examples

Basic initialization with dict config:

```python
from entropy_experiments.utils.sequence_processor import SequenceProcessor, GenerationConfig

# model, tokenizer are pre‑loaded (LoRA/PEFT kept intact)
cfg = {
  'generation': {'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 64, 'gen_batch_size': 16, 'tf_batch_size': 8},
  'precision': {
    'func_override': {'autocast': False, 'dtype': 'float32', 'cast_params': True},
    'tf_nograd': {'autocast': False, 'dtype': 'float32', 'cast_logits_fp32': True},
    'allow_tf32': True, 'matmul_precision': 'high', 'entropy_fp64': False,
  },
}
sp = SequenceProcessor(model, tokenizer, logger=None, config=cfg)

# Generate and score from dataset
seqs, lps, diags = sp.generate_with_logprobs(
  prompts=None, G=4, dataset_name='gsm8k_r1_template', split='test', num_prompts=8, compute_rb=True
)

# Parameter‑only override (θ + η·v) scoring on existing sequences
params_mapping = sp._build_params_override(v_named=update_vector_named, eta=1e-6)
lp_no_grad = sp.teacher_force_logprobs(seqs, with_grad=False, compute_rb=True, params_override=params_mapping)
```

Replacement sampling (independent draws):

```python
seqs, lps, diags = sp.generate_with_replacement_sampling(
  total_sequences=32768, dataset_name='gsm8k_r1_template', split='train', seed=42, G=1, compute_rb=True
)
```

## Gotchas and Best Practices

- Do not replace `SequenceProcessor.config` (a `GenerationConfig`) with a dict after initialization. If you need to adjust precision, update `sp._fo_cfg`/`sp._tf_cfg` (or pass a dict at construction time).
- Keep the sampling policy (temperature, top‑p) identical between generation and diagnostics/RB computations.
- For importance sampling correctness across the probe, prefer `top_p=1.0` during E‑batch generation.
- The forward path used in teacher‑forcing no‑grad disables autocast and relies on fp32 runtime parameters for stability.

## Extension Points

- Differentiable RB entropies: set `GenerationConfig.rb_requires_grad=True` and use `compute_rb=True`.
- Baseline features (regression): enable `return_baseline_features=True` in `_teacher_force_with_grad` path to produce per‑step features.
- Precision tuning: enable `entropy_fp64` to reduce numeric drift in entropy/log‑prob computations without changing the forward runtime dtype.

## Related Modules

- `entropy_experiments/offline_entropy_probe.py` — orchestrates E/U batch sampling and probe phases; binds a `SequenceProcessor` and controls top‑p/temperature for IS.
- `entropy_experiments/utils/param_overrides.py` — builds functional state mappings for `functional_call`.
- `entropy_experiments/utils/precision_utils.py` — precision context managers and dtype helpers.

## Teacher Forcing: No‑Grad vs With‑Grad

This section expands on the two critical teacher‑forcing paths, their precision/stability choices, outputs, and intended use‑cases.

### No‑Grad Path (`_teacher_force_no_grad`)

- Purpose: numerically stable evaluation for probes/diagnostics and for scoring sequences under parameter overrides without building a computation graph.
- Forward execution:
  - Always runs through `_fc_logits_noautocast` which uses `torch.func.functional_call(self._mdl_target, mapping, ...)` with autocast disabled and `use_cache=False`.
  - Uses a params‑only mapping:
    - If `params_override` is provided: use it directly (e.g., θ + η·v mapping).
    - Else: uses a cached, fp32, zero‑eta mapping (`self._params_zero`) built via `_build_state_override(..., include_buffers=False)`.
  - Buffers are not overridden and remain those of the live module.
- Precision:
  - Forward is in the params’ actual dtype (fp32 runtime is recommended; autocast is disabled here for determinism).
  - Softmax/log‑prob/entropy can run in fp64 if `entropy_fp64=True` for extra numerical stability.
- Outputs and dtypes:
  - `logprobs[b][g]`: torch CPU tensor (detached) built from numpy; no grad.
  - `entropies[b][g]`: numpy array (naive surprisal).
  - `rb_entropies[b][g]`: numpy array; no `rb_entropies_torch` in this path.
  - `token_logqs[b][g]`: torch CPU tensor (detached) useful for IS diagnostics.
  - `sequence_logprobs/sequence_logqs`: Python floats (sums).
  - `DiagnosticsResults`: per‑token and aggregate stats under the sampling policy.
- Performance/memory: light on memory (no graph). Suitable for large B/G and long sequences. Ideal for Phase 0/5 probe analytics and ablations under different θ + η·v.
- Typical use‑cases:
  - E‑batch entropy evaluations for probes.
  - Ground‑truth importance sampling measurements (Phase 5).
  - Continuity/linearity checks across tiny η using parameter overrides.

### With‑Grad Path (`_teacher_force_with_grad`)

- Purpose: compute graph‑carrying per‑token log‑probabilities (and optionally differentiable RB entropies) for building surrogate objectives and taking gradients outside the routine.
- Forward execution:
  - Runs the live module directly (no `functional_call`) with `forward_precision_ctx(autocast=..., dtype=...)` controlled by `precision.tf_withgrad`.
  - Does not accept `params_override`/`buffers_override`; it evaluates the current θ in the module. If you need gradients under θ + η·v, construct a separate module/mapping and manage the functional call at the call‑site.
  - Uses `use_cache=False` (teacher forcing), preserves graph through logits/log‑probs.
- Precision:
  - `precision.tf_withgrad` controls autocast and target dtype (defaults: autocast=False, dtype=float32). `cast_logits_fp32=True` will upcast logits to fp32 before softmax/log.
  - Prefer disabling autocast for reproducible probe gradients; enable carefully if memory‑constrained.
- Outputs and dtypes:
  - `logprobs[b][g]`: torch Tensor on device with gradients (graph‑carrying).
  - `rb_entropies_torch[b][g]`: torch Tensor [T] present when `compute_rb=True` and `config.rb_requires_grad=True`.
  - `rb_entropies[b][g]`: numpy copy for logging even when the torch version is present.
  - `token_logqs[b][g]`: torch Tensor (diagnostic; typically not used in the loss but carries graph by construction).
  - `baseline_feats_torch[b][g]` (optional): torch Tensor [T, 7], detached features to support regression/control‑variates outside.
  - `sequence_logprobs/sequence_logqs`: Python floats (detached sums for convenience).
- Performance/memory: higher memory footprint due to graph retention; consider smaller `tf_batch_size` or gradient checkpointing at the caller level if needed.
- Typical use‑cases:
  - Building entropy‑gradient estimators (Phase 1‑3) using per‑token `logprobs` and differentiable RB terms.
  - Surrogate losses combining REINFORCE and RB residuals; taking `.backward()` or `torch.autograd.grad` afterwards.

### Side‑by‑Side Differences

- Parameter overrides:
  - No‑grad: supports `params_override` mapping (θ + η·v) and uses a stable params‑only path; caches a zero‑eta mapping.
  - With‑grad: does not accept mapping; evaluates the live module θ. To differentiate w.r.t. θ + η·v, orchestrate a functional call at the call‑site.
- Autocast and dtype:
  - No‑grad: autocast disabled in `_fc_logits_noautocast`; optional `entropy_fp64` for softmax/log.
  - With‑grad: `precision.tf_withgrad` controls autocast/dtype; logits optionally cast to fp32.
- RB entropies:
  - No‑grad: RB returned as numpy only; no gradient path.
  - With‑grad: RB has a torch version (`rb_entropies_torch`) when `rb_requires_grad=True`, enabling gradient‑based estimators.
- Outputs (logprobs):
  - No‑grad: torch CPU tensors detached (from numpy) — analysis‑friendly, not for backprop.
  - With‑grad: torch device tensors with graph — suitable for building losses.
- Stability vs. flexibility:
  - No‑grad prioritizes numeric determinism (functional_call, fixed dtypes, no autocast, fp32 runtime).
  - With‑grad prioritizes gradient availability and defers precision trade‑offs to `precision.tf_withgrad`.

### Suggested Patterns

- Probing/diagnostics (no‑grad):
  - Use `params_override` to inspect continuity/linearity in η.
  - Keep `top_p=1.0` for importance sampling correctness; if using top‑p, ensure `_compute_logq_top_p` matches generation.
- Gradient‑based estimation (with‑grad):
  - Enable `rb_requires_grad=True` and `compute_rb=True` when you want RB residuals in the loss.
  - Build REINFORCE terms as `(G−b).detach() * logprobs` and add `rb_entropies_torch` as needed; take grads externally.
  - If precision issues arise, disable autocast and keep logits in fp32 (or use `entropy_fp64` only for the no‑grad path to cross‑check numerics).

