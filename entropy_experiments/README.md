# Entropy Experiments

## Purpose
- Study how RL-style parameter updates change policy entropy on curated evaluation batches.
- Provide reproducible pipelines for AdamW update reconstruction, baseline decomposition, and per-sequence attributions.
- Support both ground-truth entropy deltas via importance sampling and faster approximations for iteration.
- Capture rich diagnostics (logging, cached stats, per-token views) that feed downstream analysis notebooks and scripts.

## Workflows At A Glance
- `EntropyMeasurements` (from `entropy_experiment_runner.py`) runs smaller sweeps directly from a YAML config.
- `EntropyInfluenceRunner` (see `entropy_influence.py`) drives large studies with explicit workspace/evaluation plans and eta sweeps.
- `scripts/run_entropy_influence_large.py` is the current entry point for batch jobs; it saves JSON summaries plus NumPy dumps for post-hoc plotting.
- Result folders (for example `results/entropy_influence/third_test/`) host raw dumps, helper scripts, and generated plots for each study.

## Directory Map

### Core Modules
- `entropy_experiment_runner.py` - thin wrapper that consumes `configs/*.yaml` files to run end-to-end probes.
- `entropy_influence.py` - orchestrates update batch construction, eta sweeps, logging, and per-sequence/baseline bookkeeping.
- `delta_entropy_true.py` - self-normalised importance sampling (SNIS) estimator with caching and multi-direction support.
- `delta_entropy_approx.py` - first-order estimator scaffold (kept for completeness; expect follow-up work before use).
- `update_vector.py` - rebuilds AdamW directions, exposing full vector, per-sequence components, and the momentum/weight-decay baseline.
- `fisher_kernel.py` - shared dataclasses (`WorkspaceSpec`, `BatchRequest`, etc.) describing batch requests and experiment layouts.

### Utilities
- `utils/sequence_processor.py` - handles teacher forcing, parameter overrides, and batched no-grad evaluations (incl. vmap path for multiple overrides).
- `utils/sample_generator.py` - wraps `SequenceProcessor` to build U/E batches, resample away zero-advantage prompt groups, and return rich metadata.
- `utils/model_loader.py` - loads backbone + adapters and (optionally) Adam optimizers from checkpoints.
- `utils/param_overrides.py` - builds functional parameter maps for `torch.func.functional_call` execution.
- `utils/detailed_logger.py` - structured logging helpers used across runners and generators.
- `utils/precision_utils.py` - helpers for dtype management, TF32 toggles, and gradient casting.

### Results & Configs
- `configs/config_template.yaml` - canonical starting point; copy/adapt per experiment.
- `results/entropy_influence/*` - experiment-specific dumps (`data/`, `plots/`, analysis scripts).
- `results/testing_linear_regime/`, `results/entropy_variance/`, etc. - earlier studies kept for reference.
- `baselines/` - baseline policy evaluation scripts and helpers.
- `other_scripts/` - ad-hoc drivers built during exploration.
- `architecture_notes.txt`, `debug/` - design notes and scratch assets.

## Sampling and Update Construction
- Update batches are produced via `SampleGenerator.generate_update_batch`, which resamples until every prompt group has a non-zero max advantage (tolerance configurable via `batch_config`).
- Generated batches retain prompt/response text, token IDs, per-token logprobs, advantages, and provenance metadata for later joins.
- `update_vector.compute_update_vector_adamw` reconstructs the AdamW step, returning `(direction, baseline, stats)` plus optional per-sequence directions when a callback is supplied.
- Baseline terms (momentum + weight decay) are stored separately and per-parameter shares are attached to component diagnostics for exact decomposition checks.
- Gradient clipping and AMP choices are honoured during reconstruction; statistics (norms, applied scales, microbatch counts) are logged for reproducibility.

## Entropy Estimators
- `DeltaEntropyTrue` caches baseline teacher-forced stats per evaluation batch and reuses them across eta sweeps to reduce GPU passes.
- `compute_delta_h_true_multi` accepts a list of `v_i` directions, building functional overrides in parallel and dispatching through the vectorised no-grad path.
- `tf_batch_size` (configurable under `true_delta_h`) controls how many evaluation sequences are processed per forward chunk; raise it to better saturate the accelerator.
- Returned diagnostics include per-sequence integrands, token counts, SNIS weight summaries, effective sample sizes, and clipping fractions for stability audits.
- The first-order estimator scaffold remains in place for future work; today all production runs rely on the ground-truth path above.

## Running Experiments
1. Copy `configs/config_template.yaml` and fill out checkpoint, dataset, precision, and logging sections (ensure optimizer state paths are present for AdamW reconstruction).
2. Optionally set `batch_config.filter_zero_advantage_prompts=true` (default) and tweak `max_resample_attempts`, tolerance, or prompt counts for your study.
3. Launch `scripts/run_entropy_influence_large.py` (adjust `ETAS`, `WorkspaceSpec`, and evaluation requests inline or by editing the script) to materialise a `EntropyInfluencePlan` and execute it.
4. Monitor stdout/logs: progress bars are emitted via `tqdm`, and `DetailedLogger` messages note sampling retries, update-vector norms, and estimator timing.
5. Inspect the freshly created run directory under `results/entropy_influence/`, which contains `summary.json`, per-eta aggregates, and NumPy dumps for each decomposition.

## Outputs
- `summary.json` - high-level metadata (plan parameters, aggregate DeltaH values, gradient/baseline reconstructions, norms, timings).
- `eval_XX_delta_matrix.npy` - matrix of per-(evaluation seq x update seq) entropy deltas at the reference eta.
- `eval_XX_per_sequence/full|baseline|grad|baseline_plus_grad/` - NumPy dumps for each decomposition flavour across eta values.
- `eval_XX_grad_eta/*.npy` - stacked grad-only deltas per eta along with JSON diagnostics for SNIS statistics.
- `plots/` - analysis scripts save figures here; keep per-study notebooks or scripts beside the data for provenance.

## Analysis Helpers
- `results/entropy_influence/*/analyze_grad_linearity.py` - aggregates per-sequence deltas, fits linear models over eta, and reports goodness-of-fit metrics.
- `results/entropy_influence/*/plot_grad_delta_samples.py` - quick-look plots for selected update directions to visualise linearity vs. noise.
- Use these scripts as templates for bespoke studies; they assume the standard directory layout produced by `run_entropy_influence_large.py`.

## Performance & Precision Tips
- Increase `true_delta_h.tf_batch_size` when GPU memory allows; entropy evaluations are no-grad and typically memory-light.
- Use `EntropyInfluencePlan.grad_chunk_size` to evaluate multiple `v_i` directions in a single pass; keep an eye on SNIS variance and RAM usage.
- Baseline + grad reconstructions can be sensitive to numerical precision at very small eta; prefer FP64 entropy accumulators when diagnosing mismatches.
- Logging at `standard` level highlights resampling retries and estimator clips; bump to `detailed` only when debugging due to large log volume.
- If aggregate DeltaH signs look surprising, compare `summary.json` aggregates against per-sequence dumps to confirm reconstruction and numerical stability.

## Development Notes
- Teacher forcing paths run strictly in eval/no-grad mode; gradient checkpoint toggles in the checkpoint should remain disabled during probes.
- All parameter overrides are built via `torch.func.functional_call`; avoid in-function `requires_grad_` toggles when extending these paths.
- When adding new estimators or analyses, keep experiment-specific scripts in the corresponding results subfolder to preserve provenance.
- Commit messages and configs should document eta grids, batch sizes, and precision settings so reruns on Lambda or Colab can be reproduced quickly.
