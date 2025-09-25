# Entropy Experiments

## Basic Idea
This is the main folder for all of my experiments. Generally what I do is take a model checkpoint with saved optimizer state and do repeated experiments from that checkpoint.

The main theme of these experiments is to understand "What properties of a training step cause certain changes measurable quantities". One approach to this I see a lot is to compare the results of a full training run with no changes to one where the training algorithm is tweaked. Clearly this is not a good way to get reliable results for things that can be measured on small timescales in training. For such things, like the step change in entropy, which is the quantity I'm focusing on for now, we can do controlled experiments and get reliable results by running repeated experiments from the same training state.

For example, I'm interested in understanding what causes the model's entropy to decrease over training (why does it explore less over time?). In this paper https://arxiv.org/abs/2505.22617 which motivated my project, they identify a candidate criterion for tokens in training samples that have an outsized contribution on entropy decrease (low probability tokens in incorrect responses). To demonstrate this they run multiple full training runs with algorithm tweaks that decrease the influence of these tokens to varying degrees, and find that as they decrease the influence, the entropy flattens out at a higher value. With sufficiently many training runs, an experiment like this can of course give reliable results, but that is sensitive to the variance of the quantity of interest (In this case the floor of the entropy late in training - for N randomly sampled runs how does this value vary?), and is pretty expensive.

Furthermore, experiments like this seem inflexible - each expensive series of runs only tests one feature (e.g. the influence of those outlier tokens). 

So I'm particularly interested in learning how to do experiments that are both scientifically reliable, inexpensive, and scalable. My current approach is to focus on a relatively small number of checkpoints taken from a single training run (or maybe a small number of runs), and use those as a proxy for studying the long-timescale dynamics during training. 

As an illustration consider the question: "Do tokens with low probability in incorrect responses have an outsized contribution on the entropy decrease?". To study this with the naive "do many runs" approach, we need to identify some way to control for the effect of those tokens (done imperfectly but practically by the algorithm tweak proposed in that paper). Then we do N runs of T training steps, so a total of $N\times T$ steps. 

Instead of doing this, I would do a single run (let's assume a single run is cheap) and produce K checkpoints over the T steps. Each of the K checkpoints is morally a representative of that period during the full training run. Then we can run N' experiments (each requiring an order one number of simulated training steps) from each of the K steps, for a total of $N'\times K$ steps. With $K\ll T$ we can get more "controlled data points" out of the same budget - essentially, doing repeated full training runs wastes resources becauses we can, for these purposes, study all timescales during training by re-running from a much smaller number of checkpoints many times. Doing this re-running gives us controlled results, whereas needing to re-run every step for each run means having a lot of wasted training steps. 

Furthermore, we can test a lot more things at once, and get much higher quality data, by doing these repeated simulated training step experiments. For example, we can simultaneously compute the correlation between our quantity of interest (entropy change) and many properties of the training samples. In this case we can do even better - in the regime where the entropy change can be linearized in the learning rate (As I show is valid for the learning rates I used in training), we can approximately isolate the effects of individual samples in a highly controlled way. By being more careful about our measurements in these ways, we can not only get much more reliable results for cheaper, but we can more easily discover new effects (e.g. if the criterion for samples with outlier influence is more general than the one identified in https://arxiv.org/abs/2505.22617). We could even do more interesting things like study the tradeoff between exploration and performance gain by doing controlled experiments that compare the influence of individual samples, or tweaks to the training algorithm, on the rate of performance gain vs entropy decrease.

A minor caveat is that its harder to test long term effects. For example, the motivation behind studying this entropy stuff is that one would hope that if you can encourage more exploration at low 'local' cost in performance gain, long term performance gain would be higher. This is harder to measure.

Another thing I want to emphasize in these experiments is getting reliable measurments of our quantities of interest (e.g. step change in entropy). I'm pretty new to all this stuff about variance reduction, but from my brief experience so far it seems pretty important to take it seriously. A large fraction of the time I've spent on this project has been trying to get more precise measurements, which meant learning about fancy stats techniques. Before this I had no idea how useful they were!

## What I do in practice

For all the experiments, we have to run repeated measurements from the same model state. For various reasons, I decided that the best approach would be that for each measurement we should compute the 'update vector' (change in model parameters normalized by learning rate) explicitly from the saved optimizer state and the gradients from the samples used for an update. This is done in `update_vector.py` (I confirmed it works correctly by comparing with the results form a true model update). This update vector is used in two different ways: 

- We can do simulated model updates by calling the model using Pytorch's `functional_call` with parameters $\theta +\eta*v$.

- We can compute the formulas for the linearized entropy change directy by estimating $\nabla H$ and taking the dot product with $\eta*v$. It is most convenient to do this using Pytorch's `jvp`, which requires using `functional_call` (this is why I did this in the first place)

We can also decompose the update vector as $v = v_{momentum} + \sum_{i\in U} v_i$, where $U$ is the batch of update samples. With these individual components we can isolate the effect of each sample (This crucially takes advantage of the linearized approximation).

Currently, I'm studying the correlation between properties of each sample and its effect on the entropy.


Below is an AI generated summary of the structure of this folder.




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
