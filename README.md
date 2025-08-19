# RL Practice: Entropy Dynamics in Reinforcement Learning

A research project exploring entropy dynamics during reinforcement learning training, with a focus on understanding the theoretical foundations and empirical behavior of entropy changes in policy optimization.

## Project Overview

Sort of inspired by https://arxiv.org/abs/2505.22617. Their derivations are done under very unrealistic assumptions. A more proper analysis (as I attempt to do in the docs folder) is somewhat more involved, and points to a lot of interesting things to study. So while a "main goal" would be to see if a proper analysis suggests any further modifications to the algorithm changes suggested by Cui et al, I'm mostly trying to just get some experience with running experiments to study the things encountered in the high level theory stuff.

We can roughly split what I'm doing into:

- **1/3 Theoretical Investigation**: Understanding entropy dynamics during gradient steps in policy optimization (see [`docs/RL_studies (1).pdf`](docs/RL_studies%20(1).pdf))
- **1/3 Infrastructure & Tooling**: Building robust distributed training workflows, evaluation pipelines, and analysis tools
- **1/3 Empirical Exploration**: Investigating gradient noise scale, Fisher kernels, and other phenomena encountered along the way

In Cui et al (https://arxiv.org/abs/2505.22617), they find that tokens with a large covariance between logprob and advantage have a large effect on the entropy. It's easy to see that these are tokens in incorrect responses that have low probability. They propose methods of reducing the effect of these tokens on updates, and find that the entropy stabilizes. That makes intuitive sense: the basic policy gradient algorithms penalize this "unsuccessful exploration" but we can decide not to penalize these instances at small local cost in performance gains.

In a more thorough analysis of the step change in entropy, it is easy to see that something close to their results comes out if we assume the Fisher Kernel ($$\nabla \log \pi(t) \cdot \nabla \log \pi(t')$$, where t, t' are sequences of tokens ) is diagonal in sequences. While this is certainly not true (in fact the Fisher Kernel is massively degenerate), the emprical results from Cui et all suggests that in some sense this is not a horrible approximation. But it would be good to understand the structure of the Fisher Kernel more generally (especially since it is related to other important things, e.g. gradient noise scale), and possibly with a more thorough analysis we would see better ways to tweak algorithms along the lines of Cui et al.

Here is a sampling of some current thoughts, so that any reader can get an idea of the way I am trying to think about this:

The Fisher Kernel is a product of two rectangular matrices $$K = w^T w$$ with $$w_{\alpha, t} = \nabla_\alpha \log \pi(t)$$ So if the number of sequences is larger than the number of model parameters, K is degenerate. The number of sequences $$V^L$$ for vocab of size V grows exponentially in L, so this is true even for short sequences. Really we should be concerned with just the typical set of sequences, of size $$exp(h L)$$ where h is the per token entropy, but even then for small sequence lengths, $$N_{seqs}^{(typical)} \gg N_{params}$$. So K must be very degenerate. Also, we expect a lot of structure in K - e.g. matrix elements between sequences with the same prompt are probably correlated. It would be nice to have a simple model of $K$, perhaps as some structured matrix with random noise. Doing studies of the Fisher Kernel from RL runs and trying to form some sort of model is sort of a "first order of business" in my project, after I get the infrastructure set up. Then, among other things, a natural question would be to test some extreme limits of this model. For example, in the case of long sequences, perhaps the deviations from the "naive" Cui et al approach would be enhanced (however this would be difficult to test experimentally).



### Key Components
```
rl_training/
├── algs/dr_grpo.py           # Main RL algorithm with entropy probes
├── runners/                  # Training orchestration
│   ├── rl_runner.py         # Main training loop
│   ├── eval_batch.py        # Batch evaluation of checkpoints
│   └── resume_training.py   # Resume from any checkpoint
└── utils/                   # Gradient noise scale, Fisher kernels, etc.

evals/                       # Evaluation pipeline
└── metrics/                 # Custom metrics for entropy analysis
```


### Models & Datasets
- **Models**: Qwen2.5-1.5B (LoRA fine-tuned)
- **Dataset**: GSM8K mathematical reasoning with R1 template format
- **Training Results**: In early runs (single A100 on colab) it took several hours to do ~50 steps, but that was enough for a ~10% pass@1 gain on evals. But this was so slow I decided I needed to switch to multi-gpu
- **Environment**: Lambda Cloud GPU instances (e.g 2x H100 80GB)

### Current Investigations
1. **Linear Order Entropy Changes**: Testing whether δH₁ ≈ actual entropy changes
2. **Fisher Kernel Structure**: Measuring K₁(t,t') between sequences
3. **Sequence Correlations**: Beyond Cui et al.'s token-level analysis



### File Organization
```
/lambda/nfs/localfs/
├── training_runs/              # Complete training sessions
│   └── run_YYYY-MM-DD_HH-MM-SS/
│       ├── training_state/     # Full checkpoints (model + optimizer)
│       ├── logs/              # Training metrics, rollouts, ratios
│       └── tensorboard/       # TensorBoard events
└── eval_runs/                 # Evaluation results
    └── *_gsm8k_r1_template/
        ├── consolidated_metrics.csv
        └── step_*/            # Per-checkpoint results
```


## Documentation

A lot of the documentation was written for claude code's usage. I started using claude code after switching from using colab's gpus to the lambda cloud. I found it pretty annoying to interact with the lambda cloud, especially having to set up every time, but now claude can run anything I'd like pretty seamlessly. Thanks Claude! 

- [`NEW_TRAINING_WORKFLOW.md`](NEW_TRAINING_WORKFLOW.md) - Complete workflow documentation
- [`CLAUDE_GUIDE.md`](CLAUDE_GUIDE.md) - Project context and commands
- [`lambda/LAMBDA_SETUP_GUIDE.md`](lambda/LAMBDA_SETUP_GUIDE.md) - Infrastructure setup
- [`docs/RL_studies (1).pdf`](docs/RL_studies%20(1).pdf) - Theoretical foundations

