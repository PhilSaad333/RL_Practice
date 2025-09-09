# RL Practice: Entropy Dynamics in Reinforcement Learning

A research project exploring entropy dynamics during reinforcement learning training, with a focus on understanding the theoretical foundations and empirical behavior of entropy changes in policy optimization.

## Project Overview

Sort of inspired by https://arxiv.org/abs/2505.22617. Their derivations are done under very unrealistic assumptions. A more proper analysis (as I attempt to do in the docs folder) is somewhat more involved, and points to a lot of interesting things to study. So while a "main goal" would be to see if a proper analysis suggests any further modifications to the algorithm changes suggested by Cui et al, I'm mostly trying to just get some experience with running experiments to study the things encountered in the high level theory stuff.

We can roughly split what I'm doing into:

- **1/3 Theoretical Investigation**: Understanding entropy dynamics during gradient steps in policy optimization (see [`docs/RL_studies.pdf`](docs/RL_studies.pdf))
- **1/3 Infrastructure & Tooling**: Setting up training and experimentation code in the distributed setting, as well as various analysis tools
- **1/3 Empirical Exploration**: Taking lots of detours to get more familiar with interesting and useful things I'm learning about along the way (grdient noise scale for getting a sense of what is a good batch size, experimenting with various variance reduction techniques like Rao-Blackwellization and control variates, basically anything that seems worthwhile for doing better science)

I've started keeping a log of what I've done and am doing in the project for my own benefit, but it explains some of the design choices (see [`docs/Project_Log.pdf`](docs/Project_Log.pdf))

**For anyone reading this: the main folders to look at to see what I'm doing are the docs folder, containing my notes, and the entropy_experiments folder, which contains the main experimentation code.** 

Here's a summary of the idea behind the project:

In Cui et al (https://arxiv.org/abs/2505.22617), they find that tokens with a large covariance between logprob and advantage have a large effect on the entropy. It's easy to see that these are tokens in incorrect responses that have low probability. They propose methods of reducing the effect of these tokens on updates, and find that the entropy stabilizes. That makes intuitive sense: the basic policy gradient algorithms penalize this "unsuccessful exploration" but we can decide not to penalize these instances at small local cost in performance gains.

In a more thorough analysis of the step change in entropy, it is easy to see that something close to their results comes out if we assume the Fisher Kernel ($$\nabla \log \pi(t) \cdot \nabla \log \pi(t')$$, where t, t' are sequences of tokens) is proportional to the identity. While this is certainly not true, the emprical results from Cui et all suggests that in some sense this is not a horrible approximation. But it would be good to understand the structure of the Fisher Kernel more generally as it seems like a really fundamental object, and possibly with a more thorough analysis we would see better ways to tweak algorithms along the lines of Cui et al. The fisher kernel, by the way, has a simple interpretation as a measure of the influence a sequence $t_i'$ used in a gradient update has on the model's log probability for another sequence $t$. That's true in ordinary supervised learning with a cross entropy loss, and for policy gradients

$$\delta \log \pi(t) = \eta \frac{1}{B}\sum_{i=1}^B K(t,t_i') A(t_i')$$

Here the advantage just tells us the strength and direction of the update to the logprob. (Also, using adam instead of SGD simply modifies the inner product used in the definition of the fisher kernel, doesn't change the above forumula.) Clearly the fisher kernel should NOT behave like it is diagonal - the fact that sequences used for updates have a big effect on the model's probability for totally different sequences is the whole point! 

Here is a sampling of some current thoughts, so that any reader can get an idea of the way I am trying to think about this:

1) - Current goal is to thoroughly test the first-order approximation to the step change in entropy $\delta \mathcal{H}$. I want to do this very properly, as it seems like a good way to force myself to learn important stuff.

2) - Next goal is to do a detailed exploration of the first order step change formula. Beyond the naive contributions identified by Cui et al, what groups of sequences have an outsized effect on the entropy change? Does the behavior depend in an interesting way on the sequence lengths? I can't study long responses in pratice, but perhaps I can come up with a simple model of the Fisher kernel which would make some predictions.

Some more thoughs:

One immediate thing we can see is a separate scenario from the one identified by Cui et al in which certain sequences can cause the entropy to decrease a lot: If there is an sequence $t$, not necessarily in the batch of sequences used for updates, that has "fisher correlation" with sequences $t_i'$ used in the update, such that $\sum_i K(t, t_i') A(t_i')$ is negative, then this sequence will be made less likely - whether or not it was a good sequence! So incorrect sequences can punish other, yet-to-be-observed sequences, making unlikely but good sequences (good exploration) less likely. But unfortunately it's not immediately obvious to me how to make use of this observation.

Probably it would be good to have a more quantitative understanding of the Fisher Kernel. The Fisher Kernel is a product of two rectangular matrices $$K = w^T w$$ with $$w_{\alpha, t} = \nabla_\alpha \log \pi(t)$$ So if the number of sequences is larger than the number of model parameters, K is degenerate. The number of sequences $$V^L$$ for vocab of size V grows exponentially in L, so this is true even for short sequences. Really we should be concerned with just the typical set of sequences, of size $$exp(h L)$$ where h is the per token entropy, but even then for small sequence lengths, $$N_{seqs}^{(typical)} \gg N_{params}$$. So K must be very degenerate. Also, we expect a lot of structure in K - e.g. matrix elements between sequences with the same prompt are probably correlated. It would be nice to have a simple model of $K$, perhaps as some structured matrix with random noise. 



### Key Components

entropy_experiments - At least for now this is where have our offline experiments, where we load checkpoints and do some tests on them. Currently implemented is the a measurement of the predicted value of $\delta\mathcal{H}$ and the true value. As part of that we also measure an estimator of the variance of the estimator of $\delta\mathcal{H}$ so we can get a sense of what batch sizes we need to make a good comparison (this is forcing me to become familiar with more sophisticated stats stuff). Next we will add some measurments of the Fisher Kernel itself. See the README.md in this folder for a hopefully up-to-date explanation of the various components.

rl_training - Some training code so we can do online measurements during training, or save checkpoints and do offline measurements on them. Online experiments will be put in this folder, including measuring the gradient noise scale during training, and measuring the predicted entropy step change.




### Models & Datasets
- **Models**: Qwen2.5-1.5B (LoRA fine-tuned)
- **Dataset**: GSM8K mathematical reasoning with R1 template format
- **Training Results**: Pass@1 improvement from about 52% to 65@ over 64 steps, approximately linear growth in pass rate over this range of training with no signs of slowing down yet, but I stopped at 64 since that gave me enough checkpoints to do some experiments with (for now), and the training is getting expensive. Batch size was chosen based on measuring the gradient noise scale (not quite, but something close to it which was more convenient to measure), and learning rate chosen by starting at a value I saw used for a similar model and task, then increased by 2 when the gradient noise to signal ratio seemed low enough for it to be safe. It would be nice to do a more thorough search for optimal choices but I'm not made of money and it's not like I'm aiming to do an extended training run.
- **Environment**: Lambda Cloud GPU instances (e.g 2x H100 80GB)




