# Entropy vs Learning Rate Linearity Experiments

This directory contains scripts to test whether entropy changes are approximately linear in learning rate for small learning rates.

## Hypothesis

For small learning rates η, the entropy change ΔH after one optimization step should be approximately:

```
ΔH ≈ α * η + β
```

Where α is the slope (entropy sensitivity) and β is the intercept (baseline change).

## Scripts

### 1. `entropy_vs_learning_rate.py` (Main Script)

Comprehensive experiment runner that:
- Tests multiple learning rates (default: 1e-8 to 1e-4)  
- Runs multiple repeats per learning rate for statistical reliability
- Computes mean ± standard error for each learning rate
- Performs linear regression analysis
- Creates visualizations

**Usage:**
```bash
# Full experiment (recommended)
python scripts/entropy_vs_learning_rate.py \\
    --checkpoint /home/ubuntu/localfs/rl_training_runs/training_state/step_60 \\
    --base-config rl_training/cfg/h100_dual_gns_64step.yaml \\
    --learning-rates 1e-8,1e-7,1e-6,1e-5,1e-4 \\
    --num-repeats 5 \\
    --output-dir entropy_lr_results

# Custom learning rates
python scripts/entropy_vs_learning_rate.py \\
    --checkpoint /path/to/checkpoint \\
    --base-config rl_training/cfg/h100_dual_gns_64step.yaml \\
    --learning-rates 1e-6,2e-6,5e-6,1e-5,2e-5 \\
    --num-repeats 10 \\
    --output-dir fine_grained_entropy_test
```

### 2. `quick_entropy_lr_test.py` (Quick Test)

Simplified version for quick validation:
- Tests only 3 learning rates (1e-7, 1e-6, 1e-5)
- Only 3 repeats per learning rate  
- Good for initial testing

**Usage:**
```bash
python scripts/quick_entropy_lr_test.py \\
    --checkpoint /home/ubuntu/localfs/rl_training_runs/training_state/step_60 \\
    --output-dir quick_test
```

## How It Works

### Experiment Protocol

1. **Load Checkpoint**: Load model + optimizer state from previous training
2. **Configure Experiment**: Set specific learning rate, enable entropy measurement  
3. **Run 2 Steps**: Execute exactly 2 optimizer steps
4. **Measure Entropy**: Extract entropy before/after from logs
5. **Compute Change**: ΔH = entropy_after - entropy_before
6. **Repeat**: Run N times per learning rate for statistics

### Technical Details

- **Hardware**: Requires 2x H100 GPUs (uses distributed training)
- **Buffer Size**: 128 (optimized for 2x H100)
- **Microbatch Size**: 4 per GPU
- **Steps**: Exactly 2 steps for before/after measurement
- **Entropy Source**: Simple entropy probe logs

### Configuration Modifications

The script automatically modifies the base config:
- Sets `lr` to the test learning rate
- Sets `total_steps: 2` for before/after measurement  
- Sets `save_every: 1` to capture both steps
- Enables `simple_entropy_probe` for measurements
- Disables other probes to focus on entropy

## Output Structure

```
entropy_lr_results/entropy_lr_experiment_2025-08-28_10-30-45/
├── analysis.json                      # Summary statistics and linear fit
├── raw_results.json                   # All experiment results
├── entropy_vs_learning_rate.png       # Main visualization
├── raw_measurements.png               # Scatter plot of all data points
├── lr_1e-08_run_0/                   # Individual experiment results
│   └── result.json
├── lr_1e-07_run_0/
│   └── result.json
└── ...
```

## Expected Results

If the linearity hypothesis holds:
- **R² > 0.95**: Strong linear correlation
- **Small residuals**: Data points close to linear fit
- **Consistent slope**: Entropy sensitivity measure

If linearity breaks down:
- **Low R²**: Non-linear relationship
- **Large residuals**: Systematic deviations from linear fit
- **May indicate**: Higher-order effects become important

## Example Analysis Output

```
📊 LINEARITY ANALYSIS
==================================================
Linear fit: ΔH = 2.34e+05 * lr + -0.001234
R² = 0.9876
Slope = 2.34e+05 (entropy change per unit learning rate)

📋 DETAILED RESULTS
Learning Rate N   Mean ΔH      Std Error    Range               
------------------------------------------------------------
1e-08        5   -0.000234    0.000012     [-0.0003, -0.0002]
1e-07        5   -0.002340    0.000120     [-0.0025, -0.0021] 
1e-06        5   -0.023400    0.001200     [-0.0250, -0.0220]
1e-05        5   -0.234000    0.012000     [-0.2500, -0.2200]
1e-04        5   -2.340000    0.120000     [-2.5000, -2.2000]
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `buffer_size` in base config
2. **No entropy measurements**: Check simple entropy probe logs  
3. **Connection refused**: Ensure Lambda instance is running
4. **Missing checkpoint**: Verify checkpoint path and structure

### Debug Tips

- Use `quick_entropy_lr_test.py` first to validate setup
- Check individual `result.json` files for detailed logs
- Look for entropy probe output in training logs
- Monitor GPU memory usage during experiments

## Lambda Setup

```bash
# SSH to Lambda instance
ssh -i ~/.ssh/lambda_new ubuntu@YOUR_IP

# Navigate to project
cd ~/RL_Practice

# Pull latest changes
git pull

# Run experiment
python scripts/entropy_vs_learning_rate.py \\
    --checkpoint /home/ubuntu/localfs/rl_training_runs/training_state/step_60 \\
    --base-config rl_training/cfg/h100_dual_gns_64step.yaml \\
    --learning-rates 1e-8,1e-7,1e-6,1e-5,1e-4 \\
    --num-repeats 5 \\
    --output-dir entropy_lr_results
```

The experiment will automatically handle distributed training setup and produce comprehensive analysis results.