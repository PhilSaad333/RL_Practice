# Entropy Probe Results

This directory contains results from entropy probe experiments and studies.

## Directory Structure

- **`batch_size_scaling_TIMESTAMP/`** - Results from B_E batch size scaling studies
  - `results.json` - Raw experimental data
  - `SUMMARY.md` - Human-readable analysis and findings  
  - `batch_size_scaling.log` - Execution logs
  - `config_B_E_*.yaml` - Config files used for each batch size

## Current Studies

### Batch Size Scaling Study

Tests the new conditional variance estimator SE_E(δH₁|U) across different evaluation batch sizes (B_E) while keeping update batch size (B_U) fixed.

**Purpose:** Validate the new conditional variance estimator and determine optimal batch sizes for precision.

**Parameters:**
- B_E: [64, 128, 256, 512] (powers of 2)
- B_U: 128 (fixed, 64 per GPU)
- Multiple runs per batch size for statistical reliability
- Only conditional variance enabled (other estimators disabled)

**Expected Results:**
- SE_E(δH₁|U) should decrease with larger B_E (better precision)  
- δH₁ should remain stable (unbiased estimator)
- Relative SE should improve with larger batch sizes

## Running Studies

```bash
# Batch size scaling study
python entropy_experiments/run_batch_size_scaling_study.py \
  --config entropy_experiments/configs/test_conditional_variance_config.yaml \
  --checkpoint /home/ubuntu/localfs/stage3_checkpoints/step_40
```

## Experimental Context

These experiments test the entropy probe fixes from August 2025:

1. **Attention mask in forward pass** - Fixed padding token handling
2. **New conditional variance estimator** - Implements V_E|U from variance_estimator_change.txt  
3. **Granular config options** - Independent control over probe components
4. **Removed fractional variance** - Cleaner SE analysis vs confusing η²(V_X+V_Y)/δH₁²