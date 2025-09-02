# Entropy Probe Logs

This directory contains comprehensive log files from entropy probe runs.

## Log Sources

### Comprehensive Logging Script
**Script**: `entropy_experiments/run_probe_with_comprehensive_logging.py`

**Purpose**: Detailed debugging and analysis of entropy probe execution with full error tracking.

**Log Format**: `entropy_probe_comprehensive_YYYYMMDD_HHMMSS.log`

**Contents**:
- Complete execution trace from initialization to completion
- Model loading diagnostics (LoRA parameters, optimizer state)
- Phase-by-phase timing (sampling, X/Y accumulation, deltaH1 computation)
- Configuration validation (rb_requires_grad enforcement, estimator modes)
- Gradient flow validation and error handling
- Final results (deltaH1, bars_dot, learning_rate)
- Crash analysis with full stack traces when failures occur

### Key Debugging Sessions
- **2025-09-01**: Complete entropy probe debugging and fixes
  - Fixed model initialization order bug
  - Resolved gradient computation issues (rb_requires_grad enforcement)
  - Fixed generation problems (top_p 1.0 → 0.995)
  - Cleaned up conditional variance references
  - **Final Success**: deltaH1 = 2.659, runtime = 3.5 minutes

## Log Analysis
Each comprehensive log contains:
1. **Setup Phase**: Model/optimizer loading with parameter diagnostics
2. **Phase 0**: E/U batch sampling with sequence generation
3. **Phase 1**: X gradient accumulation (entropy gradients)  
4. **Phase 2**: Y gradient accumulation (Adam preconditioned policy gradients)
5. **Phase 3**: deltaH1 computation (learning_rate × bars_dot)
6. **Results**: Final metrics and timing breakdown

## Success Criteria
A successful run should show:
- ✅ Model loading with correct LoRA parameter count (392 total)
- ✅ SequenceProcessor config: `rb_requires_grad=True` for rb_residual mode
- ✅ Phase 0 completion with proper batch shapes
- ✅ X/Y accumulation completion without gradient errors
- ✅ Final deltaH1 computation with finite result
- ✅ "Stage 1 Mixed probe analysis completed" success message

---

## New Detailed Logging System (2025-01)

### Overview
A comprehensive structured logging system has been added to `OfflineEntropyProbe` that captures detailed debugging information in JSON format.

### Configuration
Enable detailed logging in your config:
```yaml
detailed_logging:
  enabled: true
  level: "standard"              # minimal, standard, detailed, debug
  log_sequences: true            # Include sequence text and individual scores
  output_directory: "entropy_experiments/logs"
  compress: true                 # Gzip large files
```

### Log File Structure
Each run generates three files organized by date:
```
logs/YYYY-MM-DD/
├── entropy_probe_HH-MM-SS_checkpoint.json.gz    # Main detailed log
├── entropy_probe_HH-MM-SS_checkpoint_summary.json # Quick summary
└── entropy_probe_HH-MM-SS_checkpoint_config.yaml  # Exact config used
```

### Logging Levels

#### 1. Minimal (~1KB)
Core metrics only:
- bars_dot, deltaH1, deltaH_true
- Phase timings, batch sizes

#### 2. Standard (~5-10KB) 
Minimal + diagnostics:
- Batch statistics (generation lengths, rewards)
- Ground truth diagnostics (ESS, importance weights)

#### 3. Detailed (~50-500KB)
Standard + sequence data:
- Individual prompts and responses
- Per-sequence logprobs and RB entropies
- Importance sampling intermediate results

#### 4. Debug (~1-10MB)
Everything + raw data:
- Token-level data
- Raw tensor dumps
- Complete intermediate computations

### Usage Examples

```python
# Enable detailed logging
from entropy_experiments.offline_entropy_probe import OfflineEntropyProbe

probe = OfflineEntropyProbe.from_config_file('configs/detailed_logging_example.yaml')
results = probe.run_mixed_probe('/path/to/checkpoint')

# Log files automatically saved to entropy_experiments/logs/
```

### Log Analysis
```bash
# Quick results check
cat logs/2025-01-15/entropy_probe_14-30-25_step_60_summary.json | jq '.core_results'

# Detailed sequence analysis  
zcat logs/2025-01-15/entropy_probe_14-30-25_step_60.json.gz | jq '.sequences.E_batch[0:3]'

# Compare runs
for f in logs/2025-01-15/*_summary.json; do
  echo "=== $f ==="
  jq '.core_results | {deltaH1, deltaH_true, bars_dot}' "$f"
done
```

### Benefits
- **Complete debugging visibility**: All intermediate computations captured
- **Reproducibility**: Exact configs saved with results
- **Performance analysis**: Detailed timing breakdown
- **Research analysis**: Rich data for post-hoc analysis and plotting