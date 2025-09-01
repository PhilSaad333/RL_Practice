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