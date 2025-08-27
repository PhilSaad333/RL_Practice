# 🎯 Entropy Experiments - Unified Test Runner

**One script to rule them all!** This directory contains the unified entropy probe test runner that replaces all previous overlapping scripts.

## 🚀 Quick Start

```bash
# Batch scaling test - test δH₁ convergence vs 1/B_E scaling
python run_entropy_test.py batch-scaling \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16,32,64 --runs 3 --verbose

# Sanity check - validate probe is working correctly  
python run_entropy_test.py sanity-check \\
  --checkpoint /path/to/checkpoint --runs 15

# Single test - clean individual run
python run_entropy_test.py single \\
  --checkpoint /path/to/checkpoint \\
  --be 256 --bu 32 --minimal

# Debug mode - detailed analysis with full logging
python run_entropy_test.py debug \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16,32 --verbose --save-gradients
```

## 📁 Directory Structure

```
entropy_experiments/
├── run_entropy_test.py              # 🎯 UNIFIED test runner (replaces 11 old scripts!)
├── README.md                        # This file
├── configs/                         # Clean configuration templates
│   ├── batch_scaling.yaml           # For δH₁ convergence testing
│   ├── sanity_check.yaml            # For validation runs
│   ├── single_run.yaml              # For individual tests  
│   └── debug.yaml                   # For detailed debugging
├── results/                         # Auto-organized results
│   ├── batch_scaling/               # Categorized by test type
│   │   └── δH₁_convergence_2025-08-27_22-30-15/
│   ├── sanity_check/
│   ├── single_run/
│   └── debug/
└── [core probe files]               # Essential probe components only
```

## 🧪 Test Types

### 1. Batch Scaling Test (`batch-scaling`)

**Purpose**: Test whether δH₁ converges to true value (good) or scales as 1/B_E (indicates bug)

**Key Features**:
- Tests multiple B_E values with fixed B_U
- Multiple runs per batch size for statistics
- Automatic convergence vs scaling analysis
- Focus on conditional variance estimator V_E(δH₁|U)

**Example Usage**:
```bash
# Test common scaling pattern
python run_entropy_test.py batch-scaling \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16,32,64,128 \\
  --runs 4 \\
  --config configs/batch_scaling.yaml \\
  --verbose

# Quick scaling test
python run_entropy_test.py batch-scaling \\
  --checkpoint /path/to/checkpoint \\
  --be-values 32,64 --runs 2 --minimal
```

**Output Analysis**:
- Per-batch statistics (mean δH₁, std, signal/noise ratio)
- Scaling analysis (convergence vs 1/B_E patterns)
- Individual run logs with gradient details (if `--verbose`)

### 2. Sanity Check (`sanity-check`)

**Purpose**: Validate that probe is working correctly with statistical analysis

**Key Features**:
- Single B_E value with many runs
- Comprehensive validation (tests all variance estimators)
- Statistical diagnostics and bias analysis
- Success rate reporting

**Example Usage**:
```bash
# Standard validation with 15 runs
python run_entropy_test.py sanity-check \\
  --checkpoint /path/to/checkpoint \\
  --runs 15 --be 256

# Quick validation  
python run_entropy_test.py sanity-check \\
  --checkpoint /path/to/checkpoint \\
  --runs 5 --minimal
```

### 3. Single Test (`single`)

**Purpose**: Clean individual entropy probe runs for focused testing

**Key Features**:
- Single B_E/B_U combination
- Fast execution
- Focus on conditional variance estimator
- Perfect for parameter exploration

**Example Usage**:
```bash
# Test specific batch configuration
python run_entropy_test.py single \\
  --checkpoint /path/to/checkpoint \\
  --be 256 --bu 32

# Quick test with minimal output
python run_entropy_test.py single \\
  --checkpoint /path/to/checkpoint \\
  --be 128 --bu 16 --minimal
```

### 4. Debug Mode (`debug`)

**Purpose**: Detailed analysis with comprehensive logging for debugging

**Key Features**:
- Tests all variance estimators for comparison
- Full debug logging (gradient sizes, parameter states)
- Optional gradient saving
- Small batch sizes for detailed visibility

**Example Usage**:
```bash
# Full debug analysis
python run_entropy_test.py debug \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16,32 \\
  --verbose --save-gradients

# Quick debug check
python run_entropy_test.py debug \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16 --minimal
```

## ⚙️ Configuration System

### Clean Config Templates

**❌ Old Confusing Format**:
```yaml
batch_config:
  B: 256                    # What does this mean??
  B_E: 256                  # Single value only
  B_U: 32
```

**✅ New Clean Format**:
```yaml
batch_config:
  B_E_values: [16, 32, 64]  # ✅ Support lists for scaling tests!
  B_U: 32                   # ✅ Clear, usually fixed
  runs_per_batch: 3         # ✅ Clear repetition control
  # Removed confusing "B" entirely!
```

### Template Usage

- **`configs/batch_scaling.yaml`**: Optimized for convergence testing
- **`configs/sanity_check.yaml`**: Comprehensive validation setup  
- **`configs/single_run.yaml`**: Fast individual test setup
- **`configs/debug.yaml`**: Full debug with detailed logging

Custom configs can override defaults:
```bash
python run_entropy_test.py batch-scaling \\
  --config my_custom_config.yaml \\
  --checkpoint /path/to/checkpoint
```

## 📊 Results Organization

### Auto-Categorized Structure

Results are automatically organized by test type with descriptive names:

```
results/
├── batch_scaling/
│   ├── δH₁_convergence_2025-08-27_22-30-15/     # Descriptive naming
│   │   ├── BE016_run001/, BE032_run001/, BE064_run001/
│   │   ├── convergence_analysis.json            # Scaling analysis
│   │   └── [individual run directories]
│   └── memory_scaling_2025-08-28_10-15-30/      # Another experiment
├── sanity_check/
│   └── validation_2025-08-27_22-30-15/
│       ├── run001/, run002/, ..., run015/
│       └── validation_summary.json
├── single_run/
│   └── single_run_2025-08-27_22-30-15/
│       ├── BE256_BU32/
│       │   ├── config.yaml                      # Exact config used
│       │   ├── probe_log.txt                    # Full debug logs
│       │   └── results.json                     # Probe JSON output
│       └── single_test_results.json
└── debug/
    └── debug_analysis_2025-08-27_22-30-15/
        ├── debug_BE016/, debug_BE032/
        └── debug_report.json
```

### Individual Run Structure

Each run contains complete artifacts:
- **`config.yaml`**: Exact configuration used
- **`probe_log.txt`**: Full Python logging (includes gradient sizes)
- **`results.json`**: Probe's JSON output with all metrics
- **`error.txt`**: If run failed, detailed error information

## 🔧 Command Line Options

### Common Arguments
- `--checkpoint`: Path to checkpoint directory (required)
- `--config`: Custom configuration file (optional)

### Logging Control
- `--verbose`: Enable DEBUG logging with gradient details
- `--minimal`: INFO logging with key results only
- `--quiet`: Final summary only

### Batch Configuration
- `--be-values`: Comma-separated B_E values (e.g., "16,32,64")
- `--bu`: B_U value (default: 32)
- `--runs`: Number of runs per configuration (default: 3)

### Single Test
- `--be`: B_E value for single test (default: 256)
- `--bu`: B_U value for single test (default: 32)

### Debug Options
- `--save-gradients`: Save detailed gradient information

## 🎯 Key Features

### Unified Interface
- **ONE script** replaces 11 overlapping scripts
- **Consistent command patterns** across all test types
- **No timeouts** - each run completes naturally
- **Graceful error handling** - failed runs don't stop tests

### Smart Results Management
- **Auto-categorization** by test type
- **Descriptive naming** - immediately understand purpose
- **Zero conflicts** - isolated timestamped experiments
- **Complete traceability** - every run fully logged

### Advanced Analysis
- **Convergence detection** - distinguishes convergence from 1/B_E scaling
- **Statistical validation** - comprehensive sanity checking
- **Signal/noise analysis** - |SE/δH₁| ratio monitoring
- **Performance tracking** - runtime and memory analysis

## 🔄 Migration from Old Scripts

### Old Scripts → New Commands

| Old Script | New Command |
|------------|-------------|
| `batch_size_convergence_test_*.py` | `python run_entropy_test.py batch-scaling` |
| `flexible_batch_test_*.py` | `python run_entropy_test.py batch-scaling` |
| `run_probe_sanity_check.py` | `python run_entropy_test.py sanity-check` |
| `debug_batch_scaling_test.py` | `python run_entropy_test.py debug` |
| Various single-run scripts | `python run_entropy_test.py single` |

### Result Migration

Old results can be migrated to new structure:
```bash
# Move old results to appropriate categories
mv results/test_fixed_logging → results/batch_scaling/legacy_test_2025-08-27/
mv results/BE016_run001 → results/single_run/legacy_BE016_2025-08-27/
```

## 🚨 Breaking Changes

1. **Removed confusing "B" parameter** - use `B_E_values` and `B_U` explicitly
2. **Config templates use lists** - `B_E_values: [16, 32, 64]` instead of single values
3. **Results auto-organized** - no more manual directory management
4. **Unified logging** - consistent log formats across all test types

## ⚡ Performance Tips

1. **Use `--minimal` for batch jobs** - reduces log volume
2. **Start with small B_E values** - faster iteration for debugging
3. **Use `debug` mode sparingly** - comprehensive but slower
4. **Batch scaling with `--quiet`** - for automated analysis

## 🐛 Troubleshooting

### Common Issues

**"Checkpoint not found"**: Ensure checkpoint path exists and contains `/model` directory
```bash
ls /path/to/checkpoint/model/  # Should show adapter files
```

**Memory errors**: Reduce batch sizes in config or use smaller `--be-values`
```bash
# Use smaller batch sizes
python run_entropy_test.py batch-scaling --be-values 8,16,32
```

**Import errors**: Ensure you're running from project root
```bash
cd /path/to/RL_Practice
python entropy_experiments/run_entropy_test.py --help
```

### Debug Mode

For detailed troubleshooting, always use debug mode:
```bash
python run_entropy_test.py debug \\
  --checkpoint /path/to/checkpoint \\
  --be-values 16 --verbose
```

## 📈 Future Enhancements

- [ ] Export results to CSV for analysis
- [ ] Auto-cleanup of old experiments (keep last 10)  
- [ ] Integration with tensorboard logging
- [ ] Distributed multi-GPU support
- [ ] Custom analysis plugins

---

**This unified system replaces 11 overlapping scripts with a single, clean, professional interface. No more confusion about which script to use!** 🎉