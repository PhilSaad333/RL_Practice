# Entropy Influence Experiment Roadmap

## Motivation
We study how small optimizer updates change the sequence-level entropy estimates produced by `DeltaEntropyTrue`. Earlier sweeps showed that the overall entropy shift `DeltaH(eta)` for a full optimizer step behaves nearly linearly over learning rates `eta` in `[1e-6, 1e-4]`. Writing the update direction as `v = baseline + sum_i v_i` (baseline captures momentum and weight decay terms, each `v_i` comes from one sequence in the update batch), the linearized model predicts:

```
DeltaH(eta) / eta ≈ gradH * baseline + sum_i gradH * v_i.
```

Directly computing those inner products is difficult, but the linearization implies three measurable relations:

1. `eta * gradH * baseline ≈ H(theta + eta * baseline) - H(theta)`.
2. `eta * gradH * v_i ≈ H(theta + eta * v_i) - H(theta)`.
3. To reduce curvature error, use the mixed evaluation `H(theta + eta * baseline + eta * v_i) - H(theta + eta * baseline)` as a proxy for `eta * gradH * v_i`.

When these approximations hold, the per-sequence entropy deltas isolate how each training sequence pushes entropy up or down, enabling downstream correlation studies with sequence features.

## Current Focus
The fourth test run provides large-batch measurements at multiple learning rates with decompositions for:
- The full update direction (`full/eta_*.npy`).
- The baseline-only step (`baseline/eta_*.npy`).
- Individual gradient components (`grad_eta/delta_*.npy`).
- Mixed baseline-plus-sequence evaluations (`baseline_plus_grad/eta_*.npy`).

Our immediate goal is to quantify how well these components obey the linear model above and to diagnose deviations (e.g., curvature or numerical precision). Future analyses will build on these diagnostics to relate sequence attributes to their entropy influence.
