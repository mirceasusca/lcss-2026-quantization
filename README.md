# lcss-2026-quantization

Source code and experiment data for the L-CSS submission:

**Performance-Aware Quantizer Synthesis for LQI Tracking Subject to Fixed Level Budgets**

The main reproducibility script is [`cdc_lqi_quant.m`](cdc_lqi_quant.m). It implements the revised case study used in the manuscript, including:

- LQI design for the unstable second-order example;
- mixed step-ramp-sine reference generation with bounded multiplicative perturbations;
- training/validation datasets with 500 training trajectories and 250 validation trajectories;
- quantizer range certification consistent with the no-saturation certificate in the paper;
- sensitivity-weight computation from the discounted Lyapunov bound;
- globally optimal scalar DP quantizer construction;
- globally optimal DP allocation of the total level budget across channels;
- comparison against uniform, logarithmic companding, Lloyd-Max, high-rate allocation, signal-only allocation, and equal allocation baselines;
- LaTeX-ready statistics for the allocation-sensitivity and quantizer-type comparison tables;
- figures for tracking response, sample histograms, and quantizer centroids.

The saved experiment data are also provided in [`exp_LCSS_2026_05_17.mat`](exp_LCSS_2026_05_17.mat). This file stores the numerical data generated for the current revised experiment, so the reported tables and plots can be inspected without rerunning the full script.

## Running the Experiment

Run the script from MATLAB in this folder:

```matlab
run('cdc_lqi_quant.m')
```

The script was developed with MATLAB R2025b. It uses standard control-design routines such as `dlqr`, `dlyap`, `ss`, and `stepinfo`, so the Control System Toolbox is expected.

## Main Parameters

The current manuscript configuration is:

- sampling period: `Ts = 0.1`;
- discounted cost factor: `gamma = 0.9992`;
- simulation horizon: `160 s`, i.e., `1600` samples;
- training trajectories: `nTrainTraj = 500`;
- validation trajectories: `nTestTraj = 250`;
- total level budget: `Ntot = 384`;
- minimum per-channel level count for allocation: `NminAlloc = 16`;
- allocation-curve subset: `allocDesignMaxSamples = 60000` samples per channel;
- certified quantizer ranges: `(X_1,X_2,Y_1) = (9.15,15.79,17.15)`;
- proposed allocation: `(N_{x,1},N_{x,2},N_y) = (201,77,106)`.

For reproducibility, the allocation curves `Phi_l(N)` are computed on a deterministic subset of `6e4` samples per channel for `N in [16,352]`. The final DP quantizers are then recomputed on the full training set of `500*(1600+1)` samples per channel. The perturbed reference is generated as
`r_k^{\mathrm{reg}}=r_k(1+\delta_k)$, with $\delta_k\in[-0.5,0.5]` uniformly sampled when the perturbation event is active `p_{\mathrm{reg}}=0.10`. This is implemented by the line `r = r .* (1 + frac * (2*rand(size(r)) - 1))` with `frac = 0.50`.

## Current Reported Statistics

Allocation sensitivity for DP quantizers under `Ntot = 384`:

| Allocation | `(N_{x,1},N_{x,2},N_y)` | mean `J_gamma` | mean surrogate |
|---|---:|---:|---:|
| Proposed | `(201,77,106)` | **1.249** | **1.238** |
| High-rate | `(157,68,159)` | 1.605 | 1.615 |
| Signal-only | `(129,156,99)` | 2.677 | 2.698 |
| Equal-level | `(128,128,128)` | 2.479 | 2.503 |

Quantizer-type comparison using the proposed allocation:

| Method | mean `J_gamma` | std `J_gamma` | mean surrogate | std surrogate |
|---|---:|---:|---:|---:|
| Uniform | 177.8422 | 23.6404 | 83.3794 | 1.2689 |
| Logarithmic | 9.0158 | 0.7286 | 8.9300 | 0.2656 |
| Lloyd-Max | 2.1262 | 0.6090 | 2.1107 | 0.4360 |
| Proposed DP | **1.4106** | **0.2573** | **1.3997** | **0.1754** |

