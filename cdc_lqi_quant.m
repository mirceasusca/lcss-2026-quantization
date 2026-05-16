%% case_study_quantized_lqi_dp.m
% Quantized LQI case study (2nd-order SISO)
% Compares scalar quantizer designs:
%   - Uniform
%   - Logarithmic (mu-law companded)
%   - Lloyd / K-means (1D weighted)
%   - Dynamic Programming (globally optimal weighted 1D)

% DONE: Works correctly if you let the steady-state stay enough 
% in the trajectory
% DONE: define a number of full trajectories + 
% define a number of initializations to the first part of the trajectory
% DONE: added regularization through reference definition and state update

if ~exist('runAllocationSensitivityOnly','var')
    runAllocationSensitivityOnly = false;
end
clearvars -except runAllocationSensitivityOnly;
clc; close all;
rng(7);

%% =========================
% 1) Plant + LQI controller
% ==========================
Ts = 0.1;

% UNSTABLE + NONMINIMUM PHASE
A = [1.2,0.4;
    -0.3,0.9];
B = [1;0.5];
C = [1,1];
D = 0;

n = size(A,1);
m = size(B,2);
p = size(C,1);

% Augmented nominal (unquantized) servo dynamics
% xi = [x; z],  z_{k+1} = z_k + Ts (r_k - y_k)
Aaug = [A,            zeros(n,p);
       -Ts*C,         eye(p)];
Baug = [B; zeros(p,m)];

% LQI/LQR weights on augmented state and command
% vastly different weights to have slow and fast closed-loop dynamics
% Qaug = diag([12, 1.5, 25]);
Qaug = diag([8, 1, 16]);
R = 1; 

% MATLAB dlqr uses u = -Kdlqr * xi
Kdlqr = dlqr(Aaug, Baug, Qaug, R);
Kfull = -Kdlqr;            % paper convention u = K * xi
Kx = Kfull(:,1:n);         % 1x2
Kz = Kfull(:,n+1:end);     % 1x1

% Closed-loop matrices in manuscript form
Acl = [A + B*Kx,   B*Kz;
      -Ts*C,       1];
Gx = [B*Kx;
      zeros(p,n)];          % size (n+p) x n = 3x2
Gy = [zeros(n,p);
     -Ts*eye(p)];           % size (n+p) x p = 3x1
E = [zeros(n,p); Ts*eye(p)]; % not used here (regulation form)

fprintf('\n=== Open-loop checks ===\n');
fprintf('eig(Acl) = \n');
disp(eig(A).');
fprintf('zero(ss(A,B,C,D))) = \n');
disp(zero(ss(A,B,C,D,Ts)))

% figure,step(ss(A,B,C,D,Ts),5)

fprintf('\n=== Closed-loop checks ===\n');
fprintf('eig(Acl) = \n');
disp(eig(Acl).');
assert(all(abs(eig(Acl)) < 1), 'Acl is not Schur. Retune Qaug/R.');

figure('Name','Closed-loop ideal step response')
step(ss(Acl,E,[C,0],0,Ts))

% settling time to know how much the step response should take
stepperf = stepinfo(ss(Acl,E,[C,0],0,Ts));
settlingTime = stepperf.SettlingTime

%% ============================================
% Reference signal for evaluation/training
% ============================================

% num_windows = 3
% gamma = 0.98;   % discounted command-distortion factor
% refCfg.type = 'const'; refCfg.r0 = 0.0;  % stabilization to origin
% x0_box     = [1.5; 1.5]*2;   % random x0 in [-box, box]
% z0_box     = 0.8*2;          % random z0 in [-z0_box, z0_box]

% num_windows = 3
% gamma = 0.98;   % discounted command-distortion factor
% refCfg.type = 'const'; refCfg.r0 = -0.7;
% x0_box     = [1.5; 1.5]*2;   % random x0 in [-box, box]
% z0_box     = 0.8*2;          % random z0 in [-z0_box, z0_box]

% num_windows = 5
% gamma = 0.99;   % discounted command-distortion factor
% refCfg.type = 'sine'; refCfg.amp = 0.4; refCfg.freq = 0.15/10/4; refCfg.bias = 0.4;
% x0_box     = [1.5; 1.5]*2;   % random x0 in [-box, box]
% z0_box     = 0.8*2;          % random z0 in [-z0_box, z0_box]
% 
% % num_windows = 6
% num_windows = 5
% min_settlingTime_window = ceil(settlingTime/10)*10
% num_samples_per_settlingTime_window = round(min_settlingTime_window/Ts);
% tfin_sim = min_settlingTime_window*num_windows
% gamma = 0.995;   % discounted command-distortion factor
% % refCfg.type = 'piecewise'; refCfg.switch_k = [0 1 2 3 4 5]*num_samples_per_settlingTime_window; refCfg.values   = [0.0 0.7 -0.4 0.6 0.2 0.2]*2;  % long final constant segment helps show quantization limit cycles
% refCfg.type = 'piecewise'; refCfg.switch_k = [0 1 2 3 4]*num_samples_per_settlingTime_window; refCfg.values   = [0.0 0.7 -0.4 0.6 0.2]*2;  % long final constant segment helps show quantization limit cycles
% % x0_box     = [10,10];   % random x0 in [-box, box]
% % z0_box     = 10;       % random z0 in [-z0_box, z0_box]
% x0_box     = [5,5];   % random x0 in [-box, box]
% z0_box     = 1;       % random z0 in [-z0_box, z0_box]

%
num_windows = 4
min_settlingTime_window = ceil(settlingTime/10)*10*2
num_samples_per_settlingTime_window = round(min_settlingTime_window/Ts);
tfin_sim = min_settlingTime_window*num_windows
% gamma = 0.997;   % discounted command-distortion factor
% gamma = 1;   % discounted command-distortion factor
gamma = 0.9992;   % discounted command-distortion factor
refCfg.type = 'mixed';
refCfg.switch_k = [0 1 2]*num_samples_per_settlingTime_window;
refCfg.segType = {'step','ramp','sine'};
ramp_max = 0.6;
refCfg.segPar  = {
    struct('value', -0.7),                         ... % step to -0.7
    struct('start', -0.7, 'slope', ramp_max/min_settlingTime_window/4.5/1.1,'min',-0.7,'max',ramp_max),        ... % ramp: -0.4 + 0.002*tau
    struct('bias',0.5, 'amp', 0.3, 'freq', 0.01/2,'phase',pi), ... % sine in segment time
};
refCfg.trainPerturbProb = 0.10;  % bounded reference regularization
refCfg.trainPerturbFrac = 0.50;  % |r_train| <= (1+frac)|r_nom|
x0_box     = [5,5];   % random x0 in [-box, box]
z0_box     = 1;       % random z0 in [-z0_box, z0_box]


%%
% min_settlingTime_window = ceil(settlingTime/10)*10
% num_samples_per_settlingTime_window = round(min_settlingTime_window/Ts);
% tfin_sim = min_settlingTime_window*num_windows

%% =========================================
% 2) Discounted Lyapunov matrix + sensitivities
% ============================================
% Theorem 2 quantities
% Qu = K' R K (here R scalar)
% Qu = Kfull' * R * Kfull; % old; u only
Qdelta = Qaug; 
Qu = Qdelta + Kfull' * R * Kfull;
Qu = (Qu+Qu')/2;

% Discounted Lyapunov equation:
% Su = Qu + gamma * Acl' * Su * Acl
% MATLAB dlyap solves: A X A' - X + Q = 0  => X = A X A' + Q
% Use A = sqrt(gamma)*Acl'
Su = dlyap(sqrt(gamma)*Acl', Qu);

Wx_u = Gx' * Su * Gx + Kx' * R * Kx;   % 2x2
Wy_u = Gy' * Su * Gy;                  % 1x1

% Corollary row-sum weights (sufficient separable bound)
alpha_x = diag(Wx_u) + (sum(abs(Wx_u),2) - abs(diag(Wx_u)));
alpha_y = diag(Wy_u) + (sum(abs(Wy_u),2) - abs(diag(Wy_u)));

fprintf('\n=== Sensitivity matrices ===\n');
disp('Wx_u ='); disp(Wx_u);
disp('Wy_u ='); disp(Wy_u);
disp('alpha_x ='); disp(alpha_x.');
disp('alpha_y ='); disp(alpha_y.');

%% ===================================
% 3) Training data from ideal LQI loop
% ====================================
% Regulation / deviation case (r_k = 0)
% We collect ideal trajectories and design quantizers from x,y samples.

nTrainTraj = 500;
% nTrainTraj = 200;
% nTrainTraj = 100;
%
% horizon for data collection
% Htrain     = 100;   
% Htrain     = 750;
% Htrain     = 600;
Htrain = round(tfin_sim/Ts);
%
% truncated discounted horizon
nTestTraj = 250;
% nTestTraj = 50;
% nTestTraj = 20;

% Heval     = 180;
Heval     = Htrain;  % truncated discounted horizon

% Total finite-level budget across x1, x2, and y.
Ntot = 384;

train = collect_ideal_dataset_ref(A, B, C, Kx, Kz, Ts, nTrainTraj, ...
    Htrain, x0_box, z0_box, gamma, refCfg);

% Training arrays
x1_samples = train.x(:,1);
x2_samples = train.x(:,2);
y_samples  = train.y(:);
omega_t    = train.w(:);   % discount-consistent weights gamma^k

% Prescribed quantizer ranges. We keep the original data-envelope logic as a
% practical operating-set estimate, then certify it against Lemma 1 using the
% bounded initial set, bounded reference, and prescribed preliminary error
% envelopes. The larger of the two is used for synthesis.
range_margin = 1.25;
Sx_data = range_margin * [max(abs(x1_samples)), max(abs(x2_samples))];
Sy_data = range_margin * max(abs(y_samples));

xi0bar = [x0_box(:); z0_box];
rbar = reference_bound_over_horizon(refCfg, Htrain);
prelimLevelsForRange = max(1, floor(Ntot / (n + p)));
prelim_ebar_x = Sx_data(:) / prelimLevelsForRange;
prelim_ebar_y = Sy_data / prelimLevelsForRange;
Hx_cert = [eye(n), zeros(n,p)];
Hy_cert = [C, zeros(p,p)];
[Sx_cert, Sy_cert, rangeCert] = no_saturation_range_certificate( ...
    Acl, E, Gx, Gy, Hx_cert, Hy_cert, xi0bar, rbar, ...
    prelim_ebar_x, prelim_ebar_y);

Sx = max(Sx_data(:), Sx_cert(:)).';
Sy = max(Sy_data, Sy_cert);

fprintf('\n=== Lemma range certificate ===\n');
fprintf('xi0bar = [%.4f %.4f %.4f], rbar = %.4f\n', xi0bar(1), xi0bar(2), xi0bar(3), rbar);
fprintf('preliminary ebar = [%.4e %.4e %.4e]\n', prelim_ebar_x(1), prelim_ebar_x(2), prelim_ebar_y);
fprintf('data envelope S = [%.4f %.4f %.4f]\n', Sx_data(1), Sx_data(2), Sy_data);
fprintf('lemma certificate S = [%.4f %.4f %.4f] (truncation k=%d, tail %.2e)\n', ...
    Sx_cert(1), Sx_cert(2), Sy_cert, rangeCert.kStop, rangeCert.tailNorm);
fprintf('synthesis ranges S = [%.4f %.4f %.4f]\n', Sx(1), Sx(2), Sy);

% Enforce samples inside prescribed ranges (for design consistency)
assert(all(abs(x1_samples) <= Sx(1)+1e-12));
assert(all(abs(x2_samples) <= Sx(2)+1e-12));
assert(all(abs(y_samples)  <= Sy   +1e-12));

%% Fixed per-channel level counts
% % MANUAL MODE
% Nx = [9, 9];   % x1, x2
% Ny = 9;        % y
% Nx = [64, 64];   % x1, x2
% Ny = 64;        % y
% Nx = [128, 128];   % x1, x2 %% recommended (for same allocation for all)
% Ny = 128;        % y
% Nx = [211, 113];   % x1, x2 %%
% Ny = 60;        % y
% Nx = [256, 256];   % x1, x2
% Ny = 256;        % y

% High-rate allocation baseline (classical DSP-inspired rule).
% Alternative total budgets used in previous sweeps:
% Ntot = 768; Ntot = 320; Ntot = 256; Ntot = 192;
% % %
% Qdelta2 = Qaug*0; 
Qdelta2 = Qaug;
Qu2 = Qdelta2 + Kfull' * R * Kfull;
Qu2 = (Qu2+Qu2')/2;
Su2 = dlyap(sqrt(gamma)*Acl', Qu2);
Wx_u2 = Gx' * Su2 * Gx + Kx' * R * Kx;  % 2x2
Wy_u2 = Gy' * Su2 * Gy;                  % 1x1
alpha_x = diag(Wx_u2) + (sum(abs(Wx_u2),2) - abs(diag(Wx_u2)));
alpha_y = diag(Wy_u2) + (sum(abs(Wy_u2),2) - abs(diag(Wy_u2)));
% % % %
S = [Sx(1); Sx(2); Sy];
alpha = [alpha_x(1); alpha_x(2); alpha_y(1)];
c = alpha .* (S.^2);
w = c.^(1/3);
N = Ntot * w / sum(w);
Nint = round(N); 
% fix rounding to sum exactly:
delta = Ntot - sum(Nint);
[~,idx] = sort(N - Nint,'descend');
Nint(idx(1:abs(delta))) = Nint(idx(1:abs(delta))) + sign(delta);

N_highrate = [Nint(1), Nint(2), Nint(3)];

% Weighted scalar design weights for the separable sampled surrogate.
w_x1 = omega_t * alpha_x(1);
w_x2 = omega_t * alpha_x(2);
w_y  = omega_t * alpha_y(1);

% Data-driven globally optimal level allocation for the sampled separable
% surrogate: min sum_l Phi_l(N_l), sum_l N_l = Ntot.
useDataDrivenAllocation = true;
allocDesignMaxSamples = 60000; % deterministic subset for fast allocation curves
NminAlloc = 16;
allocOpt = [];
allocSignalOpt = [];
if useDataDrivenAllocation
    allocOpt = design_surrogate_level_allocation( ...
        x1_samples, x2_samples, y_samples, w_x1, w_x2, w_y, ...
        Sx, Sy, Ntot, NminAlloc, allocDesignMaxSamples, ...
        'performance-aware surrogate');

    % Signal-only allocation ablation: same scalar DP machinery, but without
    % the closed-loop sensitivity weights alpha_l. This isolates the benefit
    % of allocating levels by the performance surrogate rather than raw MSE.
    allocSignalOpt = design_surrogate_level_allocation( ...
        x1_samples, x2_samples, y_samples, omega_t, omega_t, omega_t, ...
        Sx, Sy, Ntot, NminAlloc, allocDesignMaxSamples, ...
        'signal-only MSE');

    N_selected = allocOpt.Nbest;
else
    N_selected = N_highrate;
end

Nx = [N_selected(1), N_selected(2)];
Ny = N_selected(3);
% Nx = [89+20,37]
% Ny = [66-20]

fprintf('\n=== Quantizer ranges / levels ===\n');
fprintf('Sx = [%.4f, %.4f], Sy = %.4f\n', Sx(1), Sx(2), Sy);
fprintf('High-rate allocation = [%d, %d, %d]\n', N_highrate(1), N_highrate(2), N_highrate(3));
if ~isempty(allocOpt)
    fprintf('Data-driven allocation = [%d, %d, %d] (curve surrogate %.6e)\n', ...
        allocOpt.Nbest(1), allocOpt.Nbest(2), allocOpt.Nbest(3), allocOpt.bestCost);
end
if ~isempty(allocSignalOpt)
    fprintf('Signal-only allocation = [%d, %d, %d] (curve SSE %.6e)\n', ...
        allocSignalOpt.Nbest(1), allocSignalOpt.Nbest(2), allocSignalOpt.Nbest(3), allocSignalOpt.bestCost);
end
fprintf('Selected Nx = [%d, %d], Ny = %d\n', Nx(1), Nx(2), Ny);
fprintf('Ntot = %d\n', Nx(1)+Nx(2)+Ny);

% Shared robustness-oriented validation set: independently sampled initial
% conditions and bounded reference perturbations, reused by all methods.
testset = sample_initial_conditions(nTestTraj, x0_box/3, z0_box/3);
testRefSeq = sample_reference_sequences(nTestTraj, Heval, refCfg, true);

%% =============================================
% 4) Design scalar quantizers (4 methods/channel)
% =============================================
methods = {'uniform','log','kmeans','dp'};
Qset = struct();

%% =====================================================
% 4a) Allocation perturb-and-observe sensitivity study
% ======================================================
% This section can be run standalone by calling:
% runAllocationSensitivityOnly = true; run('cdc_lqi_quant.m')
runAllocSensitivity = true;

if runAllocSensitivity
    % allocEvalTraj = min(60, nTestTraj);
    allocEvalTraj = nTestTraj;
    allocDesignMaxSamples = 60000; % deterministic subset for fast DP redesigns
    allocTestset = testset(1:allocEvalTraj,:);
    allocRefSeq = testRefSeq(1:allocEvalTraj,:);
    N_signal_baseline = [];
    if ~isempty(allocSignalOpt)
        N_signal_baseline = allocSignalOpt.Nbest;
    end

    allocSensitivity = allocation_perturb_observe( ...
        A, B, C, Kx, Kz, Ts, Qdelta, R, gamma, Wx_u, Wy_u, ...
        refCfg, Heval, Sx, Sy, [Nx(1), Nx(2), Ny], Ntot, ...
        x1_samples, x2_samples, y_samples, w_x1, w_x2, w_y, ...
        allocTestset, allocRefSeq, allocDesignMaxSamples, N_highrate, N_signal_baseline);

    if runAllocationSensitivityOnly
        return;
    end
end

% x1
Qset.uniform.x{1} = design_uniform_quantizer(Sx(1), Nx(1));
Qset.log.x{1}     = design_best_log_quantizer(x1_samples, w_x1, Sx(1), Nx(1));
Qset.kmeans.x{1}  = design_lloyd1d_quantizer(x1_samples, w_x1, Sx(1), Nx(1));
Qset.dp.x{1}      = design_dp1d_quantizer(x1_samples, w_x1, Sx(1), Nx(1));

% x2
Qset.uniform.x{2} = design_uniform_quantizer(Sx(2), Nx(2));
Qset.log.x{2}     = design_best_log_quantizer(x2_samples, w_x2, Sx(2), Nx(2));
Qset.kmeans.x{2}  = design_lloyd1d_quantizer(x2_samples, w_x2, Sx(2), Nx(2));
Qset.dp.x{2}      = design_dp1d_quantizer(x2_samples, w_x2, Sx(2), Nx(2));

% y
Qset.uniform.y{1} = design_uniform_quantizer(Sy, Ny);
Qset.log.y{1}     = design_best_log_quantizer(y_samples, w_y, Sy, Ny);
Qset.kmeans.y{1}  = design_lloyd1d_quantizer(y_samples, w_y, Sy, Ny);
Qset.dp.y{1}      = design_dp1d_quantizer(y_samples, w_y, Sy, Ny);

dp_ebar_x = [max_cell_error(Qset.dp.x{1}); max_cell_error(Qset.dp.x{2})];
dp_ebar_y = max_cell_error(Qset.dp.y{1});
[Sx_recert, Sy_recert] = no_saturation_range_certificate( ...
    Acl, E, Gx, Gy, Hx_cert, Hy_cert, xi0bar, rbar, dp_ebar_x, dp_ebar_y);
fprintf('\n=== Final DP error-envelope check ===\n');
fprintf('DP max-cell ebar = [%.4e %.4e %.4e]\n', dp_ebar_x(1), dp_ebar_x(2), dp_ebar_y);
fprintf('rechecked lemma S = [%.4f %.4f %.4f]\n', Sx_recert(1), Sx_recert(2), Sy_recert);
if any(Sx_recert(:) > Sx(:) + 1e-9) || any(Sy_recert(:) > Sy(:) + 1e-9)
    fprintf(['Note: the global max-cell recheck is conservative for the ' ...
        'unconstrained nonuniform DP tails and exceeds the preliminary ' ...
        'certificate. Validation below reports the actual overload rate.\n']);
end

%% ============================================
% 5) Training surrogate SSE (per method, design)
% =============================================
trainSSE = struct();
for im = 1:numel(methods)
    disp('Training surrogate SSE (per method, design)')
    im
    nm = methods{im};
    qx1 = Qset.(nm).x{1};
    qx2 = Qset.(nm).x{2};
    qy  = Qset.(nm).y{1};

    Jx1 = weighted_sse(x1_samples, w_x1, qx1);
    Jx2 = weighted_sse(x2_samples, w_x2, qx2);
    Jy  = weighted_sse(y_samples,  w_y,  qy);

    trainSSE.(nm) = [Jx1, Jx2, Jy, Jx1+Jx2+Jy];
end

fprintf('\n=== Training weighted scalar surrogate SSE (x1, x2, y, total) ===\n');
for im = 1:numel(methods)
    nm = methods{im};
    vals = trainSSE.(nm);
    fprintf('%-7s : %.6e   %.6e   %.6e   | total = %.6e\n', nm, vals(1), vals(2), vals(3), vals(4));
end

%% ==========================================
% 6) Closed-loop evaluation on test trajectories
% ===========================================
results = struct();
for im = 1:numel(methods)
    disp('Closed-loop evaluation on test trajectories')
    im
    nm = methods{im};
    qbundle = Qset.(nm);

    out = evaluate_quantized_case_ref( ...
        A, B, C, Kx, Kz, Ts, Qdelta, R, gamma, ...
        Wx_u, Wy_u, qbundle, testset, Heval, refCfg, Sx, Sy, testRefSeq);

    results.(nm) = out;
end

fprintf('\n=== Test-set averages (discounted, truncated to Heval) ===\n');
fprintf('Method   avg Ju_gamma      avg surrogate      overload rate\n');
for im = 1:numel(methods)
    nm = methods{im};
    r = results.(nm);
    fprintf('%-7s  %.6e   %.6e   %.4f\n', ...
        nm, mean(r.Ju), mean(r.Jsurr), r.overload_count / max(1,r.total_quant_calls));
end
fprintf('Proposed DP validation max |x|, |y| = [%.4f %.4f %.4f]\n', ...
    results.dp.max_abs_x(1), results.dp.max_abs_x(2), results.dp.max_abs_y);
fprintf('Proposed DP validation max |e_x|, |e_y| = [%.4e %.4e %.4e]\n', ...
    results.dp.max_abs_ex(1), results.dp.max_abs_ex(2), results.dp.max_abs_ey);

fprintf('\n=== LaTeX quantizer-type comparison table ===\n');
fprintf('Method & mean $J_{\\gamma}$ & std $J_{\\gamma}$ & mean surr. & std surr. \\\\'); fprintf('\n');
for im = 1:numel(methods)
    nm = methods{im};
    r = results.(nm);
    prettyName = method_pretty_name(nm);
    fprintf('%s & %.4f & %.4f & %.4f & %.4f \\\\', ...
        prettyName, mean(r.Ju), std(r.Ju), mean(r.Jsurr), std(r.Jsurr));
    fprintf('\n');
end

%% ===========================
% 7) Simple comparison plots
% ===========================
Ju_mean   = zeros(numel(methods),1);
Js_mean   = zeros(numel(methods),1);
Ju_std    = zeros(numel(methods),1);
Js_std    = zeros(numel(methods),1);
ovr_rate  = zeros(numel(methods),1);

for im = 1:numel(methods)
    nm = methods{im};
    Ju_mean(im)  = mean(results.(nm).Ju);
    Js_mean(im)  = mean(results.(nm).Jsurr);
    Ju_std(im)   = std(results.(nm).Ju);
    Js_std(im)   = std(results.(nm).Jsurr);
    ovr_rate(im) = results.(nm).overload_count / max(1, results.(nm).total_quant_calls);
end

figure('Name','Discounted command distortion');
bar(Ju_mean); hold on;
% bar(log10(Ju_mean)); hold on;
errorbar(1:numel(methods), Ju_mean, Ju_std, '.k', 'LineWidth', 1);
% errorbar(1:numel(methods), log10(Ju_mean), log10(Ju_std), '.k', 'LineWidth', 1);
set(gca,'XTick',1:numel(methods),'XTickLabel',methods);
ylabel('Average J_{u,\gamma} (test set)');
title('Discounted command distortion comparison');
grid on;

figure('Name','Theorem surrogate');
bar(Js_mean); hold on;
% bar(log10(Js_mean)); hold on;
errorbar(1:numel(methods), Js_mean, Js_std, '.k', 'LineWidth', 1);
% errorbar(1:numel(methods), log10(Js_mean), log10(Js_std), '.k', 'LineWidth', 1);
set(gca,'XTick',1:numel(methods),'XTickLabel',methods);
ylabel('Average surrogate sum');
title('Weighted error surrogate comparison');
grid on;

figure('Name','Overload rate');
bar(ovr_rate);
set(gca,'XTick',1:numel(methods),'XTickLabel',methods);
ylabel('Overload rate');
title('Quantizer range violations (should be near zero)');
grid on;

% Example quantizer visualization (x1)
figure('Name','x1 quantizers');
hold on; grid on;
xx = linspace(-Sx(1), Sx(1), 2000);
plot(xx, quantize_vec(xx, Qset.uniform.x{1}), 'LineWidth', 1.2);
plot(xx, quantize_vec(xx, Qset.log.x{1}),     'LineWidth', 1.2);
plot(xx, quantize_vec(xx, Qset.kmeans.x{1}),  'LineWidth', 1.2);
plot(xx, quantize_vec(xx, Qset.dp.x{1}),      'LineWidth', 1.5);
legend(methods, 'Location','best');
xlabel('x_1'); ylabel('q(x_1)');
title('Designed scalar quantizers for x_1');

disp('Done.');

%% ==========================================
% 8) Example trajectory plots (same x0, same r)
% ===========================================
x0_example = [0.4; -0.2; 0.1];  % [x1;x2;z]

traj = struct();
for im = 1:numel(methods)
    nm = methods{im};
    traj.(nm) = simulate_single_trajectory_ref( ...
        A, B, C, Kx, Kz, Ts, R, gamma, Wx_u, Wy_u, ...
        Qset.(nm), x0_example, Heval, refCfg);
end

% Ideal trajectory (reuse from one of the structs)
t = traj.uniform.k;
rseq = traj.uniform.r;
y_id = traj.uniform.y_ideal;
u_id = traj.uniform.u_ideal;

figure('Name','Tracking trajectories');
subplot(2,1,1); hold on; grid on;
plot(t, rseq, '--', 'LineWidth', 1.2);
plot(t, y_id, 'k', 'LineWidth', 1.5);
for im = 1:numel(methods)
    nm = methods{im};
    plot(t, traj.(nm).y, 'LineWidth', 1.0);
end
legend(['ref','ideal', methods], 'Location','best');
xlabel('k'); ylabel('y_k');
title('Output trajectories under arbitrary reference');

subplot(2,1,2); hold on; grid on;
plot(t, u_id, 'k', 'LineWidth', 1.5);
for im = 1:numel(methods)
    nm = methods{im};
    plot(t, traj.(nm).u, 'LineWidth', 1.0);
end
legend(['ideal', methods], 'Location','best');
xlabel('k'); ylabel('u_k');
title('Command trajectories');

% Tail zoom to reveal steady quantization effects / limit-cycle-like behavior
Ktail = 120;                    % show last Ktail samples for limit-cycle/chattering view
% idxTail = max(1, numel(t)-Ktail+1):numel(t);
idxTail = 1:numel(t);  % all

figure('Name','Tail zoom for quantization effects');
subplot(3,1,1); hold on; grid on;
plot(t(idxTail), rseq(idxTail), '--', 'LineWidth', 1.2);
plot(t(idxTail), y_id(idxTail), 'k', 'LineWidth', 1.3);
for im = 1:numel(methods)
    nm = methods{im};
    plot(t(idxTail), traj.(nm).y(idxTail), 'LineWidth', 1.0);
end
legend(cell(['ref','ideal',methods]), 'Location','best');
xlabel('k'); ylabel('y_k');
title('Zoom (output)');

subplot(3,1,2); hold on; grid on;
for im = 1:numel(methods)
    nm = methods{im};
    plot(t(idxTail), traj.(nm).u(idxTail) - u_id(idxTail), 'LineWidth', 1.0);
    disp(['>> RMS command distortion of ',nm,': ',num2str(sum((traj.(nm).u(idxTail) - u_id(idxTail)).^2))])
end
legend(methods, 'Location','best');
xlabel('k'); ylabel('\delta u_k');
title('Tail zoom (command distortion)');

subplot(3,1,3); hold on; grid on;
plot(traj.uniform.x1_ideal, traj.uniform.x2_ideal, 'k', 'LineWidth', 1.3);
for im = 1:numel(methods)
    nm = methods{im};
    plot(traj.(nm).x1(idxTail), traj.(nm).x2(idxTail), '.-', 'LineWidth', 0.8);
end
legend(cell(['ideal',methods]), 'Location','best');
xlabel('x_1'); ylabel('x_2');
title('Tail phase portrait (quantization-induced small cycles/chatter)');

% alternative logscale figure to subplot(3,1,2)
figure('Name','Command distortion logscale')
hold on; grid on;
for im = 1:numel(methods)
    nm = methods{im};
    plot(t(idxTail), db(abs(traj.(nm).u(idxTail) - u_id(idxTail))), 'LineWidth', 1.0);
    % yline(mean(db(abs(traj.(nm).u(idxTail) - u_id(idxTail)))),'--')
    disp(['>> RMS abs value of command distortion of ',nm,': ',num2str(sum((traj.(nm).u(idxTail) - u_id(idxTail)).^2))])
end
legend(methods, 'Location','best');
markertype = {'x','o','<','>'};
for im = 1:numel(methods)
    nm = methods{im};
    % plot(t(idxTail), db(abs(traj.(nm).u(idxTail) - u_id(idxTail))), 'LineWidth', 1.0);
    plot(t(idxTail),mean(db(abs(traj.(nm).u(idxTail) - u_id(idxTail)))),['--',markertype{im}])
    % disp(['>> RMS abs value of command distortion of ',nm,': ',num2str(sum((traj.(nm).u(idxTail) - u_id(idxTail)).^2))])
end
legend([methods,methods], 'Location','best');
xlabel('k'); ylabel('|\delta u_k| [dB]');
title('Tail zoom (command distortion)');

%% ==========================================
% 9) Centroid spacing plots (1D) for all quantizers
% ===========================================
plot_centroid_stems(Qset, methods, 'x_1 centroids', 'x', 1);
plot_centroid_stems(Qset, methods, 'x_2 centroids', 'x', 2);
plot_centroid_stems(Qset, methods, 'y centroids',   'y', 1);

% plot_centroid_stems_diff(Qset, methods, 'x_1 centroids', 'x', 1);
% plot_centroid_stems_diff(Qset, methods, 'x_2 centroids', 'x', 2);
% plot_centroid_stems_diff(Qset, methods, 'y centroids',   'y', 1);

%% ==========================================
% 10) Optional 2D partition plots (rectangular cells, not Voronoi)
% ===========================================
for im = 1:numel(methods)
    nm = methods{im};
    figure('Name',['State quantizer partitions in (x1,x2) - ',nm]);
    % subplot(2,2,im); 
    hold on; grid on; axis equal;
    plot_rect_partition(Qset.(nm).x{1}, Qset.(nm).x{2});
    title(sprintf('%s (rectangular partition)', nm));
    xlabel('x_1'); ylabel('x_2');
end

%% ==========================================
% 11) LCSS PLOT (Short summary of others)
% ===========================================
figure('Name','LCSS')

% subplot(2,3,[1,2,3]), hold on; grid on;
subplot(3,3,[1,2,3]), hold on; grid on;
plot(t(idxTail), rseq(idxTail), 'k--', 'LineWidth', 1.2);
% plot(t(idxTail), y_id(idxTail), 'k', 'LineWidth', 1.3);
for im = 1:numel(methods)
    nm = methods{im};
    if im == 4
        plot(t(idxTail), traj.(nm).y(idxTail), 'b', 'LineWidth', 1.5);
    elseif im == 1
        plot(t(idxTail), traj.(nm).y(idxTail), 'r', 'LineWidth', 1.5);
    elseif im == 2
        plot(t(idxTail), traj.(nm).y(idxTail), 'g ', 'LineWidth', 1.5);
    else
        plot(t(idxTail), traj.(nm).y(idxTail), 'c', 'LineWidth', 1.5);
    end
end
% legend(cell(['ref','ideal',methods]), 'Location','best');
% legend('ref','ideal','uniform','log','kmeans','proposed', 'Location','best');
legend('ref','uniform','logarithmic','kmeans','proposed', 'Location','best');
xlabel('k'); ylabel('y_k');
title('Tracking');

subplot(3,3,4)
histogram(train.x(:,1),80,'normalization','pdf')
xlim([-2.5,2.5])
xlabel('x_1')
ylabel('Histogram (pdf)')
% 
subplot(3,3,7)
plot_centroid_stems_LCSS(Qset, methods, 'x_1 centroids', 'x', 1);
xlim([-2.5,2.5])
xlabel('c_{x,1}')
ylim([0,1.05])
ylabel('Centroid spacing');

subplot(3,3,5)
histogram(train.x(:,2),80,'normalization','pdf')
xlim([-3.5,3.5])
xlabel('x_2')
%
subplot(3,3,8)
plot_centroid_stems_LCSS(Qset, methods, 'x_2 centroids', 'x', 2);
xlim([-3.5,3.5])
xlabel('c_{x,2}')
ylim([0,1.05])

subplot(3,3,6)
histogram(train.y,100,'normalization','pdf')
xlabel('y')
xlim([-1,1])
%
subplot(3,3,9)
plot_centroid_stems_LCSS(Qset, methods, 'y centroids', 'y', 1);
xlim([-1,1])
xlabel('c_{y,1}')
ylim([0,1.05])

%% ========================================================================
%% Local functions
%% ========================================================================
function testset = sample_initial_conditions(nTraj, x0_box, z0_box)
    testset = zeros(nTraj, 3);
    % TOL_LESS = 0.5;
    for i = 1:nTraj
        % x0 = (2*rand(2,1)-1).*x0_box(:)*(1-TOL_LESS);
        % z0 = (2*rand-1)*z0_box*(1-TOL_LESS);
        x0 = (2*rand(2,1)-1).*x0_box(:);
        z0 = (2*rand-1)*z0_box;
        testset(i,:) = [x0(:).', z0];
    end
end

function q = design_uniform_quantizer(S,N)
    disp('>> Designing uniform quantizer')
    b = linspace(-S, S, N+1);
    c = 0.5*(b(1:end-1) + b(2:end));
    q = make_quantizer_struct(b, c, sprintf('uniform N=%d',N));
end

function qbest = design_best_log_quantizer(s, w, S, N)
    % "Logarithmic" via mu-law companding -> uniform quantization in companded domain
    disp('>> Designing logarithmic quantizer')
    mu_grid = [0.5 1 2 5 10 20 50 100 200];
    bestJ = inf;
    qbest = design_uniform_quantizer(S,N); % fallback

    for mu = mu_grid
        q = design_logmulaw_quantizer(S, N, mu);
        J = weighted_sse(s, w, q);
        if J < bestJ
            bestJ = J;
            qbest = q;
        end
    end
end

function q = design_logmulaw_quantizer(S,N,mu)
    % Build finite-level static quantizer by companding thresholds/centroids
    bu = linspace(-1, 1, N+1);               % thresholds in companded domain
    cu = 0.5*(bu(1:end-1) + bu(2:end));      % centroids in companded domain

    b = S * inv_mulaw(bu, mu);
    c = S * inv_mulaw(cu, mu);

    % Enforce exact endpoints (numerical)
    b(1) = -S; b(end) = S;
    q = make_quantizer_struct(b, c, sprintf('log(mu=%.3g)',mu));
end

function y = inv_mulaw(v, mu)
    % inverse of normalized mu-law compressor
    y = sign(v) .* ((1+mu).^abs(v) - 1) / mu;
end

function q = design_lloyd1d_quantizer(s, w, S, N)
    % Weighted 1D Lloyd (K-means-like), local optimum.
    disp('>> Designing Lloyd-Max quantizer')
    s = s(:); w = w(:);
    M = numel(s);
    assert(M >= N, 'Need M >= N samples.');

    % Clamp to range (should already be inside)
    s = min(max(s, -S), S);

    % Multiple deterministic/random initializations
    init_list = cell(0,1);
    init_list{end+1} = linspace(min(s), max(s), N).';
    init_list{end+1} = weighted_quantile_init(s,w,N);
    for r = 1:3
        idx = randperm(M,N);
        init_list{end+1} = sort(s(idx));
    end

    bestJ = inf;
    bestc = [];

    for itry = 1:numel(init_list)
        c = sort(init_list{itry}(:));
        c = min(max(c,-S),S);

        num_iter = 100;  % initial
        % num_iter = 50;

        for iter = 1:num_iter
            if mod(iter,25) == 1
                disp(['-- iter=',num2str(iter)])
            end
            % Assign by nearest centroid
            edges = [-inf; 0.5*(c(1:end-1)+c(2:end)); inf];
            bin = discretize(s, edges);
            c_new = c;
            for k = 1:N
                Ik = (bin == k);
                if any(Ik)
                    c_new(k) = sum(w(Ik).*s(Ik)) / sum(w(Ik));
                else
                    % empty cluster: split largest-error point
                    [~, idxWorst] = max(w .* (s - c(bin)).^2);
                    c_new(k) = s(idxWorst);
                end
            end
            c_new = sort(min(max(c_new,-S),S));

            if norm(c_new - c, inf) < 1e-10
                c = c_new;
                break;
            end
            c = c_new;
        end

        % Build quantizer
        b = [-S; 0.5*(c(1:end-1)+c(2:end)); S];
        q_try = make_quantizer_struct(b, c, 'kmeans1d');
        J = weighted_sse(s, w, q_try);

        if J < bestJ
            bestJ = J;
            bestc = c;
        end
    end

    b = [-S; 0.5*(bestc(1:end-1)+bestc(2:end)); S];
    q = make_quantizer_struct(b, bestc, 'kmeans1d');
end

function c0 = weighted_quantile_init(s,w,N)
    [ss, idx] = sort(s(:));
    ww = w(idx);
    cw = cumsum(ww)/sum(ww);
    targets = ((1:N)-0.5)/N;
    c0 = zeros(N,1);
    for i = 1:N
        [~,j] = min(abs(cw - targets(i)));
        c0(i) = ss(j);
    end
    c0 = sort(c0);
end

function opt = design_surrogate_level_allocation( ...
    x1_samples, x2_samples, y_samples, w_x1, w_x2, w_y, ...
    Sx, Sy, Ntot, NminAlloc, designMaxSamples, label)
% Globally allocate the total level budget for the sampled separable surrogate.
% Phi_l(N) is the globally optimal weighted scalar DP cost for channel l.
    if nargin < 12 || isempty(label)
        label = 'sampled separable surrogate';
    end

    [x1_design, w_x1_design] = deterministic_subset(x1_samples, w_x1, designMaxSamples);
    [x2_design, w_x2_design] = deterministic_subset(x2_samples, w_x2, designMaxSamples);
    [y_design,  w_y_design]  = deterministic_subset(y_samples,  w_y,  designMaxSamples);

    nChannels = 3;
    NmaxAlloc = Ntot - (nChannels - 1) * NminAlloc;

    fprintf('\n=== Data-driven level allocation: %s ===\n', label);
    fprintf('Computing Phi_l(N) curves with %d samples/channel, N in [%d,%d]\n', ...
        numel(x1_design), NminAlloc, NmaxAlloc);

    curves = cell(1,nChannels);
    curves{1} = dp1d_cost_curve(x1_design, w_x1_design, Sx(1), NmaxAlloc);
    curves{2} = dp1d_cost_curve(x2_design, w_x2_design, Sx(2), NmaxAlloc);
    curves{3} = dp1d_cost_curve(y_design,  w_y_design,  Sy,    NmaxAlloc);

    [Nbest, bestCost] = optimize_level_allocation_dp(curves, Ntot, NminAlloc);

    fprintf('Global sampled-surrogate allocation: [%d %d %d], cost %.6e\n', ...
        Nbest(1), Nbest(2), Nbest(3), bestCost);

    opt = struct();
    opt.Nbest = Nbest;
    opt.bestCost = bestCost;
    opt.curves = curves;
    opt.Nmin = NminAlloc;
    opt.Nmax = NmaxAlloc;
    opt.designMaxSamples = designMaxSamples;
    opt.label = label;
end

function J = dp1d_cost_curve(samples, weights, S_s, Nmax)
% Cost curve J(N): globally optimal weighted 1D N-level DP SSE.
    fprintf('>> Computing DP cost curve up to N=%d\n', Nmax)

    s = samples(:);
    w = weights(:);

    msk = isfinite(s) & isfinite(w) & (w > 0) & (s >= -S_s) & (s <= S_s);
    s = s(msk);
    w = w(msk);
    if isempty(s)
        error('No valid samples for DP cost curve.');
    end

    [s, ord] = sort(s, 'ascend');
    w = w(ord);

    [su, ~, ic] = unique(s, 'stable');
    if numel(su) < numel(s)
        w = accumarray(ic, w);
        s = su;
    end

    M = numel(s);
    Nmax = min(Nmax, M);

    Wp  = [0; cumsum(w)];
    Sp  = [0; cumsum(w .* s)];
    S2p = [0; cumsum(w .* (s.^2))];
    cost = @(i,j) segment_cost(i,j,Wp,Sp,S2p);

    J = inf(Nmax,1);
    dp_prev = inf(M,1);
    for j = 1:M
        dp_prev(j) = cost(1,j);
    end
    J(1) = dp_prev(M);

    for k = 2:Nmax
        dp_cur = inf(M,1);
        optk = zeros(M,1,'uint32');
        [dp_cur, ~] = compute_row_dc(k, M, k-1, M-1, dp_prev, cost, dp_cur, optk);
        dp_prev = dp_cur;
        J(k) = dp_prev(M);
    end
end

function [Nbest, bestCost] = optimize_level_allocation_dp(curves, Ntot, NminAlloc)
% Dynamic program over channels and total budget.
    nChannels = numel(curves);
    D = inf(nChannels + 1, Ntot + 1);
    prevN = zeros(nChannels + 1, Ntot + 1);
    D(1,1) = 0; % zero channels, zero budget

    for ell = 1:nChannels
        maxN = numel(curves{ell});
        for b = 0:Ntot
            if ~isfinite(D(ell,b+1))
                continue;
            end
            remainingChannels = nChannels - ell;
            Nupper = min(maxN, Ntot - b - remainingChannels * NminAlloc);
            for N = NminAlloc:Nupper
                val = D(ell,b+1) + curves{ell}(N);
                if val < D(ell+1,b+N+1)
                    D(ell+1,b+N+1) = val;
                    prevN(ell+1,b+N+1) = N;
                end
            end
        end
    end

    bestCost = D(nChannels+1,Ntot+1);
    if ~isfinite(bestCost)
        error('No feasible level allocation found.');
    end

    Nbest = zeros(1,nChannels);
    b = Ntot;
    for ell = nChannels:-1:1
        Nbest(ell) = prevN(ell+1,b+1);
        b = b - Nbest(ell);
    end
end

function summary = allocation_perturb_observe( ...
    A, B, C, Kx, Kz, Ts, Qdelta, R, gamma, Wx_u, Wy_u, ...
    refCfg, Heval, Sx, Sy, Nprop, Ntot, ...
    x1_samples, x2_samples, y_samples, w_x1, w_x2, w_y, ...
    testset, refSeq, designMaxSamples, N_highrate, N_signal)
% Redesign DP quantizers under perturbed level allocations and evaluate both
% the full training surrogate and validation closed-loop costs.

    NminAlloc = 16;

    [x1_design, w_x1_design] = deterministic_subset(x1_samples, w_x1, designMaxSamples);
    [x2_design, w_x2_design] = deterministic_subset(x2_samples, w_x2, designMaxSamples);
    [y_design,  w_y_design]  = deterministic_subset(y_samples,  w_y,  designMaxSamples);

    allocRows = zeros(0,3);
    allocNames = {};

    allocRows(end+1,:) = Nprop;
    allocNames{end+1} = 'data-opt';

    if nargin >= 27 && ~isempty(N_highrate) && ~ismember(N_highrate, allocRows, 'rows')
        allocRows(end+1,:) = N_highrate;
        allocNames{end+1} = 'highrate';
    end

    if nargin >= 28 && ~isempty(N_signal) && ~ismember(N_signal, allocRows, 'rows')
        allocRows(end+1,:) = N_signal;
        allocNames{end+1} = 'signal-only';
    end

    Nequal = floor(Ntot/3) * ones(1,3);
    Nequal(1:(Ntot - sum(Nequal))) = Nequal(1:(Ntot - sum(Nequal))) + 1;
    if all(Nequal >= NminAlloc) && ~ismember(Nequal, allocRows, 'rows')
        allocRows(end+1,:) = Nequal;
        allocNames{end+1} = 'equal';
    end

    perturbDeltas = [20 40];
    perturbDirs = [
         1 -1  0;
        -1  1  0;
         1  0 -1;
        -1  0  1;
         0  1 -1;
         0 -1  1
    ];
    perturbNames = {'x1<-x2','x2<-x1','x1<-y','y<-x1','x2<-y','y<-x2'};

    for id = 1:size(perturbDirs,1)
        for dd = perturbDeltas
            Ntry = Nprop + dd * perturbDirs(id,:);
            if all(Ntry >= NminAlloc) && sum(Ntry) == Ntot && ...
                    ~ismember(Ntry, allocRows, 'rows')
                allocRows(end+1,:) = Ntry;
                allocNames{end+1} = sprintf('%s %d', perturbNames{id}, dd);
            end
        end
    end

    nAlloc = size(allocRows,1);
    allocTrain = zeros(nAlloc,1);
    allocJuMean = zeros(nAlloc,1);
    allocJuStd = zeros(nAlloc,1);
    allocJsMean = zeros(nAlloc,1);
    allocJsStd = zeros(nAlloc,1);
    allocOvr = zeros(nAlloc,1);

    fprintf('\n=== Allocation perturb-and-observe sensitivity ===\n');
    fprintf('DP redesign subset: %d samples/channel, validation trajectories: %d\n', ...
        numel(x1_design), size(testset,1));
    fprintf('Allocation             [Nx1 Nx2 Ny]   train surr.   rel train   avg Jgamma   avg Jsurr   overload\n');

    for ia = 1:nAlloc
        Ntry = allocRows(ia,:);
        fprintf('\n>> Allocation %d/%d: %s [%d %d %d]\n', ...
            ia, nAlloc, allocNames{ia}, Ntry(1), Ntry(2), Ntry(3));

        qalloc = struct();
        qalloc.x{1} = design_dp1d_quantizer(x1_design, w_x1_design, Sx(1), Ntry(1));
        qalloc.x{2} = design_dp1d_quantizer(x2_design, w_x2_design, Sx(2), Ntry(2));
        qalloc.y{1} = design_dp1d_quantizer(y_design,  w_y_design,  Sy,    Ntry(3));

        allocTrain(ia) = weighted_sse(x1_samples, w_x1, qalloc.x{1}) + ...
                         weighted_sse(x2_samples, w_x2, qalloc.x{2}) + ...
                         weighted_sse(y_samples,  w_y,  qalloc.y{1});

        outAlloc = evaluate_quantized_case_ref( ...
            A, B, C, Kx, Kz, Ts, Qdelta, R, gamma, ...
            Wx_u, Wy_u, qalloc, testset, Heval, refCfg, Sx, Sy, refSeq);

        allocJuMean(ia) = mean(outAlloc.Ju);
        allocJuStd(ia)  = std(outAlloc.Ju);
        allocJsMean(ia) = mean(outAlloc.Jsurr);
        allocJsStd(ia)  = std(outAlloc.Jsurr);
        allocOvr(ia)    = outAlloc.overload_count / max(1, outAlloc.total_quant_calls);
    end

    propIdx = find(all(allocRows == Nprop, 2), 1, 'first');
    propTrain = allocTrain(propIdx);

    fprintf('\nAllocation             [Nx1 Nx2 Ny]   train surr.   rel train   avg Jgamma   avg Jsurr   overload\n');
    for ia = 1:nAlloc
        fprintf('%-22s [%3d %3d %3d]   %.6e   %8.3f   %.6e   %.6e   %.4f\n', ...
            allocNames{ia}, allocRows(ia,1), allocRows(ia,2), allocRows(ia,3), ...
            allocTrain(ia), allocTrain(ia)/propTrain, ...
            allocJuMean(ia), allocJsMean(ia), allocOvr(ia));
    end

    fprintf('\n=== LaTeX allocation sensitivity table ===\n');
    fprintf('Allocation & $(N_{x,1},N_{x,2},N_y)$ & Rel. train surr. & $J_{\\gamma}$ & Surr. \\\\'); fprintf('\n');
    principal = {'data-opt','highrate','signal-only','equal'};
    pretty = {'Proposed','High-rate','Signal-only','Equal'};
    for ip = 1:numel(principal)
        idxp = find(strcmp(allocNames, principal{ip}), 1, 'first');
        if isempty(idxp)
            continue;
        end
        fprintf('%s & $(%d,%d,%d)$ & %.3f & %.3f & %.3f \\\\', ...
            pretty{ip}, allocRows(idxp,1), allocRows(idxp,2), allocRows(idxp,3), ...
            allocTrain(idxp)/propTrain, allocJuMean(idxp), allocJsMean(idxp));
        fprintf('\n');
    end

    figure('Name','Allocation perturb-and-observe sensitivity');
    tiledlayout(2,1);
    nexttile; bar(allocTrain / propTrain); grid on;
    ylabel('Train surrogate / proposed');
    set(gca,'XTick',1:nAlloc,'XTickLabel',allocNames,'XTickLabelRotation',30);
    title('Allocation sensitivity: training surrogate');

    nexttile; hold on; grid on;
    bar([allocJuMean / allocJuMean(propIdx), allocJsMean / allocJsMean(propIdx)]);
    ylabel('Validation cost / proposed');
    legend({'J_\gamma','Theorem surrogate'}, 'Location','best');
    set(gca,'XTick',1:nAlloc,'XTickLabel',allocNames,'XTickLabelRotation',30);
    title(sprintf('Validation costs on %d trajectories', size(testset,1)));

    summary = struct();
    summary.names = allocNames;
    summary.allocations = allocRows;
    summary.train = allocTrain;
    summary.JuMean = allocJuMean;
    summary.JuStd = allocJuStd;
    summary.JsMean = allocJsMean;
    summary.JsStd = allocJsStd;
    summary.overload = allocOvr;
    summary.propIdx = propIdx;
end

function [s_sub, w_sub] = deterministic_subset(s, w, maxSamples)
    s = s(:);
    w = w(:);
    n = numel(s);
    nKeep = min(n, maxSamples);
    if nKeep == n
        s_sub = s;
        w_sub = w;
        return;
    end
    idx = unique(round(linspace(1, n, nKeep))).';
    s_sub = s(idx);
    w_sub = w(idx);
end

function q = design_dp1d_quantizer(samples, weights, S_s, N)
% Globally optimal weighted 1D N-level scalar quantizer (memory-safe DP)
% samples : Mx1
% weights : Mx1, positive
% range   : [-S_s, S_s]
% N       : number of levels (1 <= N <= M)
    disp('>> Designing DP quantizer')

    s = samples(:);
    w = weights(:);

    % Keep only valid in-range, positive-weight samples
    msk = isfinite(s) & isfinite(w) & (w > 0) & (s >= -S_s) & (s <= S_s);
    s = s(msk);
    w = w(msk);

    if isempty(s)
        error('No valid samples for DP quantizer design.');
    end

    % Sort by sample value
    [s, ord] = sort(s, 'ascend');
    w = w(ord);

    M = numel(s);
    N = min(max(1, round(N)), M);

    % Optional exact compression of duplicates (helps a lot if duplicates exist)
    [su, ~, ic] = unique(s, 'stable');
    if numel(su) < M
        wu = accumarray(ic, w);
        s = su;
        w = wu;
        M = numel(s);
        N = min(N, M);
    end

    % Prefix sums: Wp(t+1)=sum_{i=1}^t w_i
    Wp  = [0; cumsum(w)];
    Sp  = [0; cumsum(w .* s)];
    S2p = [0; cumsum(w .* (s.^2))];

    % O(1) weighted segment SSE cost for inclusive segment [i..j], 1<=i<=j<=M
    cost = @(i,j) segment_cost(i,j,Wp,Sp,S2p);

    % DP arrays:
    % dp_prev(j) = optimal cost using (k-1) segments for first j samples
    % dp_cur(j)  = optimal cost using k segments for first j samples
    dp_prev = inf(M,1);
    splitIdx = zeros(N, M, 'uint32');  % splitIdx(k,j)=best i for recurrence at row k, col j

    % Base row k=1
    for j = 1:M
        dp_prev(j) = cost(1,j);
    end

    % Subsequent rows k=2..N using divide-and-conquer optimization
    for k = 2:N
        dp_cur = inf(M,1);
        optk   = zeros(M,1,'uint32');

        % valid j are j >= k
        [dp_cur, optk] = compute_row_dc(k, M, k-1, M-1, dp_prev, cost, dp_cur, optk);

        splitIdx(k,:) = optk;
        dp_prev = dp_cur;
    end

    % Backtrack segment boundaries [a_r .. b_r], r=1..N
    a = zeros(N,1);
    b = zeros(N,1);

    j = M;
    for k = N:-1:2
        i = double(splitIdx(k,j)); % previous endpoint
        a(k) = i + 1;
        b(k) = j;
        j = i;
    end
    a(1) = 1;
    b(1) = j;

    % Centroids (weighted means)
    c = zeros(N,1);
    for r = 1:N
        ii = a(r); jj = b(r);
        Wseg = Wp(jj+1) - Wp(ii);
        Sseg = Sp(jj+1) - Sp(ii);
        c(r) = Sseg / Wseg;
    end

    % Thresholds: prescribed endpoints + midpoints between adjacent centroids
    bth = zeros(N+1,1);
    bth(1)   = -S_s;
    bth(end) =  S_s;
    if N >= 2
        bth(2:N) = 0.5 * (c(1:end-1) + c(2:end));
    end

    % Return struct (adapt fields to your codebase)
    q = struct();
    q.N = N;
    q.S = S_s;
    q.c = c(:).';
    q.b = bth(:).';
    q.dp_cost = dp_prev(M);
end

% ---------- helpers ----------

function val = segment_cost(i,j,Wp,Sp,S2p)
% Weighted SSE of fitting samples i..j by one centroid (weighted mean)
    Wseg  = Wp(j+1)  - Wp(i);
    Sseg  = Sp(j+1)  - Sp(i);
    S2seg = S2p(j+1) - S2p(i);

    % Numerical guard (Wseg should be >0 because weights >0)
    if Wseg <= 0
        val = inf;
    else
        val = S2seg - (Sseg*Sseg)/Wseg;
        if val < 0 && val > -1e-12
            val = 0; % numerical cleanup
        end
    end
end

function [dp_cur, optk] = compute_row_dc(jL, jR, optL, optR, dp_prev, cost, dp_cur, optk)
% Divide-and-conquer optimization for one DP row
% Recurrence at fixed row k:
% dp_cur(j) = min_{i in [optL..optR] and i<=j-1} dp_prev(i) + cost(i+1, j)

    if jL > jR
        return;
    end

    jMid = floor((jL + jR) / 2);

    iMin = optL;
    iMax = min(optR, jMid-1);

    bestVal = inf;
    bestI   = uint32(iMin);

    for i = iMin:iMax
        v = dp_prev(i) + cost(i+1, jMid);
        if v < bestVal
            bestVal = v;
            bestI = uint32(i);
        end
    end

    dp_cur(jMid) = bestVal;
    optk(jMid)   = bestI;

    % Recurse left/right with monotone-optimality bounds
    [dp_cur, optk] = compute_row_dc(jL, jMid-1, optL, double(bestI), dp_prev, cost, dp_cur, optk);
    [dp_cur, optk] = compute_row_dc(jMid+1, jR, double(bestI), optR, dp_prev, cost, dp_cur, optk);
end

function q = make_quantizer_struct(b, c, name)
    b = b(:); c = c(:);
    assert(numel(b) == numel(c)+1, 'Threshold/centroid size mismatch.');
    assert(all(diff(b) >= 0), 'Thresholds must be sorted.');
    q.b = b;
    q.c = c;
    q.N = numel(c);
    q.name = name;
end

function J = weighted_sse(s, w, q)
    sq = quantize_vec(s, q);
    err = s(:) - sq(:);
    J = sum(w(:) .* (err.^2));
end

function emax = max_cell_error(q)
    emax = 0.5 * max(diff(q.b(:)));
end

function name = method_pretty_name(nm)
    switch lower(nm)
        case 'uniform'
            name = 'Uniform';
        case 'log'
            name = 'Logarithmic';
        case 'kmeans'
            name = 'Lloyd-Max';
        case 'dp'
            name = 'Proposed DP';
        otherwise
            name = nm;
    end
end

function qv = quantize_vec(v, q)
    % Saturating scalar quantizer, vectorized
    v = v(:);
    qv = zeros(size(v));

    % Saturate outside range
    left  = (v <= q.b(1));
    right = (v >= q.b(end));
    mid   = ~(left | right);

    qv(left)  = q.c(1);
    qv(right) = q.c(end);

    if any(mid)
        vm = v(mid);
        % discretize uses edges, bins 1..N on [b_i, b_{i+1})
        bin = discretize(vm, q.b);
        % Handle exact right endpoint mapping
        bin(isnan(bin)) = q.N;
        qv(mid) = q.c(bin);
    end
end

function [qv, overload] = quantize_scalar(v, q)
    overload = (v < q.b(1)) || (v > q.b(end));
    if v <= q.b(1)
        qv = q.c(1); return;
    elseif v >= q.b(end)
        qv = q.c(end); return;
    else
        bin = discretize(v, q.b);
        if isnan(bin), bin = q.N; end
        qv = q.c(bin);
    end
end

function r = reference_at_k(k, refCfg)
    switch lower(refCfg.type)

        case 'const'
            r = refCfg.r0;

        case 'sine'
            % r_k = bias + amp*sin(2*pi*freq*k + phase)
            bias  = getfield_with_default(refCfg, 'bias',  0);
            phase = getfield_with_default(refCfg, 'phase', 0);
            r = bias + refCfg.amp * sin(2*pi*refCfg.freq*k + phase);

        case 'piecewise'
            % piecewise-constant: values(idx)
            idx = find(refCfg.switch_k <= k, 1, 'last');
            if isempty(idx), idx = 1; end
            r = refCfg.values(idx);

        case 'mixed'
            % Segment start indices (must be sorted, typically start at 0)
            sw = refCfg.switch_k(:).';
            idx = find(sw <= k, 1, 'last');
            if isempty(idx), idx = 1; end
            tau = k - sw(idx);  % local time within segment (in samples)

            % Two supported ways to specify segments:
            %   (A) segType{idx} + segPar{idx}
            %   (B) segments(idx) struct with field "type" (and params)
            if isfield(refCfg, 'segments')
                seg = refCfg.segments(idx);
                segType = lower(seg.type);
                par = seg;
            else
                segType = lower(refCfg.segType{idx});
                par = refCfg.segPar{idx};
            end

            switch segType
                case 'step'
                    % r = value
                    r = par.value;

                case 'ramp'
                    % r = start + slope * tau
                    % optional clamp via par.min/par.max
                    r = par.start + par.slope * tau;
                    if isfield(par,'min'), r = max(r, par.min); end
                    if isfield(par,'max'), r = min(r, par.max); end

                case 'sine'
                    % r = bias + amp*sin(2*pi*freq*tau + phase)
                    bias  = getfield_with_default(par, 'bias',  0);
                    phase = getfield_with_default(par, 'phase', 0);
                    r = bias + par.amp * sin(2*pi*par.freq*tau + phase);

                otherwise
                    error('Unknown mixed segment type = %s', segType);
            end

        otherwise
            error('Unknown refCfg.type = %s', refCfg.type);
    end
end

function r = regularized_reference_at_k(k, refCfg)
    r = reference_at_k(k, refCfg);
    pReg = getfield_with_default(refCfg, 'trainPerturbProb', 0);
    frac = getfield_with_default(refCfg, 'trainPerturbFrac', 0);
    if frac > 0 && rand(1) < pReg
        r = r .* (1 + frac * (2*rand(size(r)) - 1));
    end
end

function Rseq = sample_reference_sequences(nTraj, H, refCfg, useRegularized)
    if nargin < 4
        useRegularized = false;
    end
    Rseq = zeros(nTraj, H+1);
    for tr = 1:nTraj
        for k = 0:H
            if useRegularized
                Rseq(tr,k+1) = regularized_reference_at_k(k, refCfg);
            else
                Rseq(tr,k+1) = reference_at_k(k, refCfg);
            end
        end
    end
end

function rbar = reference_bound_over_horizon(refCfg, H)
    rmax = 0;
    for k = 0:H
        rmax = max(rmax, max(abs(reference_at_k(k, refCfg))));
    end
    frac = getfield_with_default(refCfg, 'trainPerturbFrac', 0);
    rbar = (1 + frac) * rmax;
end

function [X, Y, cert] = no_saturation_range_certificate( ...
    Acl, E, Gx, Gy, Hx, Hy, xi0bar, rbar, ebar_x, ebar_y)
% Component-wise implementation of the no-saturation certificate.
    tol = 1e-11;
    Kmax = 20000;
    nAug = size(Acl,1);

    H = {Hx, Hy};
    beta = cell(1,2);
    GammaR = cell(1,2);
    GammaX = cell(1,2);
    GammaY = cell(1,2);
    for ih = 1:2
        beta{ih} = zeros(size(H{ih},1),1);
        GammaR{ih} = zeros(size(H{ih},1),size(E,2));
        GammaX{ih} = zeros(size(H{ih},1),size(Gx,2));
        GammaY{ih} = zeros(size(H{ih},1),size(Gy,2));
    end

    Ak = eye(nAug);
    tailNorm = inf;
    kStop = Kmax;
    for k = 0:Kmax
        for ih = 1:2
            HAk = H{ih} * Ak;
            beta{ih} = max(beta{ih}, abs(HAk) * xi0bar(:));
            GammaR{ih} = GammaR{ih} + abs(HAk * E);
            GammaX{ih} = GammaX{ih} + abs(HAk * Gx);
            GammaY{ih} = GammaY{ih} + abs(HAk * Gy);
        end

        Ak = Ak * Acl;
        tailNorm = norm(Ak, inf);
        if tailNorm < tol
            kStop = k;
            break;
        end
    end

    X = beta{1} + GammaR{1} * rbar(:) + GammaX{1} * ebar_x(:) + GammaY{1} * ebar_y(:);
    Y = beta{2} + GammaR{2} * rbar(:) + GammaX{2} * ebar_x(:) + GammaY{2} * ebar_y(:);

    cert = struct();
    cert.beta_x = beta{1};
    cert.beta_y = beta{2};
    cert.Gamma_x_r = GammaR{1};
    cert.Gamma_y_r = GammaR{2};
    cert.Gamma_x_x = GammaX{1};
    cert.Gamma_y_x = GammaX{2};
    cert.Gamma_x_y = GammaY{1};
    cert.Gamma_y_y = GammaY{2};
    cert.kStop = kStop;
    cert.tailNorm = tailNorm;
end

function v = getfield_with_default(s, fname, vdefault)
    if isfield(s, fname), v = s.(fname); else, v = vdefault; end
end

function data = collect_ideal_dataset_ref(A,B,C,Kx,Kz,Ts,nTraj,H,x0_box,z0_box,gamma,refCfg)
    n = size(A,1);
    X = [];
    Y = [];
    W = [];
    R = [];

    for tr = 1:nTraj
        if mod(tr,25) == 1
            disp(['>> Traj ',num2str(tr)]);
        end
        x0 = (2*rand(n,1)-1).*x0_box(:);
        z0 = (2*rand-1)*z0_box;
        x = x0; z = z0;

        for k = 0:H
            y = C*x;
            r = regularized_reference_at_k(k, refCfg);

            R = [R; r];
            X = [X; x.'];
            Y = [Y; y];
            W = [W; gamma^k];

            u = Kx*x + Kz*z;              % ideal controller state for this reference
            
            x_next = A*x + B*u;
            z_next = z + Ts*(r - y);      % ideal integrator uses unquantized y

            x = x_next; z = z_next;
        end
    end

    data.r = R;
    data.x = X;
    data.y = Y;
    data.w = W;

    % debug trajectories
    figure,
    % subplot(311),stem(data.x(:,1)),hold on,stem(data.x(:,2));
    % subplot(312),stem(data.y);
    % subplot(313),stem(data.w)
    subplot(411),plot(data.x(:,1)),hold on,plot(data.x(:,2));
    subplot(412),plot(data.y);
    subplot(413),plot(data.w)
    subplot(414),plot(data.r)
end

function out = evaluate_quantized_case_ref(A,B,C,Kx,Kz,Ts,Qdelta,R,gamma,Wx_u,Wy_u,qbundle,testset,Heval,refCfg,Sx,Sy,refSeq)
    if nargin < 18
        refSeq = [];
    end
    nTraj = size(testset,1);
    Ju = zeros(nTraj,1);
    Js = zeros(nTraj,1);

    overload_count = 0;
    total_quant_calls = 0;
    max_abs_ex = zeros(2,1);
    max_abs_ey = 0;
    max_abs_x = zeros(2,1);
    max_abs_y = 0;

    for tr = 1:nTraj
        xi_q   = testset(tr,:).';  % [x1;x2;z]
        xi_ref = xi_q;             % ideal starts from same initial condition

        J1 = 0;
        J2 = 0;

        for k = 0:Heval
            if isempty(refSeq)
                r = reference_at_k(k, refCfg);
            else
                r = refSeq(tr,k+1);
            end

            % Quantized system state
            x = xi_q(1:2);
            z = xi_q(3);
            y = C*x;

            % Ideal system state
            xs = xi_ref(1:2);
            zs = xi_ref(3);
            ys = C*xs;

            % Quantize measurements for implemented controller
            [xq1, ov1] = quantize_scalar(x(1), qbundle.x{1});
            [xq2, ov2] = quantize_scalar(x(2), qbundle.x{2});
            [yq,  ov3] = quantize_scalar(y,    qbundle.y{1});
            overload_count = overload_count + ov1 + ov2 + ov3;
            total_quant_calls = total_quant_calls + 3;

            xq = [xq1; xq2];

            % Commands
            u  = Kx*xq + Kz*z;
            us = Kx*xs + Kz*zs;   % ideal command under same reference (through zs)

            % dxi = [xs;zs]-[xq;z];
            dxi = [x; z] - [xs; zs];   % actual augmented state minus ideal augmented state
                                       % (or reversed sign; squared cost doesn't care)
            du = u - us;
            ex = xq - x;
            ey = yq - y;
            max_abs_ex = max(max_abs_ex, abs(ex));
            max_abs_ey = max(max_abs_ey, abs(ey));
            max_abs_x = max(max_abs_x, abs(x));
            max_abs_y = max(max_abs_y, abs(y));

            % old; u only
            % J1 = J1 + (gamma^k) * (du' * R * du);
            % J2 = J2 + (gamma^k) * (ex' * Wx_u * ex + ey' * Wy_u * ey);
            J1 = J1 + (gamma^k) * (dxi' * Qdelta * dxi + du' * R * du);
            J2 = J2 + (gamma^k) * (ex' * Wx_u * ex + ey' * Wy_u * ey);

            % Step both systems with SAME reference
            x_next  = A*x  + B*u;
            z_next  = z  + Ts*(r - yq);  % quantized output in integrator

            xs_next = A*xs + B*us;
            zs_next = zs + Ts*(r - ys);  % ideal output in integrator

            xi_q   = [x_next;  z_next];
            xi_ref = [xs_next; zs_next];
        
            try
                assert(abs(x_next(1)) <= Sx(1)+1e-12);
                assert(abs(x_next(2)) <= Sx(2)+1e-12);
                assert(abs(y)  <= Sy +1e-12);
            catch me
                disp(me)
            end

        end

        Ju(tr) = J1;
        Js(tr) = J2;
    end

    out.Ju = Ju;
    out.Jsurr = Js;
    out.overload_count = overload_count;
    out.total_quant_calls = total_quant_calls;
    out.max_abs_ex = max_abs_ex;
    out.max_abs_ey = max_abs_ey;
    out.max_abs_x = max_abs_x;
    out.max_abs_y = max_abs_y;
end

function traj = simulate_single_trajectory_ref(A,B,C,Kx,Kz,Ts,R,gamma,Wx_u,Wy_u,qbundle,x0_aug,Heval,refCfg)
    % gamma, R, Wx_u, Wy_u kept for possible extensions
    xi_q   = x0_aug(:);
    xi_ref = x0_aug(:);

    K = 0:Heval;
    Nk = numel(K);

    traj.k = K(:);
    traj.r = zeros(Nk,1);

    traj.x1 = zeros(Nk,1); traj.x2 = zeros(Nk,1); traj.z = zeros(Nk,1);
    traj.y  = zeros(Nk,1); traj.u  = zeros(Nk,1);

    traj.x1_ideal = zeros(Nk,1); traj.x2_ideal = zeros(Nk,1); traj.z_ideal = zeros(Nk,1);
    traj.y_ideal  = zeros(Nk,1); traj.u_ideal  = zeros(Nk,1);

    for ii = 1:Nk
        k = K(ii);
        r = reference_at_k(k, refCfg);

        x = xi_q(1:2); z = xi_q(3); y = C*x;
        xs = xi_ref(1:2); zs = xi_ref(3); ys = C*xs;

        [xq1, ~] = quantize_scalar(x(1), qbundle.x{1});
        [xq2, ~] = quantize_scalar(x(2), qbundle.x{2});
        [yq,  ~] = quantize_scalar(y,    qbundle.y{1});

        xq = [xq1; xq2];

        u  = Kx*xq + Kz*z;
        us = Kx*xs + Kz*zs;

        traj.r(ii) = r;

        traj.x1(ii) = x(1); traj.x2(ii) = x(2); traj.z(ii) = z;
        traj.y(ii)  = y;    traj.u(ii)  = u;

        traj.x1_ideal(ii) = xs(1); traj.x2_ideal(ii) = xs(2); traj.z_ideal(ii) = zs;
        traj.y_ideal(ii)  = ys;    traj.u_ideal(ii)  = us;

        % step
        xi_q   = [A*x + B*u;     z  + Ts*(r - yq)];
        xi_ref = [A*xs + B*us;   zs + Ts*(r - ys)];
    end
end

function plot_centroid_stems(Qset, methods, figTitle, xyTag, idxChan)
    figure('Name', figTitle); hold on; grid on;

    yLevels = 1:numel(methods);
    for im = 1:numel(methods)
        nm = methods{im};
        q = Qset.(nm).(xyTag){idxChan};
        c = q.c(:).';
        stem(c, yLevels(im)*ones(size(c)), 'filled', 'LineWidth', 1.0);
        % stem([0,diff(c)], yLevels(im)*ones(size(c)), 'filled', 'LineWidth', 1.0);  % see where it's dense
        % stem(c, yLevels(im)*[0,diff(c)], 'filled', 'LineWidth', 1.0);  % centroid density
    end
    legend(methods, 'Location','best');
    set(gca, 'YTick', yLevels, 'YTickLabel', methods);
    xlabel('centroid location');
    ylabel('quantizer type');
    title(['Centroid spacing: ', figTitle]);
end

% function plot_centroid_stems_LCSS(Qset, methods, figTitle, xyTag, idxChan)
%     hold on; grid on;
% 
%     yLevels = 1:numel(methods);
%     for im = 4:numel(methods)
%         nm = methods{im};
%         q = Qset.(nm).(xyTag){idxChan};
%         c = q.c(:).';
%         % stem(c, ones(size(c)), 'filled', 'LineWidth', 0.5);
%         stem(c, ones(size(c)), 'filled', 'LineWidth', 1);
%         % stem([0,diff(c)], yLevels(im)*ones(size(c)), 'filled', 'LineWidth', 1.0);  % see where it's dense
%         % stem(c, yLevels(im)*[0,diff(c)], 'filled', 'LineWidth', 1.0);  % centroid density
%     end
%     % legend(methods, 'Location','best');
%     % set(gca, 'YTick', yLevels, 'YTickLabel', methods);
%     % xlabel('centroid location');
%     % ylabel('Centroid spacing');
%     % title(['Centroid spacing: ', figTitle]);
% end

function plot_centroid_stems_LCSS(Qset, methods, figTitle, xyTag, idxChan)
    hold on; grid on;

    nm = 'dp';
    q = Qset.(nm).(xyTag){idxChan};
    c = q.c(:).';

    maxStems = 200;
    stride = max(1, ceil(numel(c)/maxStems));
    idx = 1:stride:numel(c);

    stem(c(idx), ones(size(idx)), 'filled', ...
        'LineWidth', 0.8, 'MarkerSize', 3);
end

% function plot_centroid_stems_diff(Qset, methods, figTitle, xyTag, idxChan)
%     figure('Name', figTitle); hold on; grid on;
% 
%     yLevels = 1:numel(methods);
%     for im = 1:numel(methods)
%         nm = methods{im};
%         q = Qset.(nm).(xyTag){idxChan};
%         c = q.c(:).';
%         % stem(c, yLevels(im)*ones(size(c)), 'filled', 'LineWidth', 1.0);
%         % stem([0,diff(c)], yLevels(im)*ones(size(c)), 'filled', 'LineWidth', 1.0);  % see where it's dense
%         stem(c, yLevels(im)*[0,diff(c)], 'filled', 'LineWidth', 1.0);  % centroid density
%     end
%     legend(methods, 'Location','best');
%     set(gca, 'YTick', yLevels, 'YTickLabel', methods);
%     xlabel('centroid location');
%     ylabel('quantizer type');
%     title(['Centroid spacing (diff/density): ', figTitle]);
% end

function plot_rect_partition(qx1, qx2)
    % Component-wise scalar quantizers => rectangular partition in (x1,x2)
    % xlim([qx1.b(1), qx1.b(end)]);
    % ylim([qx2.b(1), qx2.b(end)]);

    % Vertical lines from x1 thresholds
    for i = 1:numel(qx1.b)
        xline(qx1.b(i), '-', 'LineWidth', 0.8);
    end

    % Horizontal lines from x2 thresholds
    for j = 1:numel(qx2.b)
        yline(qx2.b(j), '-', 'LineWidth', 0.8);
    end

    % Mark reconstruction points (Cartesian product of centroids)
    [C1, C2] = meshgrid(qx1.c, qx2.c);
    plot(C1(:), C2(:), 'k.', 'MarkerSize', 10);
end
