% RUSI_I Demo
clear; clc; close all;
Mod = "RUSI_I"; delta = 10^-5;
k = 3; seed = 0; rng(seed);
test_radio = 0.3; radio = 0.5;
result_log = struct();
C_range = 2.^(-8:2:8);
K_range = 2.^(-8:2:8);
tau_range = 0.1:0.2:0.9;
% ---- Model Settings ----
model_config = struct(...
    'LSSVR', struct('fun', @VR, 'params', {'C', 'sigma_K'}, 'name', 'LSSVR'), ...
    'RUSI_I', struct('fun', @RUSI_M, 'params', {'C', 'sigma_K', 'sigma_P', 'tau'}, 'name', 'RUSI_I') ...
    );
load 'AutoMobile_Price.mat';
X = zscore(X);
cv = cvpartition(size(X, 1), 'HoldOut', test_radio);
resIdx = training(cv); testIdx = test(cv);
Test.X = X(testIdx, :); Test.Y = Y(testIdx);
Res.X = X(resIdx, :); Res.Y = Y(resIdx);
cv = cvpartition(size(Res.X, 1), 'HoldOut', radio);
trainIdx = training(cv);
Train.X = Res.X(trainIdx, :); Train.Y = Res.Y(trainIdx);
l = length(Train.Y);
E = eye(l);
best_score = inf;
[K, C] = meshgrid(K_range, C_range);
param_grid = [K(:), C(:)];
scores = nan(size(param_grid,1), 1);
for j = 1:size(param_grid, 1)
    current_params = struct('Sigma_K', param_grid(j,1), 'C', param_grid(j,2));
    current_params.V = E;
    current_params.Kertype = 'rbf';
    scores(j) = cross_validate(model_config.("LSSVR"), Train, current_params, k);
    if scores(j) < best_score
        best_score = scores(j);
        best_params = current_params;
    end
end
best_params.P = zeros(length(Train.X));
best_params.Tau = 0;
ModFun = model_config.(Mod).fun;
T_Y = ModFun(Train.X, Train, best_params);
% T0 ------------------------
best_set = [];
Ptype = ["one","linear","matrix","gaussian"];
for p = 1:length(Ptype)
    type = Ptype(p);
    P = P_matrix(Train.X, best_params.Sigma_K, 0.01, 1, type);
    diff = T_Y - Train.Y;
    B = (diff' * P * diff);
    T = B / (Train.Y'* P* Train.Y);
end
flag = true;
while flag
    Tmax = 0;
    if ~isempty(best_set)
        mask = ~ismember(Ptype,best_set);
        Ptype = Ptype(mask);
    end
    best_type = "none";
    for p = 1:length(Ptype)
        type = Ptype(p);
        P = P_matrix(Train.X, best_params.Sigma_K, 0.01, 1, type);
        diff = T_Y - Train.Y;  % m×1
        B = (diff' * P * diff);
        T = B / (Train.Y'* P* Train.Y);
        if T >= delta && T>=Tmax
            Tmax = T;
            best_type = type;
            best_P = P;
        end
    end
    for p = 1:length(best_set)
        type = best_set(p);
        P = P_matrix(Train.X, best_params.Sigma_K, 0.01, 1, type);
        diff = T_Y - Train.Y;  % m×1
        B = (diff' * P * diff);
        T = B / (Train.Y'* P* Train.Y);
    end
    if sum(best_type == "none")
        flag = false;
        break;
    end
    best_set = [best_set; best_type];
    best_score = inf;
    sigma_k = best_params.Sigma_K;
    c = best_params.C;
    if c>= 2^-6 && c<=2^6
        C_range_si = [c*(2^-2),c,c*(2^2)];
    elseif c<2^-6
        C_range_si = [c,c*(2^2),c*(2^4)];
    elseif c>2^6
        C_range_si = [c*(2^-4),c*(2^-2),c];
    end
    if sigma_k>= 2^-6 && sigma_k<=2^6
        K_range_si = [sigma_k*(2^-2),sigma_k,sigma_k*(2^2)];
    elseif sigma_k<2^-6
        K_range_si = [sigma_k,sigma_k*(2^2),sigma_k*(2^4)];
    elseif sigma_k>2^6
        K_range_si = [sigma_k*(2^-4),sigma_k*(2^-2),sigma_k];
    end
    [K, C, Tau] = ndgrid(K_range_si, C_range_si, tau_range);
    param_grid = [K(:), C(:), Tau(:)];
    scores = nan(size(param_grid,1), 1);
    for j = 1:size(param_grid, 1)
        current_params = struct('Sigma_K', param_grid(j,1), 'C', param_grid(j,2), 'Tau', param_grid(j,3));
        current_params.V = E;
        temp_P = zeros(length(Train.X));
        for num = 1:length(best_set)
            type = best_set(num);
            temp_P = temp_P + P_matrix(Train.X, current_params.Sigma_K, 0.01, 1, type);
        end
        current_params.P = temp_P/length(best_set);
        current_params.Kertype = 'rbf';
        % K-fold Cross Validation
        scores(j) = cross_validate(model_config.(Mod), Train, current_params, k);
        if scores(j) < best_score
            best_score = scores(j);
            best_params = current_params;
        end
    end
    T_Y = ModFun(Train.X, Train, best_params);

    % Tn ------------------------
    Col = ["one","linear","matrix","gaussian"];
    for p = 1:length(Col)
        type = Col(p);
        P = P_matrix(Train.X, best_params.Sigma_K, 0.01, 1, type);
        diff = T_Y - Train.Y;  % m×1
        if ismember(type, best_set)
            B = (best_params.Tau*(diff' * P * diff))/length(best_set);
        else
            B = (diff' * P * diff);
        end
        T = B / (Train.Y'* P* Train.Y);
    end
    Pred_Y = ModFun(Test.X, Train, best_params);
    metric = calculate_metrics(Test.Y, Pred_Y);
end
% ---- Train & Test ---
ModFun = model_config.(Mod).fun;
tic;
[Pred_Y] = ModFun(Test.X, Train, best_params);
time = toc;
[metrics] = calculate_metrics(Test.Y, Pred_Y);
disp(metrics);
