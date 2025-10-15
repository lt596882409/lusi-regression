function [PredY, model] = RUSI_M(TestX, Trn, Params)
%% Explain:
    % M - V or I
    % -----Input-----
    % Trn.X  -  m x n matrix, explanatory variables in training data 
    % Trn.Y  -  m x 1 vector, response variables in training data 
    % TestX  -  mt x 1 matrix, test datasets without labels  
    % Para.p1  -  the regularization constant gam 
    % Para.kpar  -  kernel para, include type and para value of kernel
    % Para.p3  -  the equilibrium parameter tao, a trade-off between P and V matrices
    
    % -----Output-----
    % PredY  -  mt x 1 vector, predicted response variables for TestX 
    % Written by Liu tian, Latest updata: 2025-02-27. 
%% Code:
    % ---- Initiation ----
    X = Trn.X;
    Y = Trn.Y;
    clear Trn 
    P = Params.P;
    V = Params.V;
    gam = Params.C;
    tau = Params.Tau;
    tau_hat = 1-tau;
    kpar.kp1 = Params.Sigma_K;
    kpar.ktype = Params.Kertype;
    % ---- Solve ----
    m = length(Y);
    H = tau_hat*V + tau*P;
    KerX = KerF(X, kpar, X);
    H_KerX = H * KerX;
    E = eye(m); 
    H_gamma = H_KerX + gam*E;
    temp_inv = H_gamma\H;
    sum_temp_inv = sum(temp_inv, 1); 
    temp_c = -gam * sum_temp_inv;
    numerator = temp_c * Y;
    denominator = temp_c * ones(m, 1);
    c = numerator / denominator;
    A = temp_inv * (Y - c*ones(m,1));
    KerTstX = KerF(TestX, kpar, X);  
    % ---- Output ----
    PredY = KerTstX*A + c; 
    model.alpha = A;
    model.w = A'*X;
    model.b = c;
    model.P = P;
end
