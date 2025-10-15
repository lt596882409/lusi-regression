function [PredY] = VR(TestX, Trn, Params)
%% Explain:
    % ---- Input ----
    % Trn.X  -  m x n matrix, explanatory variables in training data 
    % Trn.Y  -  m x 1 vector, response variables in training data 
    % TestX  -  mt x 1 matrix, test datasets without labels  
    % Para.p1  -   the regularization constant gam
    % Para.kpar  -  kernel para, include type and para value of kernel
    % Para.V  -  m x m matrix, V matrix
    
    % ---- Output ----
    % PredY  -  mt x 1 vector, predicted response variables for TestX 
    % Written by Liu tian, Latest updata: 2025-02-27. 
%% Code:
    % ---- Initiation ----
    X = Trn.X;
    Y = Trn.Y;
    clear Trn 
    V = Params.V;
    gam = Params.C;
    kpar.kp1 = Params.Sigma_K;
    kpar.ktype = Params.Kertype;
    % ---- Solve ----
    E = eye(length(Y));
    Yao = ones(length(Y),1); 
    KerX = KerF(X, kpar, X);
    temp_VK = V*KerX;
    temp_inv = (temp_VK+gam*E)\V;
    temp_c = Yao'*(temp_VK*temp_inv - V);
    Av = temp_inv*Y;
    Ac = temp_inv*Yao;
    c = (temp_c*Y) / (temp_c*Yao); 
    A = Av-c*Ac; 
    [KerTstX] = KerF(TestX, kpar, X);
    % ---- Output ----
    PredY = KerTstX*A + c;
end