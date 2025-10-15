function V = V_matrix(X, gamma)
%% Explain:
    % V-matrix emprical version
    % ---- Input ----
    % X  -  m x n matrix
    % gamma -  1/(2*sig^2)
    % ---- Output ----
    % V  -  m x m matrix
    % Written by Liu tian, Latest updata: 2025-02-27. 
%% Code:
    m = size(X, 1);
    X_sq = sum(X.^2, 2);  
    dist_sq = bsxfun(@plus, X_sq, X_sq') - 2 * (X * X');  
    G = exp(-gamma.* dist_sq );
    V = (1 / m) * (G * G');
end






