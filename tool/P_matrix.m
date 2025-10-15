function P = P_matrix(X_train, gamma, alpha, c, type)
%% Explain:
    % P-matrix emprical version
    % ---- Input ----
    % X_train  -  m x n matrix
    % gamma -  1/(2*sig^2)
    % type - Predicate type
    % ---- Output ----
    % P  -  m x m matrix
    % Written by Liu tian, Latest updata: 2025-02-27. 
%% Code:
    switch type
        case 'matrix'
            [m, n]=size(X_train);
            Phi = zeros(m, n*(n+1)/2);
            idx = 1;
            for i = 1 : n
                for j = i : n
                    Phi(:, idx) = X_train(:, i) .* X_train(:, j);
                    idx = idx + 1;
                end
            end
             P = (Phi * Phi');
        case 'one'
            P = ones(length(X_train),length(X_train));
        case 'gaussian'
            X_sq = sum(X_train.^2, 2); 
            dist_sq = bsxfun(@plus, X_sq, X_sq') - 2 * (X_train * X_train'); % 欧几里得距离的平方
            P = exp(-gamma.* dist_sq); 
        case 'linear'
            P = X_train * X_train';
        case 'sigmoid'
            P = tanh(alpha * (X_train * X_train') + c);
        case 'polynomial'
            P = (X_train * X_train' + c).^alpha;
        case 'none'
            P = eye(length(X_train));
    end
    diag_elements = diag(P);
    normalization_factor = sqrt(diag_elements * diag_elements');
    P = P ./ normalization_factor;
end
