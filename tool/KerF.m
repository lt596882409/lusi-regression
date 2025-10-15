function K = KerF(X1, kpar, X2)
%% Explain:
% Construct the positive (semi-) definite (symmetric) kernel matrix
%
% Inputs: 
%     X1          - mt x n matrix, #m1 Test vector with n dimension
%     kpar.type - kernel type (by default 'RBF_kernel')
%     kpar.kp1 - kernel parameter 1 
%     kpar.kp2 - kernel parameter 2 
%     X2          - m x n matrix, #m BASE vector with n dimension
% Outputs: 
%     K           -  mt x m kernel matrix
%     K(a,b)      -  the similarity measure of vector TstX(a) and X(b)
% 
% Written by Lingwei Huang, lateset update: 2021.09.15. 
% Copyright 2019-2021  Lingwei Huang. 
% Update by Liu tian, lateset update: 2025.02.27.

%% Code:
    switch lower(kpar.ktype)
        case 'linear'
            K = X1 * X2';
        case 'rbf'
            gamma = kpar.kp1; % gamma=1/(2*sig^2)
            %gamma = 1/(2*kpar.pars(1)^2);
            X1_sq = sum(X1.^2, 2);
            X2_sq = sum(X2.^2, 2)';
            K = exp(-gamma*(X1_sq + X2_sq - 2*X1*X2')); % lib version
            % sig = kp1;
            % K = exp(  - (pdist2(TstX,X).^2) ./ (2*sig^2)  ); % micro version
            % sig2 = kp1;
            % K = exp(  - (pdist2(TstX,X).^2) ./ (2*sig2)  ); % old version
        case 'poly'
            degree = kpar.kp1;
            offset = kpar.kp2;
            K = (X1*X2' + offset).^degree;
        case 'rbfnt' % rbf with negative tail
            gamma = kpar.kp1; % gamma=1/(2*sig^2), default 1/n
            coef0 = kpar.kp2; % default 1
            K = (  coef0 - gamma .* (pdist2(TstX,X).^2)  ) * ...
                exp(  -gamma .* (pdist2(TstX,X).^2)  );
        case 'sigmoid'
            gamma = kpar.kp1; % default 1/n 
            coef0 = kpar.kp2;    % default 0 
            K = tanh( gamma * TstX*X' + coef0 );
        %case 'sinc'
            % emt = ones(mt,1); 
            % em = ones(m,1); 
            % XXh1 = TstX*emt * em';
            % XXh2 =      X*em   * emt';
            % K = XXh1-XXh2';
            % K = sinc(kp1.*K);
        %case 'wav'
            % XXh1 = sum(TstX.^2,2)*ones(1,m);
            % XXh2 = sum(X.^2,2)*ones(1,mt);
            % K = XXh1+XXh2' - 2*(TstX*X');
            % XXh11 = sum(TstX,2)*ones(1,m);
            % XXh22 = sum(Xt,2)*ones(1,mt);
            % K1 = XXh11-XXh22';
            % K = cos(kp2.*K1).*exp(-kp1.*K);
        otherwise
            error('Unsupported kernel type');
    end
end