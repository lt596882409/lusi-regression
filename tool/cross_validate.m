function [score] = cross_validate(model_config, Train, Params, fold)
    model_fun = model_config.fun;
    model_name = model_config.name;
    cv = cvpartition(size(Train.X, 1), 'KFold', fold);
    fold_loss = zeros(cv.NumTestSets, 1);
    if isfield(Params, 'V')
        V = Params.V;
    end
    if isfield(Params, 'P')
        P = Params.P;
    end
    for k = 1:cv.NumTestSets
        trainIdx = training(cv, k);
        testIdx = test(cv, k);
        Tes.X = Train.X(testIdx,:);
        Tes.Y = Train.Y(testIdx);
        Trn.X = Train.X(trainIdx,:);
        Trn.Y = Train.Y(trainIdx);
        switch model_name
            case {'VR', 'LSSVR'}
                Params.V = V(trainIdx, trainIdx);
            case {'RUSI_V', 'RUSI_I'}
                Params.V = V(trainIdx, trainIdx);
                Params.P = P(trainIdx, trainIdx);
        end
        [Pred_Y] = model_fun(Tes.X, Trn, Params);
        fold_loss(k) = sqrt(mean((Pred_Y - Tes.Y).^2));
    end
    score = mean(fold_loss);
end