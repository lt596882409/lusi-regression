function [best_params, best_score] = grid_search(fun, param_grid, X, y, cv)
    param_names = fieldnames(param_grid);
    param_comb = allcomb(param_grid.(param_names{1}), param_grid.(param_names{2}));
    best_score = inf;
    for i = 1:size(param_comb,1)
        params = cell2struct(num2cell(param_comb(i,:)), param_names, 2);
        score = cross_val(fun, X, y, params, cv);
        if score < best_score
            best_score = score;
            best_params = params;
        end
    end
end