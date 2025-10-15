function [metrics] = calculate_metrics(y_true, y_pred)
    residual = y_pred - y_true;
    metrics.RMSE = sqrt(mean(residual.^2));
    metrics.MAE = mean(abs(residual));
    
    % R²计算
    tss = sum((y_true - mean(y_true)).^2);
    rss = sum(residual.^2);
    metrics.R2 = 1 - (rss / tss);
    
    % NMSE
    metrics.NMSE = mean(residual.^2) / var(y_true);
    
    % MAPE
    valid_idx = y_true ~= 0;
    if any(valid_idx)
        metrics.MAPE = mean(abs(residual(valid_idx)./y_true(valid_idx)));
    else
        metrics.MAPE = NaN;
    end
end

