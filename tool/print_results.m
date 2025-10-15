function print_results(result_log, ModS, type)
fprintf('------\n');
fprintf('Result\n');
fprintf('------\n');
for m = 1:length(ModS)
    Mod = ModS(m);
    metrics_all = [result_log.(Mod).Metrics];
    metric_fields = fieldnames(metrics_all);
    avg_metrics = struct();
    for j = 1:numel(metric_fields)
        field = metric_fields{j}; % 指标名，例如 'rmse', 'mape', 'mae', 'r2'
        avg_metrics.(field) = mean([metrics_all.(field)]);
        std_metrics.(field) = std([metrics_all.(field)]);
    end
    switch type
        case '6'
            fprintf(Mod);
            fprintf(' ');
            for j = 3:numel(metric_fields)
                field = metric_fields{j};
                fprintf('%.4f ± %.4f', avg_metrics.(field), std_metrics.(field));
                fprintf(' ');
            end
            time = result_log.(Mod).Time;
            fprintf(' %.4e\n', mean(time));
    end
end
end


