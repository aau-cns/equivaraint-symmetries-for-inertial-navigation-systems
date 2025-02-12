function shade_std(s, mean, std, alpha, color)

    % Make sure we got a single row and multi column
    if size(s,1) ~= 1
        s = s';
    end
    if size(mean,1) ~= 1
        mean = mean';
    end
    if size(std,1) ~= 1
        std = std';
    end
    
    std_curve_up = mean + std;
    std_curve_down = mean - std;
    
    % Set default value of alpha to 0.5
    if exist('alpha', 'var') == 0 || isempty(alpha)
        alpha = 0.5;
    end
    
    % Set default value of color to black
    if exist('color', 'var') == 0 || isempty(color)
        color = 'k';
    end
    
    fill([s fliplr(s)], [std_curve_up fliplr(std_curve_down)], color, 'FaceAlpha', alpha,'linestyle','none');
%     plot(s, std_curve_up, 'LineWidth', 0.1, 'Color', color)
%     plot(s, std_curve_down, 'LineWidth', 0.1, 'Color', color)
    
end

