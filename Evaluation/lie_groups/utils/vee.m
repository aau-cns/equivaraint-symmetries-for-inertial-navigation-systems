function [x] = vee(X)

% Vee operator
% so(3) -> R^3
% se(3) -> R^6
% se_2(3) -> R^9

    x(1:3) = [X(3,2);X(1,3);X(2,1)];

    if size(X,1) > 3
        x(4:6) = X(1:3,4);
        if size(X,1) > 4
            x(7:9) = X(1:3,5);
        end
    end

    [r, c] = size(x);

    if c > r
        x = x';
    end
     
end

