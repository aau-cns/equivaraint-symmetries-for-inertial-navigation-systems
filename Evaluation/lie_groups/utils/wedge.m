function [U] = wedge(u)

% Wedge operator
% R^3 -> so(3)
% R^6 -> se(3)
% R^9 -> se_2(3)

    W = [0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
    
    if length(u) == 3
        U = W;
    elseif length(u) == 6
        U = [W u(4:6); zeros(1,4)];
    elseif length(u) == 9
        U = [W u(4:6) u(7:9); zeros(2,5)];
    end
    
end

