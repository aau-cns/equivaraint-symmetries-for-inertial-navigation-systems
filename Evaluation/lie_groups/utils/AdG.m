function [Ad] = AdG(T)

    % Adjoint matrix

    if isequal(size(T), [3,3])
        Ad = T(1:3,1:3);
    elseif isequal(size(T), [4,4])
        R = T(1:3,1:3);
        x = T(1:3,4);   
        Ad = [R zeros(3); wedge(x)*R R];
    elseif isequal(size(T), [5,5])
        R = T(1:3,1:3);
        x = T(1:3,4);
        y = T(1:3,5);
        Ad = [R zeros(3) zeros(3); wedge(x)*R R zeros(3); wedge(y)*R zeros(3) R];
    else
        warning('Unimplemented Adjoint for the given matrix size.')
    end
    
end

