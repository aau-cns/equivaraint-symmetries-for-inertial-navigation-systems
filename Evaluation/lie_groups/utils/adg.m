function [ad] = adg(U)

    % adjoint matrix
    
    if isequal(size(U), [3,3])
        ad = U(1:3,1:3);
    elseif isequal(size(U), [4,4])
        W = U(1:3,1:3);
        v = U(1:3,4);
        ad = [W zeros(3); wedge(v)*W W];
    elseif isequal(size(U), [5,5])
        W = U(1:3,1:3);
        v = U(1:3,4);
        a = U(1:3,5);
        ad = [W zeros(3) zeros(3); wedge(v)*W W zeros(3); wedge(a)*W zeros(3) W];
    else
        warning('Unimplemented adjoint matrix for the given matrix size.')
    end

end

