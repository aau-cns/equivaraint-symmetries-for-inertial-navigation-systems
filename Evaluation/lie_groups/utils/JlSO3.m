function [J] = JlSO3(w)

norm_w = norm(w);

if norm(w) < 1e-3
    J = eye(size(w,1));
else
    skew_w = wedge(w);
    J = eye(size(w,1)) + ((1-cos(norm_w))/(norm_w^2))*skew_w + ((norm_w-sin(norm_w))/(norm_w^3))*skew_w*skew_w;
end


end