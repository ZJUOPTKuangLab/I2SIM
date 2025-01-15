function [z_out,w_out] = z_w_iter_isotropic(C_ij, C_sum, gamma)
%Z_W_ITER_ISOTROPIC 迭代isotropic形式下的z和w
%   此处显示详细说明
denominator = C_sum;
denominator(denominator == 0) = 1;
z_out = max(C_sum-gamma, 0) .* C_ij ./ denominator;
w_out = C_ij - z_out;
end