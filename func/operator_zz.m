function operator = operator_zz(lambda_t)
%OPERATOR_ZZ 
%   此处显示详细说明
operator(1, 1, :) = [1 -2 1];
operator = lambda_t * operator;
end