function operator = operator_yz(lambda_t)
%OPERATOR_YZ 
%   此处显示详细说明
operatorSlice1(:,1,:) = [0  0  0;
                         0  1 -1;
                         0 -1  1];
operator = 2 * sqrt(lambda_t) * operatorSlice1;
end