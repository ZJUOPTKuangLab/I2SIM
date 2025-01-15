function p = softthresh(y,gamma)
% proxl1 计算正则项为l1范数时的软阈值函数
% y:     x-1/L*det(f(x))
% gamma: λ/L
p = max(y-gamma,0) + min(y+gamma, 0);
end