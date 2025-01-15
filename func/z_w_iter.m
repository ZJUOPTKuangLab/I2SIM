function [z_out, w_out] = z_w_iter(opeFFT, fFFT, w, gamma)
%Z_W_ITER 此处显示有关此函数的摘要
%   此处显示详细说明
operated_f = real(ifftn(opeFFT.*fFFT));
z_out = softthresh(operated_f + w, gamma);
w_out = w + operated_f - z_out;
end