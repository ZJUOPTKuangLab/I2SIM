function [shiftedFFT] = all_dim_fft(img)
%ALL_DIM_FFT 此处显示有关此函数的摘要
%   此处显示详细说明
imgSize = size(img);
dim = length(imgSize);
switch dim
    case 1
        shiftedFFT = fftshift(fft(img));
    case 2
        shiftedFFT = fftshift(fft2(img));
    case 3
        shiftedFFT = fftshift(fftshift(fftshift(fftn(img), 1), 2), 3);
    otherwise
        disp("Input image has wrong dimensions!\n");
end

end