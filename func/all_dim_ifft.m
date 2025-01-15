function [shiftedIFFT] = all_dim_ifft(img)
%ALL_DIM_iFFT 此处显示有关此函数的摘要
%   此处显示详细说明
imgSize = size(img);
dim = length(imgSize);
switch dim
    case 1
        shiftedIFFT = ifft(ifftshift(img));
    case 2
        shiftedIFFT = ifft2(fftshift(img));
    case 3
        shiftedIFFT = ifftn(ifftshift(ifftshift(ifftshift(img, 1), 2), 3));
    otherwise
        disp("Input image has wrong dimensions!\n");
end

end