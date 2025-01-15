clc; close all; clear;
%本程序实现各向同性（isotropic）Hessian正则迭代。
%%
addpath('func\');
rawImg = double(loadtiff(strcat('.\src\tubule_647_4Pi_2.tif')));
rawImg = rawImg - min(rawImg(:));
rawImg = rawImg ./ max(rawImg(:));
imgSize = size(rawImg);

% load OTF Mask
load(".\src\otfMask\simOtfMask_z21_680.mat");
otfMask = Mask_final;
otfMask = fftshift(otfMask);
clear("Mask_final");

%% initialize

% Change parameter for better performance
% ------------------------------------------------------
lambda = [0.03 0.01 1];   % lambda_1, lambda_2, lambda_z   % sample
p = 0.6;   % rho
% lambda = [0.05 0.0005 1];   % lambda_1, lambda_2, lambda_z      % beads
% p = 0.4;   % rho
useGpu = 1;
iterNum = 100;
Ttol = 1e-7;
% ------------------------------------------------------

% Do not change
ope_xx = [1 -2 1];
ope_yy = [1 -2 1]';
ope_xy = sqrt(2) * [0  0  0;
                    0  1 -1;
                    0 -1  1];
ope_xz = operator_xz(lambda(3))/sqrt(2);
ope_yz = operator_yz(lambda(3))/sqrt(2);
ope_zz = operator_zz(lambda(3));

ope_xxFFT = gpuArray(psf2otf(ope_xx, imgSize));
ope_yyFFT = gpuArray(psf2otf(ope_yy, imgSize));
ope_xyFFT = gpuArray(psf2otf(ope_xy, imgSize));
ope_xzFFT = gpuArray(psf2otf(ope_xz, imgSize));
ope_yzFFT = gpuArray(psf2otf(ope_yz, imgSize));
ope_zzFFT = gpuArray(psf2otf(ope_zz, imgSize));

if useGpu == 1
    f = gpuArray.zeros(imgSize);
    w1 = f;
    w_xx = f;
    w_xy = f;
    w_yy = f;
    w_xz = f;
    w_yz = f;
    w_zz = f;
    f = gpuArray(rawImg);
    b = gpuArray(rawImg);
else
    f = zeros(imgSize);
    w1 = f;
    w_xx = f;
    w_xy = f;
    w_yy = f;
    w_xz = f;
    w_yz = f;
    w_zz = f;
    f = rawImg;
    b = rawImg;
end

denominator = 1 + abs(ope_xxFFT).^2 + abs(ope_xyFFT).^2 + abs(ope_yyFFT).^2 + ...
              abs(ope_xzFFT).^2 + abs(ope_yzFFT).^2 + abs(ope_zzFFT).^2 + 1/p*abs(otfMask).^2;

bFFT = fftn(b);
fChange = gpuArray.zeros(1, iterNum);

%% iteration begin
fprintf("iteration begin. total number: %d", iterNum);
for ii = 1:iterNum
    if ii == 1
        numerator = conj(otfMask).*bFFT/p;
    else
        numerator = fftn(z1-w1) + conj(ope_xxFFT).*fftn(z_xx-w_xx) + conj(ope_xyFFT).*fftn(z_xy-w_xy) + ...
                    conj(ope_yyFFT).*fftn(z_yy-w_yy) + conj(ope_xzFFT).*fftn(z_xz-w_xz) + ...
                    conj(ope_yzFFT).*fftn(z_yz-w_yz) + conj(ope_zzFFT).*fftn(z_zz-w_zz) + conj(otfMask).*bFFT/p;
    end
    fPre = f;
    f = real(ifftn(numerator./denominator));
    f(f<0) = 0;
    fFFT = fftn(f);

    z1 = softthresh(f + w1, lambda(1)/p);
    w1 = w1 + f - z1;
    % C_i,j caliculation, C_i,j = Af+w
    C_xx = ifftn(ope_xxFFT.*fFFT) + w_xx;
    C_xy = ifftn(ope_xyFFT.*fFFT) + w_xy;
    C_yy = ifftn(ope_yyFFT.*fFFT) + w_yy;
    C_xz = ifftn(ope_xzFFT.*fFFT) + w_xz;
    C_yz = ifftn(ope_yzFFT.*fFFT) + w_yz;
    C_zz = ifftn(ope_zzFFT.*fFFT) + w_zz;
    C_sum = sqrt(C_xx.^2 + C_xy.^2 + C_yy.^2 + C_xz.^2 + C_yz.^2 + C_zz.^2);

    [z_xx, w_xx] = z_w_iter_isotropic(C_xx, C_sum, lambda(2)/p);
    [z_xy, w_xy] = z_w_iter_isotropic(C_xy, C_sum, lambda(2)/p);
    [z_yy, w_yy] = z_w_iter_isotropic(C_yy, C_sum, lambda(2)/p);
    [z_xz, w_xz] = z_w_iter_isotropic(C_xz, C_sum, lambda(2)/p);
    [z_yz, w_yz] = z_w_iter_isotropic(C_yz, C_sum, lambda(2)/p);
    [z_zz, w_zz] = z_w_iter_isotropic(C_zz, C_sum, lambda(2)/p);

    if mod((ii-1), 20) == 0
        fprintf("\n");
    end
    fprintf(".");
    imshow(f(:,:,(imgSize(3)+1)/2));

    fChange(ii) = normArray(f-fPre)/normArray(fPre);
    if fChange(ii) < Ttol
        break;
    end
end

fChange = fChange(1:ii);
if useGpu == 1
    fChange = gather(fChange);
    restoredImg = gather(f);
end
otfMask = gather(otfMask);

%% result
close all;
% % xy
% figure(); sliceViewer(rawImg, []); title("Raw Image");
% figure(); sliceViewer(restoredImg, []); title("Restored Image");
% % yz
% figure(); sliceViewer(permute(rawImg, [3 1 2]), []); title("yz Raw Image");
% figure(); sliceViewer(permute(restoredImg, [3 1 2]), []); title("yz Restored Image");
% % xz
% figure(); sliceViewer(permute(rawImg, [3 2 1]), []); title("xz Raw Image");
% figure(); sliceViewer(permute(restoredImg, [3 2 1]), []); title("xz Restored Image");
% convergence
% figure(); plot(fChange(2:end), '-o'); title("Convergence");
% figure(); sliceViewer(log(abs(restoredFFT)+1), []); title("Fixed FFT");
%% save image
simImgStack = restoredImg./max(restoredImg(:));
imgToSave = uint16((2^16-1)*simImgStack);
% mkdir("W:\phd\simulation\4pi hessian SIM\result\tubule", stackName);
savePath = '.\result\restored_microtubule_1.tif';
for zz = 1:imgSize(3)
    imwrite(imgToSave(:,:,zz), savePath, 'WriteMode', 'append');
end