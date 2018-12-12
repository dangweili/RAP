addpath(genpath('../LOMO_XQDA/'))
% default size is 128*48
% delete the resize commond
srcPath = '../../data/RAP_dataset/'
dstPath = '../../features/'
saveMat = 'rap2_features_lomo.mat'
% load the RAP_annotation.mat 
load ../../data/RAP_annotation/RAP_annotation.mat
image_names = RAP_annotation.name;
N = length(image_names)
% img_size = [128 48];
imgs_feature = zeros(26960, N);
parfor i = 1:N
    name = image_names{i};
    img = imread(strcat(srcPath,name)); % load the previous image
    imgs_feature(:, i) = LOMO(imresize(img, [128 48]));
    i
end
imgs_feature = LOMO(images_128_48); % for 128*48
imgs_feature = imgs_feature';
save(strcat(dstPath,saveMat), 'imgs_feature', '-v7.3');

