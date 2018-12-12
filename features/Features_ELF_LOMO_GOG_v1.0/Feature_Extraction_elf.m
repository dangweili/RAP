% set the root path
addpath(genpath('../ELF-v2.0-Descriptor/'))
% delete the resize commond
srcPath = '../../data/RAP_dataset/'
dstPath = '../../features/'
saveMat = 'rap2_features_elf.mat'
% load the RAP_annotation.mat 
load ../../data/RAP_annotation/RAP_annotation.mat
image_names = RAP_annotation.name;
% img_size = [160 64];
imgs_feature = zeros(length(image_names),2784);
parfor i = 1:length(image_names)
    name = image_names{i};
    img = imread(strcat(srcPath,name)); % load the previous image
    imgs_feature(i,:) = genDescriptor(img,[1 1 1 1 1],[16 16 16 16 16],6,[]);
    i
end
save(strcat(dstPath,saveMat), 'imgs_feature');

