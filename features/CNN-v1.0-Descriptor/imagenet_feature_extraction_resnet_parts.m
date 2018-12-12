%%% this script compute the casia_nlpr_hari dataset feature
clc;
clear all;
close all;
tic
% genpath and addpath to the matlab search path
path_caffe = '/home/lidw/caffe_test/matlab';
% addpath caffe to the system search
addpath( genpath(path_caffe) );

% set the feature layer
FeatureLayer = 'pool5'; % useless
model_def = './models/ResNet/deploy_pool5.prototxt' % this is caffenet model def
model_file = './models/ResNet/ResNet-50-model.caffemodel' % this is caffenet model
caffe_GPU = 1;
GPU_ID = 2;
if caffe_GPU
    caffe.set_mode_gpu();
    caffe.set_device(GPU_ID);
else
    caffe.set_mode_cpu();
end
net = caffe.Net( model_def, model_file, 'test');

%%% load the image data
load ../../data/RAP_annotation/RAP_annotation.mat
images_name = RAP_annotation.name;
ImgCnt = length(images_name);
Feature_Dim = 2048;
% imgs_feature = zeros([Feature_Dim, ImgCnt], 'single');
CropSize = 224;
load ./models/ResNet/ResNet_mean.mat 
mean_data = imagenet_mean;
indices = [0 224-CropSize] + 1;
center = floor(indices(2) / 2) + 1;
mean_data = single(mean_data(center:center+CropSize-1, center:center+CropSize-1, :));

% processing the headshoulder
path_data_all = './../../data/RAP_dataset_hs/'
savemat = '../rap2_features_resnet50_pool5_parths.mat'
imgs_feature = zeros([Feature_Dim, ImgCnt], 'single');
% preprocessing the image
for i = 1:ImgCnt
    if mod(i, 100) == 0
        i
    end
    batchData = imread( strcat( path_data_all,images_name{i}) );
    batchData = imresize(batchData, [CropSize, CropSize], 'bilinear');
    batchData = batchData(:,:,[3 2 1]); % change from rgb to bgr
    batchData = permute(batchData, [2 1 3]); % change col to row
    batchData = single(batchData) - mean_data;
    batchData = {reshape(batchData, [size(batchData) 1])};
    batchFeature = net.forward(batchData);
    batchFeature = batchFeature{1};
    imgs_feature(:, i) = batchFeature(:);
end
imgs_feature = imgs_feature';
save(savemat, 'imgs_feature', '-v7.3');

% processing the uppperbody
path_data_all = './../../data/RAP_dataset_ub/'
savemat = '../rap2_features_resnet50_pool5_partub.mat'
imgs_feature = zeros([Feature_Dim, ImgCnt], 'single');
% preprocessing the image
for i = 1:ImgCnt
    if mod(i, 100) == 0
        i
    end
    batchData = imread( strcat( path_data_all,images_name{i}) );
    batchData = imresize(batchData, [CropSize, CropSize], 'bilinear');
    batchData = batchData(:,:,[3 2 1]); % change from rgb to bgr
    batchData = permute(batchData, [2 1 3]); % change col to row
    batchData = single(batchData) - mean_data;
    batchData = {reshape(batchData, [size(batchData) 1])};
    batchFeature = net.forward(batchData);
    batchFeature = batchFeature{1};
    imgs_feature(:, i) = batchFeature(:);
end
imgs_feature = imgs_feature';
save(savemat, 'imgs_feature', '-v7.3');


% processing the lowerbody
path_data_all = './../../data/RAP_dataset_lb/'
savemat = '../rap2_features_resnet50_pool5_partlb.mat'
imgs_feature = zeros([Feature_Dim, ImgCnt], 'single');
% preprocessing the image
for i = 1:ImgCnt
    if mod(i, 100) == 0
        i
    end
    batchData = imread( strcat( path_data_all,images_name{i}) );
    batchData = imresize(batchData, [CropSize, CropSize], 'bilinear');
    batchData = batchData(:,:,[3 2 1]); % change from rgb to bgr
    batchData = permute(batchData, [2 1 3]); % change col to row
    batchData = single(batchData) - mean_data;
    batchData = {reshape(batchData, [size(batchData) 1])};
    batchFeature = net.forward(batchData);
    batchFeature = batchFeature{1};
    imgs_feature(:, i) = batchFeature(:);
end
imgs_feature = imgs_feature';
save(savemat, 'imgs_feature', '-v7.3');


caffe.reset_all(); % reset caffe
toc
