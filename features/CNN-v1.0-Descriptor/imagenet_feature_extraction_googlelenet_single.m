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
FeatureLayer = 'pool5/7x7_s1'; % useless
savemat = '../rap2_features_googlelenet_pool5.mat'
model_def = './models/GoogleLeNet/deploy_pool5.prototxt' % this is caffenet model def
model_file = './models/GoogleLeNet/bvlc_googlenet.caffemodel' % this is caffenet model
caffe_GPU = 1;
GPU_ID = 1;
if caffe_GPU
    caffe.set_mode_gpu();
    caffe.set_device(GPU_ID);
else
    caffe.set_mode_cpu();
end
net = caffe.Net( model_def, model_file, 'test');

%%% load the image data
path_data_all = './../../data/RAP_dataset/'
load ../../data/RAP_annotation/RAP_annotation.mat
images_name = RAP_annotation.name;
ImgCnt = length(images_name);
Feature_Dim = 1024;
imgs_feature = zeros([Feature_Dim, ImgCnt], 'single');
CropSize = 224;
mean_data = single(repmat(reshape([104 117 123], [1 1 3]), [CropSize, CropSize]));
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
caffe.reset_all(); % reset caffe
toc
imgs_feature = imgs_feature';
save(savemat, 'imgs_feature', '-v7.3');
