% this script implement the linear svm classifier for 54 attributes
clear all;
% load the path for of feature and liblinear lib
liblinearsvm_path = ['../utils/liblinear-master/matlab'];
feature_path = ['../../features'];
addpath(genpath(liblinearsvm_path))
addpath(genpath(feature_path))

% select the feature types, six types of feature, including 
% elf, lomo, caffenet_fc6, caffenet_fc7, googlenet_pool5, resnet50_pool5
% all the features are named as imgs_feature with N*feature_dim
stage='trainval'; % using train+val for training
feature_types = {'resnet50_pool5'};
feature_Norm = 'L2';
% load the orignal annotation groundtruth file, named as RAP_annotation.XXX
load ../../data/RAP_annotation/RAP_annotation.mat
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);

% svm parameters
% svm_C = [0.01 0.10 1.0 10.0 100.0];
svm_C = [1.0]; % selected the best C
method = 1; % 1 is sample else is weighting
% partition_Cnt = length(RAP_annotation.partition_attribute);
partition_Cnt = 1;
partition = RAP_annotation.partition_attribute;
% set the model path, the store should be model_featureture_svmC_partition.mat
model_path = './models';
% add the store for visualization
Label_gt = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_pt = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_score = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_index = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
% obtain the occlusion index for futher analysis, only use the person occluded by other persons

%occlusion_type1 = sum(RAP_annotation.data(:, 113:116),2) >=1;
%occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
%for i=1:3
%    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
%end
%occlusion_type2 = occlusion_type2 >= 1;
%occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
%occlusion_index = find(occlusion_type);
occlusion_type = RAP_annotation.data(:, 119) == 1;
occlusion_index = find(occlusion_type);


%% this script only used for test, not for training
tic
% start to train the overall svm
for feature_idx = 1:length(feature_types)
    load(sprintf('rap2_features_%s.mat', feature_types{feature_idx}))
    if strcmp(feature_Norm, 'L2')
        tmp = sqrt(sum(imgs_feature .* imgs_feature, 2));
        imgs_feature = bsxfun(@rdivide, imgs_feature, tmp);
    end
    for c_index = 1: length(svm_C)
        for iter = 1: partition_Cnt
            % parsing all the examples using index information
            if strcmp(stage, 'trainval')
                test_index = partition{iter}.test_index;
            else
                test_index = partition{iter}.val_index;
            end
            % load all the classifier in parallel
            if strcmp(feature_Norm, 'L2')
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_bodycc_trainval_%s_%s_%1.2f_part%d.mat', model_path, feature_types{feature_idx}, feature_Norm, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_bodycc_%s_%s_%1.2f_part%d.mat', model_path, feature_types{feature_idx}, feature_Norm, svm_C(c_index), iter);
                end
            else
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_bodycc_trainval_%s_%1.2f_part%d.mat', model_path, feature_types{feature_idx}, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_bodycc_%s_%1.2f_part%d.mat', model_path, feature_types{feature_idx}, svm_C(c_index), iter);
                end
            end
            load(store_name, 'Model') % only load Model variable, some other varibals are overwrite

            % as we want to use the no-occlusion for training, so here is it
            effiective_test_index = intersect(occlusion_index, test_index);

            for label = 1:length(selected_attribute)
                sprintf('%s: %d %d %d\n', feature_types{feature_idx}, c_index, iter, label)
                 
                model = Model{label};
                % compute the distribution of pos and neg examples
                [p_label, acc, prob] = liblinearsvmpredict(double(labeldata(effiective_test_index, label)), ...
                    sparse(double(imgs_feature(effiective_test_index,:))), model);
                % record the gt,pt label, index and score
                Label_gt{feature_idx, c_index, iter, label} = labeldata(effiective_test_index, label);
                Label_pt{feature_idx, c_index, iter, label} = p_label;
                Label_score{feature_idx, c_index, iter, label} = prob;
                Label_index{feature_idx, c_index, iter, label} = effiective_test_index;
            end
        end
    end
end
toc

%% store the results for visualization
gt_result = Label_gt{1,1,1,1};
pt_result = Label_pt{1,1,1,1};
pt_prob = Label_score{1,1,1,1};
test_index = effiective_test_index;
for i = 2:length(selected_attribute)
    gt_result = [gt_result Label_gt{1,1,1,i}];
    pt_result = [pt_result Label_pt{1,1,1,i}];
    pt_prob = [pt_prob Label_score{1,1,1,i}];
end
pt_prob = 1./(1 + exp(-pt_prob));
%% store the rank list for each attribute
fid = fopen('./personbyperson_recognition_result/personbyperson_result.txt', 'w+');
for i = 1:length(test_index)
   % image_name, 1, 1, ..., 1, 1
   % save the image name
   image_name = RAP_annotation.name{test_index(i)};
   fprintf(fid, sprintf('%s\n', image_name));
   % save the attribute index
   for j=1:length(selected_attribute)
       if j <= 9
          fprintf(fid, sprintf('%.2f ', j));
       else
          fprintf(fid, sprintf('%.1f ', j));
       end
   end
   fprintf(fid, '\n');
   % save the gt labels
   for j=1:length(selected_attribute)
       fprintf(fid, sprintf('%.2f ', gt_result(i, j)));
   end
   fprintf(fid, '\n');
   % save the pt labels
   for j=1:length(selected_attribute)
       fprintf(fid, sprintf('%.2f ', pt_result(i, j)));
   end
   fprintf(fid, '\n');
   % save the scores
   for j=1:length(selected_attribute)
       fprintf(fid, sprintf('%.2f ', pt_prob(i, j)));
   end
   fprintf(fid, '\n');
   % load and save the image
   % copyfile(['/mnt/data1/lidw/dataset/RAP2/images-pedestrian-orignal/' image_name], ...
   %     '/mnt/data1/lidw/journal-tcsvt/person-attribute/baseline-svm/personbyperson_recognition_result/images');
end
fclose(fid);



