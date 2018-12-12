% this script implement the linear svm classifier for 54 attributes
% load the path for of feature and liblinear lib
liblinearsvm_path = ['../utils/liblinear-master/matlab'];
feature_path = ['../../features'];
addpath(genpath(liblinearsvm_path))
addpath(genpath(feature_path))

% select the feature types, six types of feature, including 
% elf, lomo, caffenet_fc6, caffenet_fc7, googlenet_pool5, resnet50_pool5
% all the features are named as imgs_feature with N*feature_dim
stage='trainval'; % trainval or others. 
feature_types = {'pcaelf', 'caffenet_fc6', 'caffenet_fc7', 'googlelenet_pool5', 'resnet50_pool5'};
feature_Norm = 'L2'; % L2

% load the orignal annotation groundtruth file, named as RAP_annotation.XXX
load ../../data/RAP_annotation/RAP_annotation.mat
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);

% svm parameters
svm_C = [0.01 0.10 1.0 10.0 100.0];
% svm_C = [0.01];
method = 1; % 1 is sample else is weighting
partition_Cnt = length(RAP_annotation.partition_attribute);
partition = RAP_annotation.partition_attribute;
% set the model path, the store should be model_featureture_svmC_partition.mat
model_path = './models';
% obtain the occlusion index for futher analysis
occlusion_type1 = sum(RAP_annotation.data(:, 113:116),2) >=1;
occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
for i=1:3
    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
end
occlusion_type2 = occlusion_type2 >= 1;
occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
occlusion_index = find(occlusion_type);

% malloc the results
pt_Results = cell(length(feature_types), length(svm_C), partition_Cnt); 

tic
% start to train the overall svm
for feat_idx = 1:length(feature_types)
    % load the test features
    load(sprintf('rap2_features_%s.mat', feature_types{feat_idx}))
    if strcmp(feature_Norm, 'L2')
        tmp = sqrt(sum(imgs_feature .* imgs_feature, 2));
        imgs_feature = bsxfun(@rdivide, imgs_feature, tmp);
    end
    for c_index = 1:length(svm_C) 
        for iter = 1: partition_Cnt
            % parsing all the examples using index information
            if strcmp(stage, 'trainval')
                % load the model of svm_C
                test_index = partition{iter}.test_index;
            else
                test_index = partition{iter}.val_index;
            end
            pt_tmp = zeros(length(test_index), length(selected_attribute));
            pt_tmp_score = zeros(length(test_index), length(selected_attribute));
            gt_tmp = zeros(length(test_index), length(selected_attribute));
            % load the pretrained linear svm model
            if strcmp(feature_Norm, 'L2')
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_trainval_%s_%s_%1.2f_part%d.mat', model_path, feature_types{feat_idx}, feature_Norm, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_%s_%s_%1.2f_part%d.mat', model_path, feature_types{feat_idx}, feature_Norm, svm_C(c_index), iter);
                end
            else
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_trainval_%s_%1.2f_part%d.mat', model_path, feature_types{feat_idx}, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_%s_%1.2f_part%d.mat', model_path, feature_types{feat_idx}, svm_C(c_index), iter);
                end
            end
            load(store_name)
            % train all the classifier in parallel
            parfor label = 1:length(selected_attribute)
                % output the test information
                sprintf('%s: %d %d %d\n', feature_types{feat_idx}, c_index, iter, label)
                model = Model{label};
                [p_label, ~, p_score] = liblinearsvmpredict(double(labeldata(test_index, label)), ...
                    sparse(double(imgs_feature(test_index, :))), model);
                pt_tmp(:, label) = p_label;
                pt_tmp_score(:, label) = p_score;
                gt_tmp(:, label) = labeldata(test_index, label);
            end
            pt_Results{feat_idx, c_index, iter}.pt_label = pt_tmp;
            pt_Results{feat_idx, c_index, iter}.gt_label = gt_tmp;
            pt_Results{feat_idx, c_index, iter}.pt_score = pt_tmp_score;
        end
    end
end
toc

Results = cell(length(feature_types), length(svm_C), partition_Cnt); 
% obtain the label-based and example-based results
for feat_idx = 1:length(feature_types)
    for c_index = 1:length(svm_C) 
        for iter = 1: partition_Cnt
            % parsing all the examples using index information
            Results{feat_idx, c_index, iter} = rap_evaluation(...
                pt_Results{feat_idx, c_index, iter}.pt_label, ...
                pt_Results{feat_idx, c_index, iter}.gt_label);
        end
    end
end

% save all the results for display
if strcmp(feature_Norm, 'L2')
    if strcmp(stage, 'trainval')
        save('results/Results_L2_test.mat', 'Results', 'pt_Results', 'svm_C', 'feature_types', '-v7.3')
    else
        save('results/Results_L2_val.mat', 'Results', 'pt_Results', 'svm_C', 'feature_types', '-v7.3')
    end
else
    if strcmp(stage, 'trainval')
        save('results/Results_test.mat', 'Results', 'pt_Results', 'svm_C', 'feature_types', '-v7.3')
    else
        save('results/Results_val.mat', 'Results', 'pt_Results', 'svm_C', 'feature_types', '-v7.3')
    end
end
