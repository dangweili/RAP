% this script implement the linear svm classifier for 54 attributes
% load the path for of feature and liblinear lib
liblinearsvm_path = ['../utils/liblinear-master/matlab'];
feature_path = ['../../features'];
addpath(genpath(liblinearsvm_path))
addpath(genpath(feature_path))

% select the feature types, six types of feature, including 
% elf, lomo, caffenet_fc6, caffenet_fc7, googlenet_pool5, resnet50_pool5
% all the features are named as imgs_feature with N*feature_dim
stage='trainval'; % using train+val for training
feature_types = {'caffenet_fc6', 'caffenet_fc7', 'resnet50_pool5'}; % change it by hand
part_types = {'hs', 'ub', 'lb'};
feature_Norm = 'L2';

% load the orignal annotation groundtruth file, named as RAP_annotation.XXX
load ../../data/RAP_annotation/RAP_annotation.mat
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);

% svm parameters
% svm_C = [0.01 0.10 1.0 10.0 100.0];
svm_C = [1.0]; % selected the best C
method = 1; % 1 is sample else is weighting
partition_Cnt = length(RAP_annotation.partition_attribute);
partition = RAP_annotation.partition_attribute;
% set the model path, the store should be model_featureture_svmC_partition.mat
model_path = './models';
Results = zeros(length(feature_types), length(part_types), length(svm_C), partition_Cnt, length(selected_attribute), 3); % 
tic
for feature_idx = 1:length(feature_types)
    for part_idx = 1: length(part_types)
        % start to train the overall svm
        for c_index = 1: length(svm_C)
            for iter = 1: partition_Cnt
                % load the model
                if strcmp(feature_Norm, 'L2')
                    if strcmp(stage, 'trainval')
                        store_name = sprintf('%s/model_partcc_%s_trainval_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_types{feature_idx}, feature_Norm, svm_C(c_index), iter);
                    else
                        store_name = sprintf('%s/model_partcc_%s_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_types{feature_idx}, feature_Norm, svm_C(c_index), iter);
                    end
                else
                    if strcmp(stage, 'trainval')
                        store_name = sprintf('%s/model_partcc_%s_trainval_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_types{feature_idx}, svm_C(c_index), iter);
                    else
                        store_name = sprintf('%s/model_partcc_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_type{feature_idx}, svm_C(c_index), iter);
                    end
                end
                % save(store_name, 'Model', 'Accuracy_pos', 'Accuracy_neg', 'Noise_results', 'Label_gt', 'Label_pt', 'Label_score', 'Label_index','-v7.3')
                load(store_name)
                Tmp = zeros(length(selected_attribute), 3);
                for label = 1:length(selected_attribute)
                    result_tmp = rap_evaluation(Label_pt{label}, Label_gt{label});
                    Tmp(label, 1) = mean(result_tmp.label_accuracy_pos);
                    Tmp(label, 2) = mean(result_tmp.label_accuracy_neg);
                    Tmp(label, 3) = (Tmp(label, 1) + Tmp(label, 2))/2;
                end
                Results(feature_idx, part_idx, c_index, iter, :, :) = Tmp;
            end
        end
    end
end
toc

% static all the three results for display
Results_parts = squeeze(mean(Results, 4));
save('./results/Result_V4_Parts.mat', 'Results_parts', 'part_types', 'feature_types');

% display some results
Results_parts_ave = squeeze(mean(Results_parts, 3));
% 1: feature, 2: part 3: results of {pos, neg, ave}
Results_parts_ave

