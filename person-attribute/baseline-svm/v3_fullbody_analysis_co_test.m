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
feature_types = {'caffenet_fc6', 'resnet50_pool5'};
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
% add the store for visualization
Label_gt = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_pt = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_score = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
Label_index = cell(length(feature_types), length(svm_C), partition_Cnt, length(selected_attribute));
% obtain the occlusion index for futher analysis
occlusion_type1 = sum(RAP_annotation.data(:, 113:116),2) >=1;
occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
for i=1:3
    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
end
occlusion_type2 = occlusion_type2 >= 1;
occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
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

            for label = 1:length(selected_attribute)
                sprintf('%s: %d %d %d\n', feature_types{feature_idx}, c_index, iter, label)
                % as we want to use the no-occlusion for training, so here is it
                effiective_test_index = intersect(occlusion_index, test_index);
                
                % train a linear svm model
                if label == 1 % process the gender attribute and ignore the uncertain
                    idx_temp = find(labeldata(:,label) == 2);
                    effiective_test_index = setdiff(effiective_test_index, idx_temp);
                end
                 
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

%% store all the results into files
Results = zeros(length(feature_types), length(svm_C), 5, partition_Cnt, length(selected_attribute), 3);
for feature_idx = 1:length(feature_types)
    for c_index = 1: length(svm_C)
        for iter = 1: partition_Cnt
            for occ = 1:4
                Tmp = zeros(length(selected_attribute), 3);
                for label = 1:length(selected_attribute)
                    % process the four types of occlusion
                    % verify the efficience of pos and neg
                    index = Label_index{feature_idx, c_index, iter, label};
                    occ_index = index(find(RAP_annotation.data(index, 112 + occ) == 1));
                    occ_index_flag = RAP_annotation.data(index, 112 + occ ) == 1;

                    tmp_label = RAP_annotation.data(occ_index, label);
                    if sum(tmp_label) < length(tmp_label)
                        eff = 1;
                    else
                        eff = 0;
                    end
                    if eff == 0
                        Tmp(label,:) = -1;
                    else
                        result_tmp = rap_evaluation(Label_pt{feature_idx, c_index, iter, label}(occ_index_flag), ...
                            Label_gt{feature_idx, c_index, iter, label}(occ_index_flag));
                        Tmp(label, 1) = [mean(result_tmp.label_accuracy_pos)];
                        Tmp(label, 2) = [mean(result_tmp.label_accuracy_neg)];
                        Tmp(label, 3) = ( Tmp(label, 1) + Tmp(label, 2) )/2;
                    end
                end
                Results(feature_idx, c_index, occ, iter, :, :) = Tmp;
            end
            % compute the mean results
            Tmp = zeros(length(selected_attribute), 3);
            for label=1:length(selected_attribute)
                result_tmp = rap_evaluation(Label_pt{feature_idx, c_index, iter, label}, Label_gt{feature_idx, c_index, iter, label});
                Tmp(label, 1) = [mean(result_tmp.label_accuracy_pos)];
                Tmp(label, 2) = [mean(result_tmp.label_accuracy_neg)];
                Tmp(label, 3) = ( Tmp(label, 1) + Tmp(label, 2) )/2;
            end
            Results(feature_idx, c_index, 5, iter, :, :) = Tmp;
        end
    end
end

% statistics
Results_occ = mean(Results, 4);
Results_occ = squeeze(Results_occ); % feature_type, occ, att, results
save('./results/Results_V3_Occlusiontypes.mat', 'Results_occ', 'feature_types');

% display
tmp = Results == -1;
sum(tmp(:))
R = squeeze(mean(Results_occ, 3));
R(1, :, 3)
R(2, :, 3)

