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
feature_type = 'resnet50_pool5';
feature_Norm = 'L2';
load(sprintf('rap2_features_%s.mat', feature_type))
if strcmp(feature_Norm, 'L2')
    tmp = sqrt(sum(imgs_feature .* imgs_feature, 2));
    imgs_feature = bsxfun(@rdivide, imgs_feature, tmp);
end
% load the orignal annotation groundtruth file, named as RAP_annotation.XXX
load ../../data/RAP_annotation/RAP_annotation.mat
% selected_attribute = RAP_annotation.selected_attribute;
% labeldata = RAP_annotation.data(:, selected_attribute);
viewpoint_index = 112;
selected_attribute = 4;
labeldata = zeros(size(RAP_annotation.data, 1),4);
labeldata(:,1) = RAP_annotation.data(:,viewpoint_index) == 1;
labeldata(:,2) = RAP_annotation.data(:,viewpoint_index) == 2;
labeldata(:,3) = RAP_annotation.data(:,viewpoint_index) == 3;
labeldata(:,4) = RAP_annotation.data(:,viewpoint_index) == 4;

% svm parameters
svm_C = [0.01 0.10 1.0 10.0 100.0];
method = 1; % 1 is sample else is weighting
partition_Cnt = length(RAP_annotation.partition_attribute);
partition = RAP_annotation.partition_attribute;
% set the model path, the store should be model_featureture_svmC_partition.mat
model_path = './models';
Model = cell(length(selected_attribute),1);
Accuracy_pos = cell(length(selected_attribute), 1);
Accuracy_neg = cell(length(selected_attribute), 1);
% obtain the occlusion index for futher analysis
occlusion_type1 = sum(RAP_annotation.data(:, 113:116),2) >=1;
occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
for i=1:3
    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
end
occlusion_type2 = occlusion_type2 >= 1;
occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
occlusion_index = find(occlusion_type);

tic
% start to train the overall svm
for c_index = 1: length(svm_C)
    for iter = 1: partition_Cnt
        % parsing all the examples using index information
        if strcmp(stage, 'trainval')
            train_index = [partition{iter}.train_index partition{iter}.val_index];
            test_index = partition{iter}.test_index;
        else
            train_index = partition{iter}.train_index;
            test_index = partition{iter}.val_index;
        end
        Noise_results = zeros(length(selected_attribute),1);
        % train all the classifier in parallel
        parfor label = 1:4  % length(selected_attribute)
            sprintf('%s: %d %d %d\n', feature_type, c_index, iter, label)
            % extract the effiective train, val index
            effiective_train_index = train_index;
            effiective_test_index = test_index;

            % train a linear svm model
%             if label == 1 % process the gender attribute and ignore the uncertain
%                 idx_temp = find(labeldata(:,label) == 2);
%                 effiective_train_index = setdiff(effiective_train_index, idx_temp);
%                 effiective_test_index = setdiff(effiective_test_index, idx_temp);
%             end
%             
            % compute the distribution of pos and neg examples
            pos_train_idx = find(labeldata(effiective_train_index, label) == 1);
            neg_train_idx = find(labeldata(effiective_train_index, label) == 0);
            pos_train_cnt = length(pos_train_idx);
            neg_train_cnt = length(neg_train_idx);

            pos_train_weight = 1;
            neg_train_weight = 1;
            if method == 1% sample the samples for training
                if pos_train_cnt >= neg_train_cnt
                    temp_index = randperm(pos_train_cnt, neg_train_cnt);
                    pos_train_idx = pos_train_idx(temp_index);
                else 
                    temp_index = randperm(neg_train_cnt, pos_train_cnt);
                    neg_train_idx = neg_train_idx(temp_index);
                end
            else
                if pos_train_cnt >= neg_train_cnt
                    neg_train_weight = pos_train_cnt/neg_train_cnt;
                else
                    pos_train_weight = neg_train_cnt/pos_train_cnt;
                end
            end
            effiective_train_index = effiective_train_index([pos_train_idx' neg_train_idx']');
            effiective_train_index = effiective_train_index(randperm(length(effiective_train_index)));
            % add handle the noise data
            if isempty(pos_train_idx) || isempty(neg_train_idx)
                Noise_results(label) = 1;
                continue;
            end
            % generate the training commond
            commond_svm = sprintf('-c %f -s 1 -w0 %f -w1 %f', svm_C(c_index), neg_train_weight, pos_train_weight);
            % train the liblinear model
            model = liblinearsvmtrain(double(labeldata(effiective_train_index, label)), ...
                sparse(double(imgs_feature(effiective_train_index, :))), commond_svm);
            % processing the test data
            pos_test_idx = find(labeldata(effiective_test_index, label) == 1);
            neg_test_idx = find(labeldata(effiective_test_index, label) == 0);
            if isempty(pos_test_idx) || isempty(neg_test_idx)
                Noise_results(label) = 1;
                continue;
            end
            [p_label_pos, acc_pos, prob_pos] = liblinearsvmpredict(double(labeldata(effiective_test_index(pos_test_idx), label)), ...
                sparse(double(imgs_feature(effiective_test_index(pos_test_idx), :))), model);
            [p_label_neg, acc_neg, prob_neg] = liblinearsvmpredict(double(labeldata(effiective_test_index(neg_test_idx), label)), ...
                sparse(double(imgs_feature(effiective_test_index(neg_test_idx), :))), model);
            Model{label} = model;
            Accuracy_pos{label} = acc_pos;
            Accuracy_neg{label} = acc_neg;
        end
        % store the model and test accuracy
        % set the model path, the store should be model_featureture_svmC_partition.mat
        if strcmp(feature_Norm, 'L2')
            if strcmp(stage, 'trainval')
                store_name = sprintf('%s/model_viewpoints_trainval_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
            else
                store_name = sprintf('%s/model_viewpoints_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
            end
        else
            if strcmp(stage, 'trainval')
                store_name = sprintf('%s/model_viewpoints_trainval_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
            else
                store_name = sprintf('%s/model_viewpoints_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
            end
        end
        save(store_name, 'Model', 'Accuracy_pos', 'Accuracy_neg', 'Noise_results','-v7.3')
    end
end

toc
