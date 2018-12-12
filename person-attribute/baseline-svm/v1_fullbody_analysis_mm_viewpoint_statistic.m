feature_Norm = 'L2';
svm_C = [0.01 0.10 1.0 10.0 100.0];
method = 1;
model_path = './models';
stage = 'trainval';
feature_types = {'pcaelf', 'caffenet_fc6', 'caffenet_fc7', 'resnet50_pool5'};
iter_Cnt = 5;
Results = zeros(length(feature_types), length(svm_C), 4, iter_Cnt);

for feature_index = 1:length(feature_types)
    feature_type = feature_types{feature_index};
    for c_index = 1:length(svm_C)
        for iter=1:iter_Cnt
            if strcmp(feature_Norm, 'L2')
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_viewpoint_bodycc_trainval_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_viewpoint_bodycc_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
                end
            else
                if strcmp(stage, 'trainval')
                    store_name = sprintf('%s/model_viewpoint_bodycc_trainval_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
                else
                    store_name = sprintf('%s/model_viewpoint_bodycc_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
                end
            end
            load(store_name)
            % static the results
            tmp = [];
            for v=1:4
                Results(feature_index, c_index, v, iter)  = (Accuracy_pos{v}(1) + Accuracy_neg{v}(1) )/2;
            end
        end
    end
end
% average 
Results = mean(Results, 4);
% Results = mean(Results, 3);
Results = squeeze(Results(:, 3, :));  % select the most important
mean(Results,2)

