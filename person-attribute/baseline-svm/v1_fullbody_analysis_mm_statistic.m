% static the recognition results
clear all;

feature_types = {'pcaelf', 'caffenet_fc6', 'caffenet_fc7', 'googlelenet_pool5', 'resnet50_pool5'};

svm_C = [0.01 0.10 1.0 10.0 100.0];

iter_Cnt = 5;

load('results/Results_L2_val.mat') % val: train->val test:trainval->test
Results_static = zeros(length(feature_types), length(svm_C), 7, iter_Cnt);

for feat_idx = 1:length(feature_types)
    for c_index = 1:length(svm_C)
        for iter = 1:iter_Cnt
            Results_static(feat_idx, c_index, 1, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_pos(:));
            Results_static(feat_idx, c_index, 2, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_neg(:));
            Results_static(feat_idx, c_index, 3, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_all(:));
            Results_static(feat_idx, c_index, 4, iter) = mean(Results{feat_idx, c_index, iter}.instance_accuracy);
            Results_static(feat_idx, c_index, 5, iter) = mean(Results{feat_idx, c_index, iter}.instance_recall);
            Results_static(feat_idx, c_index, 6, iter) = mean(Results{feat_idx, c_index, iter}.instance_precision);
            Results_static(feat_idx, c_index, 7, iter) = mean(Results{feat_idx, c_index, iter}.instance_F1);
        end
    end
end

Results_static_val = mean(Results_static, 4);

load('results/Results_L2_test.mat') % val: train->val test:trainval->test
Results_static = zeros(length(feature_types), length(svm_C), 7, iter_Cnt);

for feat_idx = 1:length(feature_types)
    for c_index = 1:length(svm_C)
        for iter = 1:iter_Cnt
            Results_static(feat_idx, c_index, 1, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_pos(:));
            Results_static(feat_idx, c_index, 2, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_neg(:));
            Results_static(feat_idx, c_index, 3, iter) = mean(Results{feat_idx, c_index, iter}.label_accuracy_all(:));
            Results_static(feat_idx, c_index, 4, iter) = mean(Results{feat_idx, c_index, iter}.instance_accuracy);
            Results_static(feat_idx, c_index, 5, iter) = mean(Results{feat_idx, c_index, iter}.instance_recall);
            Results_static(feat_idx, c_index, 6, iter) = mean(Results{feat_idx, c_index, iter}.instance_precision);
            Results_static(feat_idx, c_index, 7, iter) = mean(Results{feat_idx, c_index, iter}.instance_F1);
        end
    end
end

Results_static_test = mean(Results_static, 4);

% display the best mean accuracy for five types of feature
Results_static_val(:, :, 3)

Results_static_test(:, :, 3)


% additional result for single attribute's mean Accuracy
best_C = 3;
Results_single = zeros(length(feature_types), 3, iter_Cnt, 54);
for feat_idx = 1:length(feature_types)
    for iter = 1:iter_Cnt
        Results_single(feat_idx, 1, iter, :) = Results{feat_idx, best_C, iter}.label_accuracy_pos(:);
        Results_single(feat_idx, 2, iter, :) = Results{feat_idx, best_C, iter}.label_accuracy_neg(:);
        Results_single(feat_idx, 3, iter, :) = Results{feat_idx, best_C, iter}.label_accuracy_all(:);
    end
end
Results_single = squeeze(mean(Results_single, 3));
Results_single_metric2 = squeeze(Results_static_test(:, best_C, :));
