load('./../baseline-svm/results/Results_L2_V1_mm_test.mat')
feature_types = {'pcaelf', 'caffenet_fc6', 'caffenet_fc7', 'googlelenet_pool5', 'resnet50_pool5'}; 
svm_C = [0.01 0.10 1.0 10.0 100.0];
iter_Cnt = 5;

for feat_idx = 1:length(feature_types)
    for iter = 1:iter_Cnt
        save_name = sprintf('data/svm_%s_%d.mat', feature_types{feat_idx}, iter);
        pt_result = pt_Results{feat_idx, 3, iter}.pt_score;
        pt_label = pt_Results{feat_idx, 3, iter}.pt_label;
        % fix the svm score error
        tmp_idx = find(pt_label == 1);
        if pt_result(tmp_idx(1))< 0
            pt_result = pt_result*-1;
        end
        gt_result = pt_Results{feat_idx, 3, iter}.gt_label;
        gt_result(gt_result == 0) = -1;
        save(save_name, 'pt_result', 'gt_result');
    end
end
