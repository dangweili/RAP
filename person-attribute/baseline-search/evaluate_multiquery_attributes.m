% this script evaluate the different query types for attribute-driven person retrieval
% model_types = {'svm_pcaelf', 'svm_caffenet_fc6', 'svm_caffenet_fc7', 'svm_resnet50_pool5', 'acn_caffenet', 'deepmar_caffenet', 'deepmar_resnet50'};
model_types = {'svm_pcaelf', 'svm_caffenet_fc6', 'svm_caffenet_fc7', 'svm_resnet50_pool5', 'sacn_caffenet', 'acn_caffenet', 'sdeepmar_caffenet', 'deepmar_caffenet', 'deepmar_resnet50'};
Iter = 5;

% the single attribute-based retrieval
mAP_single = zeros(length(model_types),54,5);
for model_idx = 1:length(model_types)
    for iter = 1:Iter
        % pt_result, pt_label, gt_result
        load(sprintf('data/%s_%d.mat', model_types{model_idx}, iter)); 
        for label = 1:54
            mAP_single(model_idx, label, iter) = compute_average_precision(gt_result(:, label), pt_result(:, label));
        end
    end
end
% mean(mAP_single, 3)
% std(mAP_single, 0, 3)

% the two-attribute-based retrieval
load multiatt_query.mat % MultiAtt_test, MultiAtt_test_cnt

eff_index = find(cell2mat(MultiAtt_test_cnt)>100);
% transform the decision score to probability
Sigma = 1;
mAP_multi = zeros(length(model_types), length(eff_index), 5);
query_att_cnt = zeros(length(eff_index),1);
for model_idx = 1:length(model_types)
    for iter = 1:Iter
        % pt_result, pt_label, gt_result
        load(sprintf('data/%s_%d.mat', model_types{model_idx}, iter));
        pt_result_femal = 1./(1+exp(-Sigma*pt_result));
        gt_result_male = gt_result;
        gt_result_male(:,1) = gt_result_male(:,1)*-1;
        pt_result_male = pt_result_femal;
        pt_result_male(:,1) = 1-pt_result_male(:,1);
        % compute the ap overall the results
        for idx = 1:length(eff_index)
            att_index = MultiAtt_test{eff_index(idx)};
            query_att_cnt(idx) = length(att_index);
            if att_index(1) == 1 % female
                prod_prob = prod(pt_result_femal(:, att_index), 2);
                gt_result_label = sum(gt_result(:, att_index), 2)==length(att_index);
                if (sum(gt_result_label) ~= MultiAtt_test_cnt{eff_index(idx)}) && (iter == 1)
                    fprintf(sprintf('num wrong in iter:%d  idx:%d\n', iter, idx));
                end
            end
            if att_index(1) == 0 % male
                prod_prob = pt_result_male(:,1) .* prod(pt_result_male(:, att_index(2:end)), 2);
                gt_result_label_1 = sum(gt_result_male(:, att_index(2:end)),2) == length(att_index)-1;
                gt_result_label_2 = gt_result_male(:,1) == 1;
                gt_result_label = gt_result_label_1 + gt_result_label_2 == 2;
                if (sum(gt_result_label) ~= MultiAtt_test_cnt{eff_index(idx)}) && (iter == 1)
                    fprintf(sprintf('num wrong in iter:%d  idx:%d\n', iter, idx));
                end
            end
            mAP_multi(model_idx, idx, iter) = compute_average_precision(gt_result_label, prod_prob);
        end
    end
end
% mean(mean(mAP_single, 3),2)'
% mean(mean(mAP_multi,3),2)'
mAP_single = squeeze(mean(mAP_single, 2));
mAP = zeros(length(model_types), 4, Iter);
for model_idx = 1:length(model_types)
    for att_cnt = 1:4
        for iter=1:5
            if att_cnt == 1
                mAP(model_idx, att_cnt, iter) = mAP_single(model_idx, iter);
            end
            if att_cnt > 1
                idx_flag = query_att_cnt == att_cnt;
                tmp = mAP_multi(model_idx, idx_flag, iter);
                mAP(model_idx, att_cnt, iter) = mean(tmp(:));
            end
        end
    end
end
% compute the average mAP across iters
mAP_ = mean(mAP, 3);
mAP_*100

% store different algorithms' ap value for 150 attributes
name = sprintf('ap_results/ap_result.txt');
fid = fopen(name, 'w+');
for i = 1:length(model_types)
    fprintf(fid, '%s ', model_types{i});
end
fprintf(fid, '\n')
for j=1:150
    for i=1:length(model_types)
        % the first result and the average result
        tmp = reshape(mAP_multi(i, j, :), [1,5]);
        fprintf(fid, '%02.2f;%02.2f\t', tmp(1), mean(tmp(:)));
    end
    fprintf(fid, sprintf('  %d\n', MultiAtt_test_cnt{eff_index(j)}));
end
fclose(fid)
