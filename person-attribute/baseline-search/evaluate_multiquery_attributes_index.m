% this script evaluate the different query types for attribute-driven person retrieval
model_types = {'svm_pcaelf', 'svm_caffenet_fc6', 'svm_caffenet_fc7', 'svm_resnet50_pool5', 'acn_caffenet', 'deepmar_caffenet', 'deepmar_resnet50'};
Iter = 5;

% the single attribute-based retrieval
mAP_single = zeros(7,54,5);
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

Index_Sorted = zeros(length(model_types), length(eff_index), Iter, 100);
Flag_Sorted = zeros(length(model_types), length(eff_index), Iter, 100);
% transform the decision score to probability
Sigma = 1;
mAP_multi = zeros(7, length(eff_index), 5);
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
            [~, tmp] = sort(prod_prob, 'descend');
            tmp_ = tmp(1:100);
            Index_Sorted(model_idx, idx, iter, 1:100) =  tmp_(:); 
            Flag_Sorted(model_idx, idx, iter, 1:100) =  gt_result_label(tmp_(:)); 
        end
    end
end
% mean(mean(mAP_single, 3),2)'
% mean(mean(mAP_multi,3),2)'
mAP_single = squeeze(mean(mAP_single, 2));
mAP = zeros(7, 4, Iter);
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


% write the first iter results into the txt files
load ./../../data/RAP_annotation/RAP_annotation.mat
% methods, mutiple, iter
for model_idx = 1:length(model_types)
    for att = 1:length(eff_index)
        for iter = 1:Iter
            name = sprintf('%s/%s_%03d_%d.txt', 'visualization', model_types{model_idx}, att, iter);
            fid = fopen(name, 'w+')
            for j=1:size(Index_Sorted, 4);
                fwrite(fid, sprintf('%s %d\n', RAP_annotation.name{RAP_annotation.partition_attribute{iter}.test_index(Index_Sorted(model_idx, att, iter, j))}, Flag_Sorted(model_idx, att, iter, j)));
            end
            fclose(fid)
        end
    end
end


