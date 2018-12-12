function Result = rap_evaluation(p_label, g_label)
% This function compute the recognition results using instance and label based metric as descripted in 
% the paper "A Richly Annotated Dataset for Pedestrian Attribute Recognition.pdf"
% p_label and g_label are predicted label matrix and groundtruth label matrix.
% The coresponding value in p_label and g_label is 0 or 1. 
% 1 means the person has the binary attribute, while otherwise not.
% The dimension of p_label and g_label is N*L, N is the number of test examples and
% L is the number of labels
if sum(size(p_label) == size(g_label)) < 2
    Result = [];
    return;
end
[N, L] = size(g_label);
% the first metric, for each label
% accuracy = zeros(L, 1);
accuracy_pos = zeros(L, 1);
accuracy_neg = zeros(L, 1);
pos_index = g_label == 1;
neg_index = g_label == 0;
pos_cnt = sum(pos_index);
neg_cnt = sum(neg_index);
zero_flag = pos_cnt == 0;
pos_cnt(zero_flag) = 1;
if sum(zero_flag(:) > 0)
    fprintf('some attribute has zeros test images.\n');
end
zero_flag = neg_cnt == 0;
neg_cnt(zero_flag) = 1;
if sum(zero_flag(:) > 0)
    fprintf('some attribute has zeros test images.\n');
end

flag = p_label == g_label;
for iter = 1:L
    accuracy_pos(iter) = sum(flag(pos_index(:, iter), iter))/pos_cnt(iter);
    accuracy_neg(iter) = sum(flag(neg_index(:, iter), iter))/neg_cnt(iter);
end
accuracy_all = (accuracy_pos + accuracy_neg)/2;
Result.label_accuracy_all = accuracy_all;
Result.label_accuracy_pos = accuracy_pos;
Result.label_accuracy_neg = accuracy_neg;
Result.label_accuracy = mean(accuracy_all);
% instance evaluation
gt_pos_index = g_label == 1;
pt_pos_index = p_label == 1;
t = gt_pos_index + pt_pos_index;
tmp = sum(t>=2, 2);
tmp1 = sum(t>=1, 2);
tmp2 = sum(gt_pos_index>=1,2);
tmp3 = sum(pt_pos_index>=1,2);
f1 = tmp1 ~= 0;
f2 = tmp2 ~=0;
f3 = tmp3 ~= 0;
instance_accuracy = sum(tmp(f1)./tmp1(f1), 1)/sum(f1);
instance_recall = sum(tmp(f2)./tmp2(f2), 1)/sum(f2);
instance_precision = sum(tmp(f3)./tmp3(f3), 1)/sum(f3);
instance_F1 = 2*instance_precision * instance_recall /(instance_recall + instance_precision);
Result.instance_accuracy = instance_accuracy;
Result.instance_recall = instance_recall;
Result.instance_precision = instance_precision;
Result.instance_F1 = instance_F1;
end
