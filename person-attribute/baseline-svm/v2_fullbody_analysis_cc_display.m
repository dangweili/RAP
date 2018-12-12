% this script extract visualization
stage = 'trainval';
model_path = './models';
feature_type = 'resnet50_pool5';
feature_Norm = 'L2';
% only use the first partition for visualization
c_index = 1;
iter = 1;
svm_C = [1.0];
load ../../data/RAP_annotation/RAP_annotation.mat
selected_attribute = RAP_annotation.selected_attribute;

if strcmp(feature_Norm, 'L2')
    if strcmp(stage, 'trainval')
        store_name = sprintf('%s/model_bodycc_trainval_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
    else
        store_name = sprintf('%s/model_bodycc_%s_%s_%1.2f_part%d.mat', model_path, feature_type, feature_Norm, svm_C(c_index), iter);
    end
else
    if strcmp(stage, 'trainval')
        store_name = sprintf('%s/model_bodycc_trainval_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
    else
        store_name = sprintf('%s/model_bodycc_%s_%1.2f_part%d.mat', model_path, feature_type, svm_C(c_index), iter);
    end
end
load(store_name)

view = RAP_annotation.data(:, 112);
imgs_name = RAP_annotation.name;

for label = 1:length(selected_attribute)
    x = find(Label_pt{label} == 1);
    if Label_score{label}(x(1)) < 0
        Label_score{label} = -1 * Label_score{label};
    end
    % obtain the different viewpoints
    for j=1:4
        idx = (view(Label_index{label}) == j);
        Label_index_j = Label_index{label}(idx);
        Label_score_j = Label_score{label}(idx);
        Label_pt_j = Label_pt{label}(idx);
        Label_gt_j = Label_gt{label}(idx);
        % true positive and false negative samples
        flag = Label_gt_j == 1;
        Label_gt_j = Label_gt_j(flag);
        Label_pt_j = Label_pt_j(flag);
        Label_score_j = Label_score_j(flag);
        Label_index_j = Label_index_j(flag);
        % sort and get the list
        [~, idx_s] = sort(Label_score_j, 'descend');
        Label_gt_j = Label_gt_j(idx_s);
        Label_pt_j = Label_pt_j(idx_s);
        Label_score_j = Label_score_j(idx_s);
        Label_index_j = Label_index_j(idx_s);
        % record and list into files
        fid = fopen(sprintf('view_rank_list/view%d_att%d_pos.txt', j, label), 'w+');
        for k = 1:length(Label_index_j)
            fprintf(fid, sprintf('%s %f %d %d\n',imgs_name{Label_index_j(k)}, Label_score_j(k), Label_pt_j(k), Label_gt_j(k)));
        end
        fclose(fid);
    end
end
