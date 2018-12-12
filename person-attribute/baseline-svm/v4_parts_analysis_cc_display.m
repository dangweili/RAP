% this script extract visualization
stage = 'trainval';
model_path = './models';
feature_type = 'resnet50_pool5';
feature_Norm = 'L2';
part_types = {'hs', 'ub', 'lb'};
part_idx = 3;
% only use the first partition for visualization
c_index = 1;
iter = 1;
svm_C = [1.0];
load ../static/LabelData_fusion_v1_v2.mat
selected_attribute = LabelData_fusion.selected_attribute;

if strcmp(feature_Norm, 'L2')
    if strcmp(stage, 'trainval')
        store_name = sprintf('%s/model_partcc_%s_trainval_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_type, feature_Norm, svm_C(c_index), iter);
    else
        store_name = sprintf('%s/model_partcc_%s_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_type, feature_Norm, svm_C(c_index), iter);
    end
else
    if strcmp(stage, 'trainval')
        store_name = sprintf('%s/model_partcc_%s_trainval_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_type, svm_C(c_index), iter);
    else
        store_name = sprintf('%s/model_partcc_%s_%s_%1.2f_part%d.mat', model_path, part_types{part_idx}, feature_type, svm_C(c_index), iter);
    end
end
load(store_name)

imgs_name = LabelData_fusion.name;

for label = 1:length(selected_attribute)
    label
    x = find(Label_pt{label} == 1);
    if Label_score{label}(x(1)) < 0
        Label_score{label} = -1 * Label_score{label};
    end
    % obtain the different viewpoints
        Label_index_j = Label_index{label};
        Label_score_j = Label_score{label};
        Label_pt_j = Label_pt{label};
        Label_gt_j = Label_gt{label};
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
        fid = fopen(sprintf('parts_rank_list/part_%s_att%d_pos.txt', part_types{part_idx}, label), 'w+');
        for k = 1:length(Label_index_j)
            fprintf(fid, sprintf('%s %f %d %d\n',imgs_name{Label_index_j(k)}, Label_score_j(k), Label_pt_j(k), Label_gt_j(k)));
        end
        fclose(fid);
end
