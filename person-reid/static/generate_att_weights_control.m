% this script generate train/val partation and generate test images 
rand('seed',0)

load ./../../data/RAP_annotation/RAP_annotation.mat

train_ids = RAP_annotation.partition_reid.train_identity;
person_identity = RAP_annotation.person_identity;
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);

labeldata(labeldata == 0) = -1;
labeldata(labeldata == 2) = 0;
% uniform the attribute representation
tmp_cnt = 0;
for i = 1:length(train_ids)
    idx = person_identity == train_ids(i);
    tmp = labeldata(idx, :);
    if abs(sum(tmp(:,1))) ~= sum(idx(:))
        tmp_cnt = tmp_cnt + 1;
    end
    % process the gender
    if sum(tmp(:,1) == 1) >= sum(tmp(:,1) == -1)
        tmp(:, 1) = 1;
    else
        tmp(:, 1) = -1;
    end
    labeldata(idx, :) = repmat(max(tmp), [sum(idx(:)), 1]);
end

% generate the train/val and trainval images for training the classification model
train_ids_cnt = length(train_ids);

fid_weight = fopen('classification-control/rap2_att_weight.txt', 'w+'); % consist labeled and unlabeled images
index = [];
for idx_pid = 1:train_ids_cnt
    idx_img = find(person_identity == train_ids(idx_pid));
    if idx_pid == 1
        index = idx_img;
    else
        index = [index' idx_img']';
    end
end

weight = sum(labeldata(index, :) == 1, 1)/length(index);

for i=1:length(weight)
    fprintf(fid_weight, sprintf('weight:%f\n', weight(i)));
end
fclose(fid_weight)
