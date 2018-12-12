% this script generate train/val partation and generate test images 
rand('seed',0)

load ../../data/RAP_annotation/RAP_annotation.mat

train_ids = RAP_annotation.partition_reid.train_identity;
person_identity = RAP_annotation.person_identity;
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);

% generate the train/val and trainval images for training the classification model
train_ids_cnt = length(train_ids);

fid_weight = fopen('classification/rap2_att_weight.txt', 'w+'); % consist labeled and unlabeled images
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
