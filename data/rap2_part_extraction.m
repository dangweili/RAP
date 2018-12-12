% this script generate the pedestrian part images
% if the pedestrian is occlued, then the orignal image is copyed into the corresponding file

load ./RAP_annotation/RAP_annotation.mat

% compute the occlusion images 

occlusion_type1 = sum(RAP_annotation.data(:, 113:116), 2) >=1;
occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
for i=1:3
    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
end
occlusion_type2 = occlusion_type2 >= 1;
occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
occlusion_index = find(occlusion_type);

% process the occlusion pedestrian, just copy
for occ_idx = 1:length(occlusion_index)
    file_name = RAP_annotation.name{occlusion_index(occ_idx)};
    src = strcat('./RAP_dataset/', file_name);
    dst = strcat('./RAP_dataset_hs/', file_name);
    copyfile(src, dst);
    dst = strcat('./RAP_dataset_ub/', file_name);
    copyfile(src, dst);
    dst = strcat('./RAP_dataset_lb/', file_name);
    copyfile(src, dst);
end
% process the clean pedestrians
clean_index = setdiff(1:length(RAP_annotation.name), occlusion_index);
dstPath = {'./RAP_dataset_hs/', ...
    './RAP_dataset_ub/', ...
    './RAP_dataset_lb/'};

pos0 = RAP_annotation.data(clean_index, 121:124);
pos1 = RAP_annotation.data(clean_index, 125:128);
pos2 = RAP_annotation.data(clean_index, 129:132);
pos3 = RAP_annotation.data(clean_index, 133:136);
% transform the global position to the local position
pos1(:, 1:2) = pos1(:, 1:2) - pos0(:, 1:2) + 1;
pos1(:, 3:4) = pos1(:, 3:4) + pos1(:, 1:2) - 1;
pos2(:, 1:2) = pos2(:, 1:2) - pos0(:, 1:2) + 1;
pos2(:, 3:4) = pos2(:, 3:4) + pos2(:, 1:2) - 1;
pos3(:, 1:2) = pos3(:, 1:2) - pos0(:, 1:2) + 1;
pos3(:, 3:4) = pos3(:, 3:4) + pos3(:, 1:2) - 1;

pos = [pos1 pos2 pos3];
cnt = [1];
error_cnt = 0;
error_index = [];
for clean_idx = 1:length(clean_index)
    clean_idx
    file_name = RAP_annotation.name{clean_index(clean_idx)};
    src = strcat('./RAP_dataset/', file_name);
    img = imread(src);
    % extract three parts
    for iter = 1:3
        dst = strcat(dstPath{iter}, file_name);
        % extract the part image
        pos_ = pos(clean_idx, 4*(iter-1)+1: 4*iter);
        x_start = max([pos_(1) 1]);
        y_start = max([pos_(2) 1]);
        x_end = min([pos_(3) size(img, 2)]);
        y_end = min([pos_(4) size(img, 1)]);
        if pos_(3) > x_end | pos_(4) > y_end | pos_(1) ==0 | pos_(2) == 0
            cnt = [cnt clean_index(clean_idx)];
        end
        img_part_ = img(y_start:y_end, x_start:x_end, :);
        if prod(size(img_part_)) == 0
            error_cnt = error_cnt + 1;
            error_index = [error_index clean_index(clean_idx)];
        end
        imwrite(img_part_, dst);
    end
end

cnt

error_cnt

error_index
