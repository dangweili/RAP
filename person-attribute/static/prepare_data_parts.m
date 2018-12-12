% this script generates train/val/test partation for pedestrian attribute recognition in rap2
load ../../data/RAP_annotation/RAP_annotation.mat 

% each row owns mutiple labels with only one image
idx_select = RAP_annotation.selected_attribute;
data = RAP_annotation.data(:, idx_select);
imgs_name = RAP_annotation.name;
% preprocess the annotation from 0,1,2 to -1,0,1
data(data == 0) = -1;
data(data == 2) = 0;
% obtain the occlusion index for futher analysis
occlusion_type1 = sum(RAP_annotation.data(:, 113:116),2) >=1;
occlusion_type2 = zeros(length(RAP_annotation.data(:,1)), 1);
for i=1:3
    occlusion_type2 = occlusion_type2 + (sum(RAP_annotation.data(:, 120+4*i+1:120+4*i+4), 2) == 0);
    end
occlusion_type2 = occlusion_type2 >= 1;
occlusion_type = (occlusion_type1 + occlusion_type2) >= 1;
occlusion_index = find(occlusion_type);

for i=1:length(RAP_annotation.partition_attribute)
    % process the training data
    fid_name = sprintf('images-list-parts/rap2_train_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.train_index;
    index = setdiff(index, occlusion_index);
    index = index(randperm(length(index)));
    for j=1:length(index)
        % fprintf(fid, 'data/RAP_dataset/%s %d ', imgs_name{index(j)}, data(index(j), 1));
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    % process the validation data
    fid_name = sprintf('images-list-parts/rap2_val_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.val_index;
    index = setdiff(index, occlusion_index);
    index = index(randperm(length(index)));
    for j=1:length(index)
        % fprintf(fid, 'data/RAP_dataset/%s %d ', imgs_name{index(j)}, data(index(j), 1));
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ;  
    % process the test data
    fid_name = sprintf('images-list-parts/rap2_test_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.test_index;
    index = setdiff(index, occlusion_index);
    index = index(randperm(length(index)));
    for j=1:length(index)
        % fprintf(fid, 'data/RAP_dataset/%s %d ', imgs_name{index(j)}, data(index(j), 1));
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ;     
    % process the train_val data
    fid_name = sprintf('images-list-parts/rap2_trainval_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = [RAP_annotation.partition_attribute{i}.train_index RAP_annotation.partition_attribute{i}.val_index];
    index = setdiff(index, occlusion_index);
    index = index(randperm(length(index)));
    for j=1:length(index)
        % fprintf(fid, 'data/RAP_dataset/%s %d ', imgs_name{index(j)}, data(index(j), 1));
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ; 
    % obtain the weight of positive examples
    weight = sum(data(index, :) == 1)/length(index);
    fid_name = sprintf('images-list-parts/rap2_trainval_weight_%d.txt' , i)
    fid = fopen(fid_name, 'w+');
    for j = 1:length(idx_select)
        fprintf(fid, '%f\n', weight(j));
    end
    fclose(fid);
end

