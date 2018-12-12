% this script generates train/val/test partation for pedestrian attribute recognition in rap2
load ../../data/RAP_annotation/RAP_annotation.mat

% each row owns mutiple labels with only one image
idx_select = RAP_annotation.selected_attribute;
data = RAP_annotation.data(:, idx_select);
imgs_name = RAP_annotation.name;
% preprocess the annotation from 0,1,2 to -1,0,1
data(data == 0) = -1;
data(data == 2) = 0;

for i=1:length(RAP_annotation.partition_attribute)
    % process the training data
    fid_name = sprintf('images-list/rap2_train_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.train_index;
    for j=1:length(index)
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    % process the validation data
    fid_name = sprintf('images-list/rap2_val_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.val_index;
    for j=1:length(index)
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ;  
    % process the test data
    fid_name = sprintf('images-list/rap2_test_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = RAP_annotation.partition_attribute{i}.test_index;
    for j=1:length(index)
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ;     
    % process the train_val data
    fid_name = sprintf('images-list/rap2_trainval_%d.txt', i);
    fid = fopen(fid_name, 'w+');
    index = [RAP_annotation.partition_attribute{i}.train_index RAP_annotation.partition_attribute{i}.val_index];
    for j=1:length(index)
        fprintf(fid, '%s %d ', [imgs_name{index(j)}(1:end-3) 'png'], data(index(j), 1));
        for k=2:length(idx_select)
            fprintf(fid, '%d ', data(index(j), k));
        end
        fprintf(fid, '\n');
    end
    fclose(fid) ; 
    % obtain the weight of positive examples
    weight = sum(data(index, :) == 1)/length(index);
    fid_name = sprintf('images-list/rap2_trainval_weight_%d.txt' , i)
    fid = fopen(fid_name, 'w+');
    for j = 1:length(idx_select)
        fprintf(fid, '%f\n', weight(j));
    end
    fclose(fid);
end

