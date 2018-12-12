% this script generates train/val/test partation for pedestrian attribute recognition in rap2
load RAP_annotation.mat

% each row owns mutiple labels with only one image
idx_select = RAP_annotation.selected_attribute;
data = RAP_annotation.data(:, idx_select);
imgs_name = RAP_annotation.name;
% preprocess the annotation from 0,1,2 to -1,0,1
data(data == 0) = -1;
data(data == 2) = 0;

for i=1:length(RAP_annotation.partition_attribute)
    for batt_idx = 1:54
        % process the training data
        fid_name = sprintf('images-list-binary/rap2_train_%d_att%d.txt', i, batt_idx);
        fid = fopen(fid_name, 'w+');
        index = RAP_annotation.partition_attribute{i}.train_index;
        for j=1:length(index)
            fprintf(fid, '%s %d\n', [imgs_name{index(j)}(1:end-3) 'bmp'], data(index(j), batt_idx));
        end
        fclose(fid);
        % process the validation data
        fid_name = sprintf('images-list-binary/rap2_val_%d_att%d.txt', i, batt_idx);
        fid = fopen(fid_name, 'w+');
        index = RAP_annotation.partition_attribute{i}.val_index;
        for j=1:length(index)
            fprintf(fid, '%s %d\n', [imgs_name{index(j)}(1:end-3) 'bmp'], data(index(j), batt_idx));
        end
        fclose(fid) ;  
        % process the test data
        fid_name = sprintf('images-list-binary/rap2_test_%d_att%d.txt', i, batt_idx);
        fid = fopen(fid_name, 'w+');
        index = RAP_annotation.partition_attribute{i}.test_index;
        for j=1:length(index)
            fprintf(fid, '%s %d\n', [imgs_name{index(j)}(1:end-3) 'bmp'], data(index(j), batt_idx));
        end
        fclose(fid) ;     
        % process the train_val data
        fid_name = sprintf('images-list-binary/rap2_trainval_%d_att%d.txt', i, batt_idx);
        fid = fopen(fid_name, 'w+');
        index = [RAP_annotation.partition_attribute{i}.train_index RAP_annotation.partition_attribute{i}.val_index];
        for j=1:length(index)
            fprintf(fid, '%s %d\n', [imgs_name{index(j)}(1:end-3) 'bmp'], data(index(j), batt_idx));
        end
        fclose(fid) ; 
    end
end

