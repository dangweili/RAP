% this script generate train/val partation and generate test images 
rand('seed',0)

load ./../../data/RAP_annotation/RAP_annotation.mat

images_name = RAP_annotation.name;
person_identity = RAP_annotation.person_identity;
Images_cnt = 41585; 
selected_attribute = RAP_annotation.selected_attribute;
labeldata = RAP_annotation.data(:, selected_attribute);
labeldata(labeldata == 0) = -1;
labeldata(labeldata == 2) = 0;

images_root = '.';


fid = fopen('classification/rap2_ide_att_trainvaltest.txt', 'w+');

trainvaltest_set = {};
for idx = 1:Images_cnt
    tmp = sprintf('%s/%s', images_root, images_name{idx});
    tmp = [tmp sprintf(' %d', person_identity(idx) )];
    for i=1:length(selected_attribute)
        tmp = [tmp sprintf(' %d', labeldata(idx, i))];
    end
    tmp = [tmp '\n'];
    trainvaltest_set{idx} = tmp;
end


for i=1:length(trainvaltest_set)
    fprintf(fid, trainvaltest_set{i});
end

fclose(fid);

