% this script evaluate the mAP and CMC curve 
tic
clear all; close all;
rand('seed', 0);
addpath( genpath('./utils/') )

% load the annotation file
load ../static/LabelData_fusion_v1_v2.mat

train_identity = LabelData_fusion.partition_reid.train_identity;
test_identity = LabelData_fusion.partition_reid.test_identity;

test_net = 'CaffeNet'
% test_net = 'ResNet50'
test_type = 0;  % 1 will including -1 persons in test stage while 0 will not.
test_alg_type = 'IDE'; % att, IDE, or IDE-att
test_att = 1; % useless, just set as 1
feature_attribute_name = sprintf('./features/%s/%s_att.mat', test_net, test_alg_type);
test_reid = 1; % whether we should test reid or not
feature_ide_name = sprintf('./features/%s/%s_ide.mat', test_net, test_alg_type);

metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 95% energy
save_mat = sprintf('results/%s/result_algtype%s_testtype%d.mat', test_net, test_alg_type, test_type);


feat_norm = 1;

% generate identity/day/cam/frame information
% only the first half of person are annotated with person identity
person_id = LabelData_fusion.person_identity(1:41585);
person_cam = zeros(41585, 1);
person_day = zeros(41585, 1);
person_seq = zeros(41585, 1);
person_frame = zeros(41585, 1);
for i =1:41585
    image_name = LabelData_fusion.name{i};
    cam_ = str2num(image_name(4:5));
    day_ = str2num([image_name([7:10 12:13 15:16])]);
    seq_ = str2num(image_name(26:31));
    pos_frame = strfind(image_name, 'frame');
    pos_line = strfind(image_name, 'line');
    frame_ = str2num(image_name(pos_frame+5:pos_line-2));
    person_cam(i) = cam_;
    person_day(i) = day_;
    person_seq(i) = seq_;
    person_frame(i) = frame_;
end
% generate the query samples for testing stage
% obtain the test persons index 
test_index = []; % not including -1
for i = 1:length(test_identity)
    idx = find(person_id == test_identity(i));
    test_index = [test_index' idx']';
end
if test_type == 1
    idx = find(person_id == -1);
    test_index = [test_index' idx']';
end
% obtain the query images in the test stage
% for each person, sample one image in one camera at one day for query
% so one person may have mutiple queries in one camera among different days
query_index = [];
for i = 1:length(test_identity)
    idx = find(person_id == test_identity(i));
    cam = person_cam(idx);
    day = person_day(idx);
    seq = person_seq(idx);
    frame = person_frame(idx);
    % random sample
    u_day = unique(day);
    u_cam = unique(cam);
    for j = 1:length(u_day)
        day_flag = day == u_day(j);
        for k = 1:length(u_cam)
            cam_flag = cam == u_cam(k);
            idx_ = idx(day_flag + cam_flag == 2);
            if length(idx_) > 0
                tmp = idx_(randperm(length(idx_), 1));
                query_index = [query_index tmp];
            end
        end
    end
end

fid = fopen('rap2_test_image_name.txt', 'w+')
for i = 1: length(query_index)
    fprintf(fid, '%s\n', LabelData_fusion.name{query_index(i)});
end
fclose(fid);
length(test_index)
