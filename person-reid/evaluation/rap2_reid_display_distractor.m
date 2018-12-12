% this script evaluate the mAP and CMC curve 
tic
clear all; close all;
rand('seed', 0);
addpath( genpath('./utils/') )

% load the annotation file
load ./../../data/RAP_annotation/RAP_annotation.mat

train_identity = RAP_annotation.partition_reid.train_identity;
test_identity = RAP_annotation.partition_reid.test_identity;

% test_net = 'CaffeNet'
test_net = 'ResNet50'
test_type = 1; % 1 will including -1 persons in test stage while 0 will not.
test_alg_type = 'IDE'; % att, IDE, or IDE-att
test_att = 1;
% feature_attribute_name = sprintf('./features/%s/%s_att.mat', test_net, test_alg_type);
test_reid = 1; % whether we should test reid or not
% feature_ide_name = sprintf('./features/%s/%s_ide.mat', test_net, test_alg_type);

metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 95% energy
save_mat = sprintf('results/%s/result_algtype%s_testtype%d.mat', test_net, test_alg_type, test_type);


feat_norm = 1;

% generate identity/day/cam/frame information
% only the first half of person are annotated with person identity
person_id = RAP_annotation.person_identity(1:41585);
person_cam = zeros(41585, 1);
person_day = zeros(41585, 1);
person_seq = zeros(41585, 1);
person_frame = zeros(41585, 1);
for i =1:41585
    image_name = RAP_annotation.name{i};
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
% save(save_mat, 'Result_att', 'mAP', 'CMC', 'test_index', 'query_index', 'IDX_eu', 'IDX_kissme', 'IDX_xqda');
load(save_mat)

% extract the sortted names for different algorithms
if test_reid && metric_euclidean
    parfor i = 1:length(query_index)
        i
        % image with the same identity but not same camera
        flag_sameid = person_id == person_id(query_index(i));
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam == 2);
        % image with same identity and same camera
        flag_samecam = person_cam == person_cam(query_index(i));
        junk_index = find(flag_sameid + flag_samecam == 2);
        index = IDX_eu(:, i);
        % store the image name for query and gallery images, test_type, index, query_index, query_id
        fid = fopen(sprintf('visualization/%s/%04d_%05d_%d_%s_eu_%04d_%03d.txt',test_net, person_id(query_index(i)), query_index(i), test_type, test_alg_type, i, length(good_index)), 'w+');
        fprintf(fid, '%s %f\n', RAP_annotation.name{query_index(i)}, ap_protocol_eu(i));
        for j=1:length(index)
            tmp = test_index(index(j)) == junk_index;
            if sum(tmp(:)) == 0
                tmp_ = test_index(index(j)) == good_index;
                if sum(tmp_(:)) == 0
                    fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 0);
                else
                    fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 1);
                end
            end
        end
        fclose(fid);
    end
end
