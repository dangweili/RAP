% this script evaluate the mAP and CMC curve 
tic
clear all; close all;
rand('seed', 0);
addpath( genpath('./utils/') )

% load the annotation file
load ./../../data/RAP_annotation/RAP_annotation.mat

train_identity = RAP_annotation.partition_reid.train_identity;
test_identity = RAP_annotation.partition_reid.test_identity;

test_net = 'CaffeNet'
% test_net = 'ResNet50'
test_type = 1; % 1 will including -1 persons in test stage while 0 will not.
test_alg_type = 'IDE-att'; % att, IDE, or IDE-att
test_att = 1;
feature_attribute_name = sprintf('./features/%s/%s_att.mat', test_net, test_alg_type);
test_reid = 1; % whether we should test reid or not
feature_ide_name = sprintf('./features/%s/%s_ide.mat', test_net, test_alg_type);

metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 95% energy
% save_mat = sprintf('results_control/%s/result_algtype%s_testtype%d.mat', test_net, test_alg_type, test_type);

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

% obtain the train persons index
train_index = [];
for i = 1:length(train_identity)
    idx = find(person_id == train_identity(i));
    train_index = [train_index' idx']';
end

fprintf('preprocess is ok now.\n')

% load feature representation for evaluatation
% and split the feature into train and test
if test_att 
    load(feature_attribute_name)
    feat_att = images_feat;
    feat_att_train = feat_att(train_index, :);
    feat_att_test = feat_att(test_index, :);
end
if test_reid
    load(feature_ide_name)
    feat_ide = images_feat;
    feat_ide_train = feat_ide(train_index, :);
    feat_ide_test = feat_ide(test_index, :);
    feat_ide_query = feat_ide(query_index, :); % N*D
end

% test the attribute recognition results
if test_att
    selected_attribute = RAP_annotation.selected_attribute;
    person_attribute_gt = RAP_annotation.data(1:41585, selected_attribute);
    person_attribute_gt_test = person_attribute_gt(test_index, :);
    person_attribute_pt_test = feat_att_test;
    person_attribute_pt_test(person_attribute_pt_test>=0) = 1;
    person_attribute_pt_test(person_attribute_pt_test<0) = 0;
    Result_att = rap_evaluation(person_attribute_pt_test, person_attribute_gt_test);
    fprintf('attribute recognition is ok now.\n')
end

% normalize each row, N*D 
if test_reid && feat_norm
    feat_ide_train = feat_ide_train./repmat(sqrt(sum(feat_ide_train.^2, 2)), [1 size(feat_ide_train, 2)]);
    feat_ide_test = feat_ide_test./repmat(sqrt(sum(feat_ide_test.^2, 2)), [1 size(feat_ide_test, 2)]);
    feat_ide_query = feat_ide_query./repmat(sqrt(sum(feat_ide_query.^2, 2)), [1 size(feat_ide_query, 2)]);
end
TK = 500;
% extract the sortted names for different algorithms
if test_reid && metric_euclidean
    fprintf('preprocessing for person re-id based on Euclidean metric starts now.\n');
    % compute the squared Euclidean distance
    dist_eu = sqdist(feat_ide_test', feat_ide_query');
    [nGalleryPerson, nProbePerson] = size(dist_eu);
    [~, IDX_eu] = sort(dist_eu, 1, 'ascend');
    
    
    cmc_protocol_eu_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_eu_sameday = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_eu_crossday = zeros(nProbePerson, 1);
    ap_protocol_eu_sameday = zeros(nProbePerson, 1);
    eu_crossday_cnt = zeros(nProbePerson, 1);
    eu_sameday_cnt = zeros(nProbePerson, 1);
    fprintf('person re-id for each query based on Euclidean metric starts now.\n');
    parfor i = 1:length(query_index)
        [1 i]
        % image with the same identity but not same camera
        flag_sameid = person_id == person_id(query_index(i));
        alldays = unique(person_day(flag_sameid));
        index = IDX_eu(:, i);
        % for each query, test the same day ap
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        flag_samecam = 1 - flag_nsamecam;
        flag_sameday = person_day == person_day(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam + flag_sameday == 3);
        junk_index = find(flag_sameid+flag_samecam+flag_sameday == 3 | flag_sameday == 0);
        if length(good_index) >= 1
            query_same_day_flag = 1;
        else
            query_same_day_flag = 0;
        end
        % process the cross day test
        local_cnt = 0;
        cmc_protocol_tmp = zeros(1, nGalleryPerson);
        ap_protocol_tmp = 0;
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue
            end
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index) >= 1
                local_cnt = local_cnt + 1;
            end
        end
        if query_same_day_flag ==0 || local_cnt==0
            continue
        end
        % compute the ap and output the list
        % for each query, test the same day ap
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        flag_samecam = 1 - flag_nsamecam;
        flag_sameday = person_day == person_day(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam + flag_sameday == 3);
        junk_index = find(flag_sameid+flag_samecam+flag_sameday == 3 | flag_sameday == 0);
        if length(good_index) >= 1
            [ap_protocol_eu_sameday(i), cmc_protocol_eu_sameday(i, :)] = compute_AP(good_index, junk_index, test_index(index));
            eu_sameday_cnt(i) = 1;
        else
            ap_protocol_eu_sameday(i) = 0;
            cmc_protocol_eu_sameday(i, :) = zeros(1, nGalleryPerson);
        end
        % store the image name for query and gallery images, test_type, index, query_index, query_id
        fid = fopen(sprintf('visualization_control/%s/%04d_%05d_%d_%s_eu_%04d_%03d_s.txt',test_net, person_id(query_index(i)), query_index(i), test_type, test_alg_type, i, length(good_index)), 'w+');
        fprintf(fid, '%s %f\n', RAP_annotation.name{query_index(i)}, ap_protocol_eu_sameday(i));
        eff_store = 0;
        for j=1:length(index)
            tmp = test_index(index(j)) == junk_index;
            if sum(tmp(:)) == 0
                eff_store = eff_store + 1;
                tmp_ = test_index(index(j)) == good_index;
                if sum(tmp_(:)) == 0
                    fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 0);
                else
                    fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 1);
                end
            end
            if eff_store >= TK
                break;
            end
        end
        fclose(fid);
        % process the cross-day person retrieval 
        local_cnt = 0;
        cmc_protocol_tmp = zeros(1, nGalleryPerson);
        ap_protocol_tmp = 0;
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue
            end
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index) >= 1
                local_cnt = local_cnt + 1;
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index));
                cmc_protocol_tmp = cmc_protocol_tmp + tmp_cmc;
                ap_protocol_tmp = ap_protocol_tmp + tmp_ap;
                fid = fopen(sprintf('visualization_control/%s/%04d_%05d_%d_%s_eu_%04d_%03d_x%d.txt',test_net, person_id(query_index(i)), query_index(i), test_type, test_alg_type, i, length(good_index), local_cnt), 'w+');
                fprintf(fid, '%s %f\n', RAP_annotation.name{query_index(i)}, tmp_ap);
                eff_store = 0;
                for j=1:length(index)
                    tmp = test_index(index(j)) == junk_index;
                    if sum(tmp(:)) == 0
                        eff_store = eff_store + 1;
                        tmp_ = test_index(index(j)) == good_index;
                        if sum(tmp_(:)) == 0
                            fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 0);
                        else
                            fprintf(fid, '%s %d\n', RAP_annotation.name{test_index(index(j))}, 1);
                        end
                    end
                    if eff_store >= TK
                        break;
                    end
                end
                fclose(fid);
            end
        end
        if local_cnt >= 1
            ap_protocol_eu_crossday(i) = ap_protocol_tmp/local_cnt;
            cmc_protocol_eu_crossday(i, :) = cmc_protocol_tmp/local_cnt;
            eu_crossday_cnt(i) = local_cnt;
        else
            ap_protocol_eu_crossday(i) = 0;
            cmc_protocol_eu_crossday(i, :) = cmc_protocol_tmp;
            eu_crossday_cnt(i) = 0;
        end
    end
    % only summary the samples who can do same-day and cross-day retrieval in the same time.
    useless_flag = eu_crossday_cnt == 0 | eu_sameday_cnt == 0;
    eu_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_eu_sameday.*eu_sameday_cnt;
    mAP_eu_sameday = sum(tmp(:))/sum(eu_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_eu_sameday, eu_sameday_cnt);
    CMC_eu_sameday = sum(tmp)/sum(eu_sameday_cnt);

    eu_crossday_cnt(useless_flag) = 0;
    tmp = ap_protocol_eu_crossday.*eu_crossday_cnt;
    mAP_eu_crossday = sum(tmp)/sum(eu_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_eu_crossday, eu_crossday_cnt);
    CMC_eu_crossday = sum(tmp)/sum(eu_crossday_cnt);
    fprintf('person re-id based on Euclidean distance is ok now.\n')
end
mAP_eu_sameday
mAP_eu_crossday
CMC_eu_sameday(:, 1:5)
CMC_eu_crossday(:, 1:5)
