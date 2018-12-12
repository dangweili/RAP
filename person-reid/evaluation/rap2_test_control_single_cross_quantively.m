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
test_type = 1; % 1 will including -1 persons in test stage while 0 will not. In test_control, this should always be 1 to add the distractors in each day.
test_alg_type = 'IDE-att'; % att, IDE, or IDE-att
test_att = 1;
threshold = [0.5, 0.25, 0.5, 0.25];
feature_attribute_name = sprintf('./features/%s/%s_att.mat', test_net, test_alg_type);
test_reid = 1; % whether we should test reid or not
feature_ide_name = sprintf('./features/%s/%s_ide.mat', test_net, test_alg_type);

metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 80% energy
save_mat = sprintf('results_control/%s/result_algtype%s_testtype%d_quantively_%.2f_%.2f_%.2f_%.2f.mat', test_net, test_alg_type, test_type, threshold(1), threshold(2), threshold(3), threshold(4));

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

% compute the cross day person identity appearance change
unique_days = unique(person_day);
cross_day_person_identity = [];
cross_day_person_identity_appearance_diff_matrix = cell(0);
cross_cnt = 1;
upper_cloth_type_index = [14,15,22,23,24,25,26,27,28,29,30];
upper_cloth_color_index = [32,33,34,35,36,37,38,39,40,41,42,43,44];
lower_cloth_type_index = [46,47,48,49,50,51,52,53];
lower_cloth_color_index = [54,55,56,57,58,59,60,61,62,63,64,65,66];
app_status_array = [];
cnt_threshold = zeros(1,4);
for i = 1:length(test_identity)
    idx = find(person_id == test_identity(i));
    unique_person_day = unique(person_day(idx));
    L = length(unique_person_day);
    if L <= 1
        continue;
    end
    cross_day_person_identity(cross_cnt) = test_identity(i);
    
    % summary of appearance change
    diff_matrix_tmp = ones(length(unique_days), length(unique_days))*-1;
    for j=1:L
        for k=1:j
            if j == k
                person_day_idx = find(unique_days == unique_person_day(j));
                diff_matrix_tmp(person_day_idx, person_day_idx) = 0;
                continue;
            end
            % process the next
            person_day_idx_j = find(unique_days == unique_person_day(j));
            person_day_idx_k = find(unique_days == unique_person_day(k));
            person_idx_j = idx(person_day(idx) == unique_person_day(j));
            person_idx_k = idx(person_day(idx) == unique_person_day(k));
            % get the corresponding attributes
            up_att_j = max(RAP_annotation.data(person_idx_j, upper_cloth_type_index), [], 1);
            up_col_j = max(RAP_annotation.data(person_idx_j, upper_cloth_color_index), [], 1);  
            low_att_j = max(RAP_annotation.data(person_idx_j, lower_cloth_type_index), [], 1);
            low_col_j = max(RAP_annotation.data(person_idx_j, lower_cloth_color_index), [], 1);
             
            up_att_k = max(RAP_annotation.data(person_idx_k, upper_cloth_type_index), [], 1);
            up_col_k = max(RAP_annotation.data(person_idx_k, upper_cloth_color_index), [], 1);  
            low_att_k = max(RAP_annotation.data(person_idx_k, lower_cloth_type_index), [], 1);
            low_col_k = max(RAP_annotation.data(person_idx_k, lower_cloth_color_index), [], 1);
            
            app_status = -1;
            up_att_diff = sum(abs(up_att_j - up_att_k))/length(upper_cloth_type_index);
            up_col_diff = sum(abs(up_col_j - up_col_k))/length(upper_cloth_color_index);
            low_att_diff = sum(abs(low_att_j - low_att_k))/length(lower_cloth_type_index);
            low_col_diff = sum(abs(low_col_j - low_col_k))/length(lower_cloth_color_index);
            app_status_up = 0;
            app_status_low = 0;
            if up_att_diff>=threshold(1) || up_col_diff >=threshold(2)
                app_status_up = 1;
            end
            if low_att_diff>=threshold(3) || low_col_diff >= threshold(4)
                app_status_low = 1;
            end
            if app_status_up ==0 && app_status_low ==0
                app_status = 1;
            end
            if app_status_up ==1 && app_status_low ==0
                app_status = 2;
            end
            if app_status_up ==0 && app_status_low ==1
                app_status = 3;
            end
            if app_status_up ==1 && app_status_low ==1
                app_status = 4;
            end
            % set up the appearance change status
            diff_matrix_tmp(person_day_idx_j, person_day_idx_k) = app_status;
            diff_matrix_tmp(person_day_idx_k, person_day_idx_j) = app_status;
            app_status_array = [app_status_array app_status];
        end
    end
    cross_day_person_identity_appearance_diff_matrix{cross_cnt} = diff_matrix_tmp;
    cross_cnt = cross_cnt + 1;
end
for i = 1:4
    cnt_threshold(i) = sum(app_status_array == i);
end
cnt_threshold


if test_reid && metric_euclidean
    fprintf('preprocessing for person re-id based on Euclidean metric starts now.\n');
    % compute the squared Euclidean distance
    dist_eu = sqdist(feat_ide_test', feat_ide_query');
    [nGalleryPerson, nProbePerson] = size(dist_eu);
    [~, IDX_eu] = sort(dist_eu, 1, 'ascend');
    % cmc_protocol_eu = zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_eu = zeros(nProbePerson, 1);
    % cmc_protocol_eu_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_eu_sameday = zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_eu_crossday = zeros(nProbePerson, 1);
    ap_protocol_eu_sameday = zeros(nProbePerson, 1);
    % eu_crossday_cnt = zeros(nProbePerson, 1);
    eu_sameday_cnt = zeros(nProbePerson, 1);
    % support the statistic of cross day at different appearance change
    % conditions.
    cmc_protocol_eu_crossday = zeros(nProbePerson, 4, nGalleryPerson);
    ap_protocol_eu_crossday = zeros(nProbePerson, 4);
    eu_crossday_cnt = zeros(nProbePerson, 4);
    
    fprintf('person re-id for each query based on Euclidean metric starts now.\n');
    parfor i = 1:length(query_index)
        [1 i]
        % check the id with different days
        flag_sameid = person_id == person_id(query_index(i));
        alldays = unique(person_day(flag_sameid));
        index = IDX_eu(:, i);
        % fot each query, test the same day ap
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        flag_samecam = 1 - flag_nsamecam;
        flag_sameday = person_day == person_day(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam + flag_sameday == 3);
        % junk_index1 = find(flag_sameid + flag_samecam + flag_sameday == 3);
        % junk_index2 = find(flag_sameday == 0); % the images in different days are junk images
        junk_index = find(flag_sameid+flag_samecam+flag_sameday == 3 | flag_sameday == 0);
        if length(good_index) >= 1
            [ap_protocol_eu_sameday(i), cmc_protocol_eu_sameday(i, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
            eu_sameday_cnt(i) = 1;
        else
            ap_protocol_eu_sameday(i) = 0;
            cmc_protocol_eu_sameday(i, :) = zeros(1, nGalleryPerson);
        end

        % process the cross day test
        local_cnt = zeros(4,1);
        cmc_protocol_tmp = zeros(4, nGalleryPerson);
        ap_protocol_tmp = zeros(4,1);
        
        % load the appearance change status
        
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            % obtain the query day and gallery day
            query_day = person_day(query_index(i));
            gallery_day = alldays(j);
            % obtain the diff_matrix for query identity
            query_pid = person_id(query_index(i));
            diff_maxtrix_index = find(cross_day_person_identity==query_pid);
            diff_matrix = cross_day_person_identity_appearance_diff_matrix{diff_maxtrix_index};
            query_day_index = find(query_day == unique_days);
            gallery_day_index = find(gallery_day == unique_days);
            appearance_status_value = diff_matrix(query_day_index, gallery_day_index);
            
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index >= 1)
                % 
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                % lookup the appearance change status
                cmc_protocol_tmp(appearance_status_value, :) = cmc_protocol_tmp(appearance_status_value, :) + tmp_cmc;
                ap_protocol_tmp(appearance_status_value, 1) = ap_protocol_tmp(appearance_status_value, 1) + tmp_ap;
                local_cnt(appearance_status_value, 1) = local_cnt(appearance_status_value, 1) + 1;
            end
        end
        for k = 1:4
            if local_cnt(k,1) >= 1
                ap_protocol_eu_crossday(i, k) = ap_protocol_tmp(k, :)/local_cnt(k,1);
                cmc_protocol_eu_crossday(i, k, :) = cmc_protocol_tmp(k, :)/local_cnt(k, 1);
                eu_crossday_cnt(i, k) = local_cnt(k, 1);
            else
                ap_protocol_eu_crossday(i, k) = 0;
                cmc_protocol_eu_crossday(i, k, :) = cmc_protocol_tmp(k, :);
                eu_crossday_cnt(i, k) = 0;
            end
        end
    end
%     tmp = ap_protocol_eu_sameday.*eu_sameday_cnt;
%     mAP_eu_sameday = sum(tmp(:))/sum(eu_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_eu_sameday, eu_sameday_cnt);
%     CMC_eu_sameday = sum(tmp)/sum(eu_sameday_cnt);

    % only summary the samples who can do same-day and cross-day retrieval in the same time.
    useless_flag = sum(eu_crossday_cnt,2) == 0 | eu_sameday_cnt == 0;
    eu_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_eu_sameday.*eu_sameday_cnt;
    mAP_eu_sameday = sum(tmp(:))/sum(eu_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_eu_sameday, eu_sameday_cnt);
    CMC_eu_sameday = sum(tmp)/sum(eu_sameday_cnt);
    
    eu_crossday_cnt(useless_flag, :) = 0;
    tmp = ap_protocol_eu_crossday.*eu_crossday_cnt;
    mAP_eu_crossday = sum(tmp)./sum(eu_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_eu_crossday, eu_crossday_cnt);
    
    CMC_eu_crossday = bsxfun(@rdivide, squeeze(sum(tmp, 1)), sum(eu_crossday_cnt, 1)');
    fprintf('person re-id based on Euclidean distance is ok now.\n')
end

if test_reid && metric_xqda
    % train the model for xqda
    fprintf('preprocessing for person re-id based on XQDA metric starts now.\n');
    label_train = person_id(train_index);
    cam_train = person_cam(train_index);
    feature_train = feat_ide_train;
    [train_sample1, train_sample2, label1, label2] = gen_train_sample_xqda(label_train, cam_train, feature_train'); % generate pairwise training features for XQDA
    [W, M_xqda] = XQDA(train_sample1, train_sample2, label1, label2); % train XQDA, input is N*D
    % test the model for xqda
    % compute the squared Xqda metric distance
    dist_xqda = MahDist(M_xqda, feat_ide_test*W, feat_ide_query*W);
    [nGalleryPerson, nProbePerson] = size(dist_xqda);
    [~, IDX_xqda] = sort(dist_xqda, 1, 'ascend');
    % cmc_protocol_xqda= zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_xqda = zeros(nProbePerson, 1);
    % cmc_protocol_xqda_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_xqda_sameday = zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_xqda_crossday = zeros(nProbePerson, 1);
    ap_protocol_xqda_sameday = zeros(nProbePerson, 1);
    % xqda_crossday_cnt = zeros(nProbePerson, 1);
    xqda_sameday_cnt = zeros(nProbePerson, 1);
    
    cmc_protocol_xqda_crossday = zeros(nProbePerson, 4, nGalleryPerson);
    ap_protocol_xqda_crossday = zeros(nProbePerson, 4);
    xqda_crossday_cnt = zeros(nProbePerson, 4);
    
    fprintf('person re-id for each query based on XQDA metric starts now.\n');
    parfor i = 1:length(query_index)
        [2 i]
        % check the id with different days
        flag_sameid = person_id == person_id(query_index(i));
        alldays = unique(person_day(flag_sameid));
        index = IDX_xqda(:, i);
        % fot each query, test the same day ap
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        flag_samecam = 1 - flag_nsamecam;
        flag_sameday = person_day == person_day(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam + flag_sameday == 3);
        % junk_index1 = find(flag_sameid + flag_samecam + flag_sameday == 3);
        % junk_index2 = find(flag_sameday == 0); % the images in different days are junk images
        junk_index = find(flag_sameid+flag_samecam+flag_sameday == 3 | flag_sameday == 0);
        if length(good_index) >= 1
            [ap_protocol_xqda_sameday(i), cmc_protocol_xqda_sameday(i, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
            xqda_sameday_cnt(i) = 1;
        else
            ap_protocol_xqda_sameday(i) = 0;
            cmc_protocol_xqda_sameday(i, :) = zeros(1, nGalleryPerson);
        end
        
        % process the cross day test
        local_cnt = zeros(4,1);
        cmc_protocol_tmp = zeros(4, nGalleryPerson);
        ap_protocol_tmp = zeros(4,1);
        
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            % obtain the query day and gallery day
            query_day = person_day(query_index(i));
            gallery_day = alldays(j);
            % obtain the diff_matrix for query identity
            query_pid = person_id(query_index(i));
            diff_maxtrix_index = find(cross_day_person_identity==query_pid);
            diff_matrix = cross_day_person_identity_appearance_diff_matrix{diff_maxtrix_index};
            query_day_index = find(query_day == unique_days);
            gallery_day_index = find(gallery_day == unique_days);
            appearance_status_value = diff_matrix(query_day_index, gallery_day_index);
            
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index >= 1)
                % [ap_protocol_xqda_crossday(cnt_crossday), cmc_protocol_xqda_crossday(cnt_crossday, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                 % lookup the appearance change status
                cmc_protocol_tmp(appearance_status_value, :) = cmc_protocol_tmp(appearance_status_value, :) + tmp_cmc;
                ap_protocol_tmp(appearance_status_value, 1) = ap_protocol_tmp(appearance_status_value, 1) + tmp_ap;
                local_cnt(appearance_status_value, 1) = local_cnt(appearance_status_value, 1) + 1;
            end
        end
        for k = 1:4
            if local_cnt(k,1) >= 1
                ap_protocol_xqda_crossday(i, k) = ap_protocol_tmp(k, :)/local_cnt(k,1);
                cmc_protocol_xqda_crossday(i, k, :) = cmc_protocol_tmp(k, :)/local_cnt(k, 1);
                xqda_crossday_cnt(i, k) = local_cnt(k, 1);
            else
                ap_protocol_xqda_crossday(i, k) = 0;
                cmc_protocol_xqda_crossday(i, k, :) = cmc_protocol_tmp(k, :);
                xqda_crossday_cnt(i, k) = 0;
            end
        end
    end
%     tmp = ap_protocol_xqda_sameday.*xqda_sameday_cnt;
%     mAP_xqda_sameday = sum(tmp(:))/sum(xqda_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_xqda_sameday,xqda_sameday_cnt);
%     CMC_xqda_sameday = sum(tmp)/sum(xqda_sameday_cnt);

    % only summary the samples who can do cross-day retrieval
    useless_flag = sum(xqda_crossday_cnt,2) == 0 | xqda_sameday_cnt == 0;
    xqda_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_xqda_sameday.*xqda_sameday_cnt;
    mAP_xqda_sameday = sum(tmp(:))/sum(xqda_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_xqda_sameday, xqda_sameday_cnt);
    CMC_xqda_sameday = sum(tmp)/sum(xqda_sameday_cnt);
      
    xqda_crossday_cnt(useless_flag, :) = 0;
    tmp = ap_protocol_xqda_crossday.*xqda_crossday_cnt;
    mAP_xqda_crossday = sum(tmp)./sum(xqda_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_xqda_crossday, xqda_crossday_cnt);
    CMC_xqda_crossday = bsxfun(@rdivide, squeeze(sum(tmp, 1)), sum(xqda_crossday_cnt, 1)');
    fprintf('person re-id based on XQDA distance is ok now.\n')
end

if test_reid && metric_kissme
     fprintf('preprocessing for person re-id based on Kissme metric starts now.\n');
    run('init.m')
    % train the model for kissme
    label_train = person_id(train_index);
    cam_train = person_cam(train_index);
    % the kissme metric learning
    [idxa, idxb, flag] = gen_train_sample_kissme(label_train, cam_train);

    params.numCoeffs = kissme_pca_dim; 
    pair_metric_learn_algs = { ...
        LearnAlgoKISSME(params), ...
        LearnAlgoMahal(), ...
        LearnAlgoMLEuclidean()
        };

    % dimension reduction by PCA
    [ux_train, u, m, v] = applypca(feat_ide_train'); % ux_train:D*N, u:orthogonal matrx, m:D*1
    if params.numCoeffs <= 0
        cmc_v = cumsum(v);
        sum_v = sum(v(:));
        for i=1:length(cmc_v)
            if cmc_v(i)/sum_v >= 0.95
                params.numCoeffs = i;
                fprintf(sprintf('default pca dimension for kissme: %d\n', i));
                break
            end
        end
    else
        energy_ratio = sum(v(1:params.numCoeffs))/sum(v(:));
        fprintf(sprintf('pca keep %f energy for kissme.\n', energy_ratio));
    end
    ux_test = u'*(feat_ide_test - repmat(m',[size(feat_ide_test,1) 1]))';
    ux_query = u'*(feat_ide_query - repmat(m', [size(feat_ide_query, 1) 1]))';
    ux_gallery = u'*(feat_ide_test- repmat(m', [size(feat_ide_test, 1) 1]))';
    
    ux_train = double(ux_train(1:params.numCoeffs, :)); 
    ux_gallery = double(ux_gallery(1:params.numCoeffs, :));
    ux_query = double(ux_query(1:params.numCoeffs, :));

    [M_kissme, ~, ~] = KISSME(pair_metric_learn_algs, ux_train, ux_gallery, ux_query, idxa, idxb, flag);

    % test the model for kissme
    % compute the squared Mahaland distance
    dist_kissme = MahDist(M_kissme, ux_gallery', ux_query');
    [nGalleryPerson, nProbePerson] = size(dist_kissme);
    [~, IDX_kissme] = sort(dist_kissme, 1, 'ascend');
    % cmc_protocol_kissme= zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_kissme = zeros(nProbePerson, 1);
    % cmc_protocol_kissme_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_kissme_sameday = zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_kissme_crossday = zeros(nProbePerson, 1);
    ap_protocol_kissme_sameday = zeros(nProbePerson, 1);
    % kissme_crossday_cnt = zeros(nProbePerson, 1);
    kissme_sameday_cnt = zeros(nProbePerson, 1);
    
    % support the statistic of cross day at different appearance change
    % conditions.
    cmc_protocol_kissme_crossday = zeros(nProbePerson, 4, nGalleryPerson);
    ap_protocol_kissme_crossday = zeros(nProbePerson, 4);
    kissme_crossday_cnt = zeros(nProbePerson, 4);
    
    fprintf('person re-id for each query based on kissme metric starts now.\n');
    parfor i = 1:length(query_index)
        [3 i]
        % check the id with different days
        flag_sameid = person_id == person_id(query_index(i));
        alldays = unique(person_day(flag_sameid));
        index = IDX_kissme(:, i);
        % fot each query, test the same day ap
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        flag_samecam = 1 - flag_nsamecam;
        flag_sameday = person_day == person_day(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam + flag_sameday == 3);
        % junk_index1 = find(flag_sameid + flag_samecam + flag_sameday == 3);
        % junk_index2 = find(flag_sameday == 0); % the images in different days are junk images
        junk_index = find(flag_sameid+flag_samecam+flag_sameday == 3 | flag_sameday == 0);
        if length(good_index) >= 1
            [ap_protocol_kissme_sameday(i), cmc_protocol_kissme_sameday(i, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
            kissme_sameday_cnt(i) = 1;
        else
            ap_protocol_kissme_sameday(i) = 0;
            cmc_protocol_kissme_sameday(i, :) = zeros(1, nGalleryPerson);
        end
        
        % process the cross day test
        local_cnt = zeros(4,1);
        cmc_protocol_tmp = zeros(4, nGalleryPerson);
        ap_protocol_tmp = zeros(4,1);
     
        
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            % obtain the query day and gallery day
            query_day = person_day(query_index(i));
            gallery_day = alldays(j);
            % obtain the diff_matrix for query identity
            query_pid = person_id(query_index(i));
            diff_maxtrix_index = find(cross_day_person_identity==query_pid);
            diff_matrix = cross_day_person_identity_appearance_diff_matrix{diff_maxtrix_index};
            query_day_index = find(query_day == unique_days);
            gallery_day_index = find(gallery_day == unique_days);
            appearance_status_value = diff_matrix(query_day_index, gallery_day_index);
            
            
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index) >= 1
                % [ap_protocol_kissme_crossday(cnt_crossday), cmc_protocol_kissme_crossday(cnt_crossday, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                 % lookup the appearance change status
                cmc_protocol_tmp(appearance_status_value, :) = cmc_protocol_tmp(appearance_status_value, :) + tmp_cmc;
                ap_protocol_tmp(appearance_status_value, 1) = ap_protocol_tmp(appearance_status_value, 1) + tmp_ap;
                local_cnt(appearance_status_value, 1) = local_cnt(appearance_status_value, 1) + 1;
            end
        end
        for k = 1:4
            if local_cnt(k,1) >= 1
                ap_protocol_kissme_crossday(i, k) = ap_protocol_tmp(k, :)/local_cnt(k,1);
                cmc_protocol_kissme_crossday(i, k, :) = cmc_protocol_tmp(k, :)/local_cnt(k, 1);
                kissme_crossday_cnt(i, k) = local_cnt(k, 1);
            else
                ap_protocol_kissme_crossday(i, k) = 0;
                cmc_protocol_kissme_crossday(i, k, :) = cmc_protocol_tmp(k, :);
                kissme_crossday_cnt(i, k) = 0;
            end
        end
    end
%     tmp = ap_protocol_kissme_sameday.*kissme_sameday_cnt;
%     mAP_kissme_sameday = sum(tmp(:))/sum(kissme_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_kissme_sameday,kissme_sameday_cnt);
%     CMC_kissme_sameday = sum(tmp)/sum(kissme_sameday_cnt);

    % only summary the samples who can do cross-day retrieval
    useless_flag = sum(kissme_crossday_cnt,2) == 0 | kissme_sameday_cnt == 0;
    kissme_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_kissme_sameday.*kissme_sameday_cnt;
    mAP_kissme_sameday = sum(tmp(:))/sum(kissme_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_kissme_sameday, kissme_sameday_cnt);
    CMC_kissme_sameday = sum(tmp)/sum(kissme_sameday_cnt);
    
    kissme_crossday_cnt(useless_flag, :) = 0;
    tmp = ap_protocol_kissme_crossday.*kissme_crossday_cnt;
    mAP_kissme_crossday = sum(tmp)./sum(kissme_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_kissme_crossday, kissme_crossday_cnt);
    CMC_kissme_crossday = bsxfun(@rdivide, squeeze(sum(tmp, 1)), sum(kissme_crossday_cnt, 1)');
    fprintf('person re-id based on KISS metric is ok now.\n');
end

toc

TopK = 500;
CMC = zeros(15, TopK);
mAP = zeros(15, 1);
if test_reid && metric_euclidean 
    % same-day
    L = size(CMC_eu_sameday, 2) ;
    mAP(1) = mAP_eu_sameday;
    if L > TopK
        CMC(1,:) = CMC_eu_sameday(1, 1:TopK);
    else
        CMC(1,1:L) = CMC_eu_sameday(1, 1:L);
    end
    % cross-day
    L = size(CMC_eu_crossday, 2) ;
    mAP(2:5) = mAP_eu_crossday;
    if L > TopK
        CMC(2:5,:) = CMC_eu_crossday(:, 1:TopK);
    else
        CMC(2:5,1:L) = CMC_eu_crossday(:, 1:L);
    end
end
if test_reid && metric_kissme 
    % same-day
    L = size(CMC_kissme_sameday, 2) ;
    mAP(6) = mAP_kissme_sameday;
    if L > TopK
        CMC(6,:) = CMC_kissme_sameday(1, 1:TopK);
    else
        CMC(6,1:L) = CMC_kissme_sameday(1, 1:L);
    end
    % cross-day
    L = size(CMC_kissme_crossday, 2) ;
    mAP(7:10) = mAP_kissme_crossday;
    if L > TopK
        CMC(7:10,:) = CMC_kissme_crossday(:, 1:TopK);
    else
        CMC(7:10,1:L) = CMC_kissme_crossday(:, 1:L);
    end
end
if test_reid && metric_xqda
    % same-day
    L = size(CMC_xqda_sameday, 2) ;
    mAP(11) = mAP_xqda_sameday;
    if L > TopK
        CMC(11,:) = CMC_xqda_sameday(:, 1:TopK);
    else
        CMC(11,1:L) = CMC_xqda_sameday(:, 1:L);
    end
    % cross-day
    L = size(CMC_xqda_crossday, 2) ;
    mAP(12:15) = mAP_xqda_crossday;
    if L > TopK
        CMC(12:15,:) = CMC_xqda_crossday(:, 1:TopK);
    else
        CMC(12:15,1:L) = CMC_xqda_crossday(:, 1:L);
    end
end

mAP
CMC(:, [1 5:5:50])
IDX_eu = IDX_eu(1:1000, :);
IDX_kissme = IDX_kissme(1:1000, :);
IDX_xqda = IDX_xqda(1:1000, :);
save(save_mat, 'Result_att', 'mAP', 'CMC', 'test_index', 'query_index', 'IDX_eu', 'IDX_kissme', 'IDX_xqda', 'ap_protocol_eu_sameday',  'ap_protocol_xqda_sameday', 'ap_protocol_kissme_sameday','ap_protocol_xqda_crossday', 'ap_protocol_kissme_crossday', 'ap_protocol_eu_crossday');
