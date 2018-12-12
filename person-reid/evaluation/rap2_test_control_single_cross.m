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
test_type = 0; % 1 will including -1 persons in test stage while 0 will not. In test_control, this should always be 1 to add the distractors in each day.
test_alg_type = 'att'; % att, IDE, or IDE-att
test_att = 1;
feature_attribute_name = sprintf('./features/%s/%s_att.mat', test_net, test_alg_type);
test_reid = 1; % whether we should test reid or not
feature_ide_name = sprintf('./features/%s/%s_ide.mat', test_net, test_alg_type);

metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 80% energy
save_mat = sprintf('results_control/%s/result_algtype%s_testtype%d.mat', test_net, test_alg_type, test_type);

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
if test_reid && metric_euclidean
    fprintf('preprocessing for person re-id based on Euclidean metric starts now.\n');
    % compute the squared Euclidean distance
    dist_eu = sqdist(feat_ide_test', feat_ide_query');
    [nGalleryPerson, nProbePerson] = size(dist_eu);
    [~, IDX_eu] = sort(dist_eu, 1, 'ascend');
    % cmc_protocol_eu = zeros(nProbePerson, nGalleryPerson);
    % ap_protocol_eu = zeros(nProbePerson, 1);
    cmc_protocol_eu_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_eu_sameday = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_eu_crossday = zeros(nProbePerson, 1);
    ap_protocol_eu_sameday = zeros(nProbePerson, 1);
    eu_crossday_cnt = zeros(nProbePerson, 1);
    eu_sameday_cnt = zeros(nProbePerson, 1);
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
        local_cnt = 0;
        cmc_protocol_tmp = zeros(1, nGalleryPerson);
        ap_protocol_tmp = 0;
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index >= 1)
                % [ap_protocol_eu_crossday(cnt_crossday), cmc_protocol_eu_crossday(cnt_crossday, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                cmc_protocol_tmp = cmc_protocol_tmp + tmp_cmc;
                ap_protocol_tmp = ap_protocol_tmp + tmp_ap;
                local_cnt = local_cnt + 1;
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
%     tmp = ap_protocol_eu_sameday.*eu_sameday_cnt;
%     mAP_eu_sameday = sum(tmp(:))/sum(eu_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_eu_sameday, eu_sameday_cnt);
%     CMC_eu_sameday = sum(tmp)/sum(eu_sameday_cnt);

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
    cmc_protocol_xqda_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_xqda_sameday = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_xqda_crossday = zeros(nProbePerson, 1);
    ap_protocol_xqda_sameday = zeros(nProbePerson, 1);
    xqda_crossday_cnt = zeros(nProbePerson, 1);
    xqda_sameday_cnt = zeros(nProbePerson, 1);
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
        local_cnt = 0;
        cmc_protocol_tmp = zeros(1, nGalleryPerson);
        ap_protocol_tmp = 0;
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index >= 1)
                % [ap_protocol_xqda_crossday(cnt_crossday), cmc_protocol_xqda_crossday(cnt_crossday, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                cmc_protocol_tmp = cmc_protocol_tmp + tmp_cmc;
                ap_protocol_tmp = ap_protocol_tmp + tmp_ap;
                local_cnt = local_cnt + 1;
            end
        end
        if local_cnt >= 1
            ap_protocol_xqda_crossday(i) = ap_protocol_tmp/local_cnt;
            cmc_protocol_xqda_crossday(i, :) = cmc_protocol_tmp/local_cnt;
            xqda_crossday_cnt(i) = local_cnt;
        else
            ap_protocol_xqda_crossday(i) = 0;
            cmc_protocol_xqda_crossday(i, :) = cmc_protocol_tmp;
            xqda_crossday_cnt(i) = 0;
        end
    end
%     tmp = ap_protocol_xqda_sameday.*xqda_sameday_cnt;
%     mAP_xqda_sameday = sum(tmp(:))/sum(xqda_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_xqda_sameday,xqda_sameday_cnt);
%     CMC_xqda_sameday = sum(tmp)/sum(xqda_sameday_cnt);

    % only summary the samples who can do cross-day retrieval
    useless_flag = xqda_crossday_cnt == 0 | xqda_sameday_cnt == 0;
    xqda_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_xqda_sameday.*xqda_sameday_cnt;
    mAP_xqda_sameday = sum(tmp(:))/sum(xqda_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_xqda_sameday, xqda_sameday_cnt);
    CMC_xqda_sameday = sum(tmp)/sum(xqda_sameday_cnt);
    
    xqda_crossday_cnt(useless_flag) = 0;
    tmp = ap_protocol_xqda_crossday.*xqda_crossday_cnt;
    mAP_xqda_crossday = sum(tmp)/sum(xqda_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_xqda_crossday, xqda_crossday_cnt);
    CMC_xqda_crossday = sum(tmp)/sum(xqda_crossday_cnt);
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
    cmc_protocol_kissme_crossday = zeros(nProbePerson, nGalleryPerson);
    cmc_protocol_kissme_sameday = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_kissme_crossday = zeros(nProbePerson, 1);
    ap_protocol_kissme_sameday = zeros(nProbePerson, 1);
    kissme_crossday_cnt = zeros(nProbePerson, 1);
    kissme_sameday_cnt = zeros(nProbePerson, 1);
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
        local_cnt = 0;
        cmc_protocol_tmp = zeros(1, nGalleryPerson);
        ap_protocol_tmp = 0;
        for j = 1:length(alldays)
            if alldays(j) == person_day(query_index(i))
                continue;
            end
            flag_crossday = person_day == alldays(j);
            good_index = find(flag_sameid + flag_nsamecam + flag_crossday == 3);
            junk_index = find(flag_sameid + flag_samecam + flag_crossday == 3 | flag_crossday == 0);
            if length(good_index) >= 1
                % [ap_protocol_kissme_crossday(cnt_crossday), cmc_protocol_kissme_crossday(cnt_crossday, :)] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                [tmp_ap, tmp_cmc] = compute_AP(good_index, junk_index, test_index(index)); % all the index value are global, from 1 to 41585
                cmc_protocol_tmp = cmc_protocol_tmp + tmp_cmc;
                ap_protocol_tmp = ap_protocol_tmp + tmp_ap;
                local_cnt = local_cnt + 1;
            end
        end
        if local_cnt >= 1
            ap_protocol_kissme_crossday(i) = ap_protocol_tmp/local_cnt;
            cmc_protocol_kissme_crossday(i, :) = cmc_protocol_tmp/local_cnt;
            kissme_crossday_cnt(i) = local_cnt;
        else
            ap_protocol_kissme_crossday(i) = 0;
            cmc_protocol_kissme_crossday(i, :) = cmc_protocol_tmp;
            kissme_crossday_cnt(i) = 0;
        end
    end
%     tmp = ap_protocol_kissme_sameday.*kissme_sameday_cnt;
%     mAP_kissme_sameday = sum(tmp(:))/sum(kissme_sameday_cnt);
%     tmp = bsxfun(@times, cmc_protocol_kissme_sameday,kissme_sameday_cnt);
%     CMC_kissme_sameday = sum(tmp)/sum(kissme_sameday_cnt);

    % only summary the samples who can do cross-day retrieval
    useless_flag = kissme_crossday_cnt == 0 | kissme_sameday_cnt == 0;
    kissme_sameday_cnt(useless_flag) = 0;
    tmp = ap_protocol_kissme_sameday.*kissme_sameday_cnt;
    mAP_kissme_sameday = sum(tmp(:))/sum(kissme_sameday_cnt);
    tmp = bsxfun(@times, cmc_protocol_kissme_sameday, kissme_sameday_cnt);
    CMC_kissme_sameday = sum(tmp)/sum(kissme_sameday_cnt);
    
    kissme_crossday_cnt(useless_flag) = 0;
    tmp = ap_protocol_kissme_crossday.*kissme_crossday_cnt;
    mAP_kissme_crossday = sum(tmp)/sum(kissme_crossday_cnt);
    tmp = bsxfun(@times, cmc_protocol_kissme_crossday, kissme_crossday_cnt);
    CMC_kissme_crossday = sum(tmp)/sum(kissme_crossday_cnt);
    fprintf('person re-id based on KISS metric is ok now.\n');
end

toc

TopK = 500;
CMC = zeros(6, TopK);
mAP = zeros(6, 1);
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
    mAP(2) = mAP_eu_crossday;
    if L > TopK
        CMC(2,:) = CMC_eu_crossday(1, 1:TopK);
    else
        CMC(2,1:L) = CMC_eu_crossday(1, 1:L);
    end
end
if test_reid && metric_kissme 
    % same-day
    L = size(CMC_kissme_sameday, 2) ;
    mAP(3) = mAP_kissme_sameday;
    if L > TopK
        CMC(3,:) = CMC_kissme_sameday(1, 1:TopK);
    else
        CMC(3,1:L) = CMC_kissme_sameday(1, 1:L);
    end
    % cross-day
    L = size(CMC_kissme_crossday, 2) ;
    mAP(4) = mAP_kissme_crossday;
    if L > TopK
        CMC(4,:) = CMC_kissme_crossday(1, 1:TopK);
    else
        CMC(4,1:L) = CMC_kissme_crossday(1, 1:L);
    end
end
if test_reid && metric_xqda
    % same-day
    L = size(CMC_xqda_sameday, 2) ;
    mAP(5) = mAP_xqda_sameday;
    if L > TopK
        CMC(5,:) = CMC_xqda_sameday(1, 1:TopK);
    else
        CMC(5,1:L) = CMC_xqda_sameday(1, 1:L);
    end
    % cross-day
    L = size(CMC_xqda_crossday, 2) ;
    mAP(6) = mAP_xqda_crossday;
    if L > TopK
        CMC(6,:) = CMC_xqda_crossday(1, 1:TopK);
    else
        CMC(6,1:L) = CMC_xqda_crossday(1, 1:L);
    end
end

mAP
CMC(:, [1 5:5:50])
IDX_eu = IDX_eu(1:1000, :);
IDX_kissme = IDX_kissme(1:1000, :);
IDX_xqda = IDX_xqda(1:1000, :);
save(save_mat, 'Result_att', 'mAP', 'CMC', 'test_index', 'query_index', 'IDX_eu', 'IDX_kissme', 'IDX_xqda', 'ap_protocol_eu_sameday',  'ap_protocol_xqda_sameday', 'ap_protocol_kissme_sameday','ap_protocol_xqda_crossday', 'ap_protocol_kissme_crossday', 'ap_protocol_eu_crossday');
