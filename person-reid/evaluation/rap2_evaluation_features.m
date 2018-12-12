% this script evaluate the mAP and CMC curve 
tic
clear all; close all;
rand('seed', 0);
addpath( genpath('./utils/') )

% load the annotation file
load ./../../data/RAP_annotation/RAP_annotation.mat

train_identity = RAP_annotation.partition_reid.train_identity;
test_identity = RAP_annotation.partition_reid.test_identity;

test_type = 1; % 1 will including -1 persons in test stage while 0 will not.
test_feat_type = 'elf'; % elf, lomo, gog, jstl 
test_reid = 1; % whether we should test reid or not
feature_name = sprintf('../../features/rap2_features_%s.mat', test_feat_type);
 
if strcmp(test_feat_type, 'gog')
    feature_name = cell(1,1);
    feature_name{1} = sprintf('../../features/ReID_GOG_v1.01/Features/feature_all_1.mat');
    feature_name{2} = sprintf('../../features/ReID_GOG_v1.01/Features/feature_all_2.mat');
    feature_name{3} = sprintf('../../features/ReID_GOG_v1.01/Features/feature_all_3.mat');
    feature_name{4} = sprintf('../../features/ReID_GOG_v1.01/Features/feature_all_4.mat');
end
metric_euclidean = 1;
metric_xqda = 1;
metric_kissme = 1;
kissme_pca_dim = -1; % if set -1,  default extract 90% energy
eig_rate = 0.95;
save_mat = sprintf('results/%s/result_testtype%d.mat', test_feat_type, test_type);

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
if test_reid && not(strcmp(test_feat_type, 'gog'))
    load(feature_name)
    % feat_ide = imgs_feature;
    feat_ide_train = imgs_feature(train_index, :);
    feat_ide_test = imgs_feature(test_index, :);
    feat_ide_query = imgs_feature(query_index, :); % N*D
    clear imgs_feature;
end

if test_reid && strcmp(test_feat_type,'gog')
    % load four types of gog descriptors
   feat_ide_train = [];
   feat_ide_test = [];
   feat_ide_query = [];
   for j=1:length(feature_name)
       load(feature_name{j});
       % feat_ide = feature_all(1:41585,:);
       feat_ide_train_ = feature_all(train_index, :);
       mean_value = mean(feat_ide_train_,1);
       feat_ide_test_ = feature_all(test_index, :);
       feat_ide_query_ = feature_all(query_index, :); % N*D
       clear feature_all;
       % subtract the meanvalue
       feat_ide_train_ = feat_ide_train_-repmat(mean_value, [size(feat_ide_train_, 1) 1]);
       feat_ide_test_ = feat_ide_test_-repmat(mean_value, [size(feat_ide_test_, 1) 1]);
       feat_ide_query_ = feat_ide_query_-repmat(mean_value, [size(feat_ide_query_, 1) 1]);
       % normalize the feature
       feat_ide_train_ = feat_ide_train_./repmat(sqrt(sum(feat_ide_train_.^2, 2)), [1 size(feat_ide_train_, 2)]);
       feat_ide_test_ = feat_ide_test_./repmat(sqrt(sum(feat_ide_test_.^2, 2)), [1 size(feat_ide_test_, 2)]);
       feat_ide_query_ = feat_ide_query_./repmat(sqrt(sum(feat_ide_query_.^2, 2)), [1 size(feat_ide_query_, 2)]);
       if j==1
          feat_ide_train = feat_ide_train_;
          feat_ide_test = feat_ide_test_;
          feat_ide_query = feat_ide_query_;
       else
          feat_ide_train = [feat_ide_train feat_ide_train_];
          feat_ide_test = [feat_ide_test feat_ide_test_];
          feat_ide_query = [feat_ide_query feat_ide_query_];
       end
   end
end
% normalize each row, N*D 
if test_reid && feat_norm && not(strcmp(test_feat_type,'gog'))
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
    cmc_protocol_eu = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_eu = zeros(nProbePerson, 1);
    fprintf('person re-id for each query based on Euclidean metric starts now.\n');
    parfor i = 1:length(query_index)
        % [1 i]
        % image with the same identity but not same camera
        flag_sameid = person_id == person_id(query_index(i));
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam == 2);
        % image with same identity and same camera
        flag_samecam = person_cam == person_cam(query_index(i));
        junk_index = find(flag_sameid + flag_samecam == 2);
        index = IDX_eu(:, i);
        [ap_protocol_eu(i), cmc_protocol_eu(i, :)] = compute_AP(good_index, junk_index, test_index(index));
    end
    mAP_eu = mean(ap_protocol_eu(:));
    CMC_eu = mean(cmc_protocol_eu, 1);
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
    % compute the squared Euclidean distance
    dist_xqda = MahDist(M_xqda, feat_ide_test*W, feat_ide_query*W);
    [nGalleryPerson, nProbePerson] = size(dist_xqda);
    [~, IDX_xqda] = sort(dist_xqda, 1, 'ascend');
    cmc_protocol_xqda = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_xqda = zeros(nProbePerson, 1);
    fprintf('person re-id for each query based on XQDA metric starts now.\n');
    parfor i = 1:length(query_index)
        % [1 i]
        % image with the same identity but not same camera
        flag_sameid = person_id == person_id(query_index(i));
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam == 2);
        % image with same identity and same camera
        flag_samecam = person_cam == person_cam(query_index(i));
        junk_index = find(flag_sameid + flag_samecam == 2);
        index = IDX_xqda(:, i);
        [ap_protocol_xqda(i), cmc_protocol_xqda(i, :)] = compute_AP(good_index, junk_index, test_index(index));
    end
    mAP_xqda = mean(ap_protocol_xqda(:));
    CMC_xqda = mean(cmc_protocol_xqda, 1);
    fprintf('person re-id based on XQDA metric is ok now.\n')
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
            if cmc_v(i)/sum_v >= eig_rate 
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
    cmc_protocol_kissme = zeros(nProbePerson, nGalleryPerson);
    ap_protocol_kissme = zeros(nProbePerson, 1);
     fprintf('person re-id for each query based on KISS metric starts now.\n');
    parfor i = 1:length(query_index)
        % [1 i]
        % image with the same identity but not same camera
        flag_sameid = person_id == person_id(query_index(i));
        flag_nsamecam = person_cam ~= person_cam(query_index(i));
        good_index = find(flag_sameid + flag_nsamecam == 2);
        % image with same identity and same camera
        flag_samecam = person_cam == person_cam(query_index(i));
        junk_index = find(flag_sameid + flag_samecam == 2);
        index = IDX_kissme(:, i);
        [ap_protocol_kissme(i), cmc_protocol_kissme(i, :)] = compute_AP(good_index, junk_index, test_index(index));
    end
    mAP_kissme = mean(ap_protocol_kissme(:));
    CMC_kissme = mean(cmc_protocol_kissme, 1);
    fprintf('person re-id based on KISS metric is ok now.\n');
end

toc

TopK = 500;
CMC = zeros(3, TopK);
mAP = zeros(3, 1);
if test_reid && metric_euclidean 
    CMC(1,:) = CMC_eu(1, 1:TopK);
    mAP(1) = mAP_eu;
end
if test_reid && metric_kissme 
    CMC(2,:) = CMC_kissme(1, 1:TopK);
    mAP(2) = mAP_kissme;
end
if test_reid && metric_xqda
    CMC(3,:) = CMC_xqda(1, 1:TopK);
    mAP(3) = mAP_xqda;
end

mAP
CMC(:, [1 5:5:50])
IDX_eu = IDX_eu(1:500, :);
IDX_kissme = IDX_kissme(1:500, :);
IDX_xqda = IDX_xqda(1:500, :);
save(save_mat, 'mAP', 'CMC', 'test_index', 'query_index', 'IDX_eu', 'IDX_kissme', 'IDX_xqda', 'ap_protocol_eu', 'ap_protocol_xqda', 'ap_protocol_kissme');
