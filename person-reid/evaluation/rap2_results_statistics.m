% this script evaluate different methods on re-identification performance on rap database

feature_types = {'elf', 'lomo', 'gog', 'jstl'};
Results = zeros(24, 6); % map, rank1, for Eulicidean, Kiss, xqda
for i=1:length(feature_types)
    load(sprintf('results/%s/result_testtype0.mat', feature_types{i}));
    Results(i*2-1, 1:2:6) = CMC(:,1);
    Results(i*2-1, 2:2:6) = mAP;
    load(sprintf('results/%s/result_testtype1.mat', feature_types{i}));
    Results(i*2, 1:2:6) = CMC(:,1);
    Results(i*2, 2:2:6) = mAP;
end

alg_types = {'IDE'};
net_types = {'ResNet101', 'ResNet152', 'DenseNet121', 'APR2', 'MSCAN_full', 'MuDeep', 'HACNN'}
for i=1:length(alg_types)
    for j=1:length(net_types)
        load(sprintf('results/%s/result_algtype%s_testtype0.mat', net_types{j}, alg_types{i}));
        Results(8 + j*2-1, 1:2:6) = CMC(:,1);
        Results(8 + j*2-1, 2:2:6) = mAP;

        load(sprintf('results/%s/result_algtype%s_testtype1.mat', net_types{j}, alg_types{i}));
        Results(8 + j*2, 1:2:6) = CMC(:,1);
        Results(8 + j*2, 2:2:6) = mAP;
    end
end

% CaffeNet
Results_att = zeros(8,5);
Results_att_each = zeros(8, 54);
alg_types = {'att', 'IDE', 'IDE-att'};
for i=1:length(alg_types)
    load(sprintf('results/CaffeNet/result_algtype%s_testtype0.mat', alg_types{i}));
    Results(22 + i*2-1, 1:2:6) = CMC(:,1);
    Results(22 + i*2-1, 2:2:6) = mAP;
    if (i==1) || (i==3)
        idx = (i+1)/2;
        Results_att(2*idx-1,:) = [Result_att.label_accuracy Result_att.instance_accuracy Result_att.instance_precision Result_att.instance_recall Result_att.instance_F1];
        Results_att_each(2*idx-1, :) = Result_att.label_accuracy_all;
    end
    load(sprintf('results/CaffeNet/result_algtype%s_testtype1.mat', alg_types{i}));
    Results(22 + i*2, 1:2:6) = CMC(:,1);
    Results(22 + i*2, 2:2:6) = mAP;
    if (i==1) || (i==3)
        idx = (i+1)/2;
        Results_att(2*idx,:) = [Result_att.label_accuracy Result_att.instance_accuracy Result_att.instance_precision Result_att.instance_recall Result_att.instance_F1];
        Results_att_each(2*idx, :) = Result_att.label_accuracy_all;
    end
end

% ResNet50
for i=1:length(alg_types)
    load(sprintf('results/ResNet50/result_algtype%s_testtype0.mat', alg_types{i}));
    Results(28 + i*2-1, 1:2:6) = CMC(:,1);
    Results(28 + i*2-1, 2:2:6) = mAP;
    if (i==1) || (i==3)
        idx = (i+1)/2;
        Results_att(4+ 2*idx-1,:) = [Result_att.label_accuracy Result_att.instance_accuracy Result_att.instance_precision Result_att.instance_recall Result_att.instance_F1];
        Results_att_each(4 + 2*idx-1, :) = Result_att.label_accuracy_all;
    end
    load(sprintf('results/ResNet50/result_algtype%s_testtype1.mat', alg_types{i}));
    Results(28 + i*2, 1:2:6) = CMC(:,1);
    Results(28 + i*2, 2:2:6) = mAP;
    if (i==1) || (i==3)
        idx = (i+1)/2;
        Results_att(4+ 2*idx,:) = [Result_att.label_accuracy Result_att.instance_accuracy Result_att.instance_precision Result_att.instance_recall Result_att.instance_F1];
        Results_att_each(4 + 2*idx, :) = Result_att.label_accuracy_all;
    end
end

save('./results/reid_results.mat', 'Results_att', 'Results', 'Results_att_each')
