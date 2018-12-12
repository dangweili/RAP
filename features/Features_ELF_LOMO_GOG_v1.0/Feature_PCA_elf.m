load ../../features/rap2_features_elf.mat

[coeff, score, latent] = princomp(single(imgs_feature));

tmp = cumsum(latent)./sum(latent);

rate = 0.95;

pos = min(find(tmp > 0.95));

imgs_feature = score(:, 1:pos-1);

save('../../features/rap2_features_pcaelf.mat', 'imgs_feature', '-v7.3')
