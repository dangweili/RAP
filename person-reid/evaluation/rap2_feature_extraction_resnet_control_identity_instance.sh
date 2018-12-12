#!/bin/bash sh

Net=ResNet50
Model_iter=50000
Iter_cnt=1300
Model_type=trainval
Alg_type=IDE-att # IDE-att or IDE or att
GPUID=1

if [ ! -d "lmdb/result_${Net}_att_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_att_lmdb
fi

if [ ! -d "lmdb/result_${Net}_ide_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_ide_lmdb
fi
extract_features ./../baseline-${Alg_type}-control/temp_models/${Net}/rap2_${Model_type}_iter_${Model_iter}.caffemodel \
    ./deploy/${Net}/deploy_IDE_att.prototxt \
    fine_fc8_att,pool5 lmdb/result_${Net}_att_lmdb,lmdb/result_${Net}_ide_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 

python parse_lmdb_to_mat.py lmdb/result_${Net}_att_lmdb 41585 ./features/${Net}/${Alg_type}_att_control_identity_instance.mat
python parse_lmdb_to_mat.py lmdb/result_${Net}_ide_lmdb 41585 ./features/${Net}/${Alg_type}_ide_control_identity_instance.mat


