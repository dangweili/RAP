#!/bin/bash sh

Net=ResNet50
Model_iter=50000
Iter_cnt=1900
Model_type=trainval
Stage=test
GPUID=0

if [ ! -d "lmdb/result_${Net}_fc8_att_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_fc8_att_lmdb
fi

extract_features ./temp_models/${Net}/v1/rap2_${Model_type}_iter_${Model_iter}.caffemodel \
    ./prototxts/${Net}/rap2_${Stage}.prototxt \
    fine_fc8_att lmdb/result_${Net}_fc8_att_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 

python compute_accuracy_fc8.py ${Net} ${Stage}

