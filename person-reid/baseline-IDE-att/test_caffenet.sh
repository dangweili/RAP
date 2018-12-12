#!/bin/bash sh

Net=CaffeNet
Model_iter=50000
Iter_cnt=304
Model_type=trainval
Stage=test
GPUID=2

if [ ! -d "lmdb/result_${Net}_fc8_att_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_fc8_att_lmdb
fi

extract_features ./temp_models/${Net}/rap2_${Model_type}_eachatt1_54_id1_iter_${Model_iter}.caffemodel \
    ./prototxts/${Net}/rap2_${Stage}.prototxt \
    fine_fc8_att lmdb/result_${Net}_fc8_att_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 
python compute_accuracy_fc8.py ${Net} ${Stage}

