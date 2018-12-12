#!/bin/bash sh

Net=CaffeNet
Partion=5
Model_iter=50000
Iter_cnt=400
Model_type=trainval
Stage=test
GPUID=0

if [ ! -d "lmdb/result_${Net}_fc8_${Partion}_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_fc8_${Partion}_lmdb
fi
extract_features ./temp_models/${Net}/rap2_${Model_type}_part${Partion}_iter_${Model_iter}.caffemodel \
    ./prototxts/${Net}/rap2_${Stage}_${Partion}.prototxt \
    fine_fc8 lmdb/result_${Net}_fc8_${Partion}_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 
python compute_accuracy_fc8.py ${Net} ${Stage} ${Partion}

