#!/bin/bash sh

Net=CaffeNet
Partion=5
Model_iter=50000
Iter_cnt=170
Model_type=trainval
Test_type=test

if [ ! -d "lmdb/result_${Net}_fc8_${Partion}_lmdb" ]; then
    echo "OK"
else
    rm -rf lmdb/result_${Net}_fc8_${Partion}_lmdb
fi


extract_features ./temp_models/${Net}/rap2_${Model_type}_part${Partion}_iter_${Model_iter}.caffemodel \
    ./prototxts/${Net}/rap2_${Test_type}_${Partion}.prototxt \
    fine_fc8 lmdb/result_${Net}_fc8_${Partion}_lmdb ${Iter_cnt} lmdb GPU 0

python compute_accuracy_fc8.py ${Net} ${Test_type} ${Partion}
