#!/bin/bash sh

Net=CaffeNet
Model_iter=50000
Iter_cnt=400
Model_type=trainval
Stage=test
GPUID=0
#for Partion in 1 2 3 4 5
#do 
#    for AttIdx in `seq 1 54`
#    do
#        if [ ! -d "lmdb/result_${Net}_fc8_${Partion}_${AttIdx}_lmdb" ]; then
#            echo "OK"
#        else
#            rm -rf lmdb/result_${Net}_fc8_${Partion}_${AttIdx}_lmdb
#        fi
#        extract_features ./temp_models/${Net}/rap2_${Model_type}_part${Partion}_${AttIdx}_iter_${Model_iter}.caffemodel \
#            ./prototxts/${Net}/rap2_${Stage}_${Partion}_${AttIdx}.prototxt \
#            fine_fc8 lmdb/result_${Net}_fc8_${Partion}_${AttIdx}_lmdb ${Iter_cnt} lmdb GPU ${GPUID} 
#    done
#done

for Partion in 1 2 3 4 5
do 
    python compute_accuracy_fc8.py ${Net} ${Stage} ${Partion}
done

python compute_average_results_fc8.py ${Net} ${Stage}
