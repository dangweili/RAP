#!/usr/bin/env bash
Partation=1
GPUID=1
Model_type=train # train or trainval
LOG="logs/CaffeNet/rap2_${Model_type}_part${Partation}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/CaffeNet/rap2_solver_${Model_type}_${Partation}.prototxt \
    --weights=./../pretrained/bvlc_reference_caffenet.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
