#!/usr/bin/env bash

GPUID=4
Model_type=train # train or trainval
LOG="logs/CaffeNet/rap2_${Model_type}_drop0.9.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/CaffeNet/rap2_solver_${Model_type}.prototxt \
    --weights=./../pretrained/bvlc_reference_caffenet.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
