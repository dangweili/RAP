#!/usr/bin/env bash

GPUID=0
Model_type=train # train or trainval
LOG="logs/DenseNet121/rap2_${Model_type}_drop0.5.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/DenseNet121/rap2_solver_${Model_type}.prototxt \
    --weights=./../pretrained/DenseNet_121.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
