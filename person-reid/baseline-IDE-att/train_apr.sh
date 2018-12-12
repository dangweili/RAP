#!/usr/bin/env bash

GPUID=4
Model_type=trainval # train or trainval
LOG="logs/APR/rap2_${Model_type}_drop0.7_lr0.001_eachatt1_54_ide1.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/APR/rap2_solver_${Model_type}.prototxt \
    --weights=./../pretrained/ResNet-50-model.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
