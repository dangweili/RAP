#!/usr/bin/env bash
Partation="$1"
GPUID="$2"
Model_type=trainval # train or trainval

LOG="logs/ResNet50/rap2_${Model_type}_part${Partation}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/ResNet50/rap2_solver_${Model_type}_${Partation}.prototxt \
    --weights=./../pretrained/ResNet-50-model.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
