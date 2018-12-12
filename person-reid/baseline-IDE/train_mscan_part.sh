#!/usr/bin/env bash

GPUID=5
Model_type=trainval # train or trainval
LOG="logs/MSCAN_part/rap2_${Model_type}_drop0.7.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/MSCAN_part/rap2_solver_${Model_type}.prototxt \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
