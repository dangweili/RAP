#!/usr/bin/env bash
Partation="$1"
AttIdx="$2"
GPUID="$3"
Model_type=trainval # train or trainval
LOG="logs/ACN/rap2_${Model_type}_part${Partation}_Att${AttIdx}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
GLOG_logtostderr=1 caffe train \
    --solver=./prototxts/ACN/rap2_solver_${Model_type}_${Partation}_${AttIdx}.prototxt \
    --weights=./../pretrained/bvlc_reference_caffenet.caffemodel \
    --gpu=${GPUID} 2>&1 | tee ${LOG}
