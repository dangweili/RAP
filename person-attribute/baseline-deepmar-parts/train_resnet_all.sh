#!/usr/bin/env bash
GPUID=6
for Partation in $(seq 2 5)
do
sh ./train_resnet_all_sub.sh ${Partation} ${GPUID}
done
