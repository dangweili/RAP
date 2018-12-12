#!/usr/bin/env bash
Partation=2
GPUID=2
for AttIdx in $(seq 37 54)
do
sh ./train_caffenet_all_sub.sh ${Partation} ${AttIdx} ${GPUID}
done
