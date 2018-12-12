#!/usr/bin/env bash
GPUID=0
for Partation in $(seq 3 5)
do
    for AttIdx in $(seq 1 18)
    do
        echo ${Partation},${AttIdx},${GPUID}
        sh ./train_caffenet_all_sub.sh ${Partation} ${AttIdx} ${GPUID}
    done
done
