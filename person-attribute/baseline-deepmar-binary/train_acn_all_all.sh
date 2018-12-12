#!/usr/bin/env bash
GPUID=5
for Partation in $(seq 3 5)
do
    for AttIdx in $(seq 37 54)
    do
        echo ${Partation},${AttIdx},${GPUID}
        sh ./train_acn_all_sub.sh ${Partation} ${AttIdx} ${GPUID}
    done
done
