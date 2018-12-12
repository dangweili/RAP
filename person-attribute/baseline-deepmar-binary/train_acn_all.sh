#!/usr/bin/env bash
Partation=2
GPUID=5
for AttIdx in $(seq 37 54)
do
sh ./train_acn_all_sub.sh ${Partation} ${AttIdx} ${GPUID}
done
