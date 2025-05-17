#!/bin/bash

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 <scene_name> <gpu_id>"
	exit 1
fi

SPECIAL_SCENE_NAME=box

if [ "$1" == $SPECIAL_SCENE_NAME ]; then
	CONFIG=colorboard
else
	CONFIG=$1
fi

# Loop to execute commands with additional parameters
for i in {0..2}; do
    CUDA_VISIBLE_DEVICES=$2 python3 run_nerf.py --config configs/$CONFIG.txt --datadir data_hdr/$1/ --expname $1_$i --basedir output/ --exps_index $i --kin
done
