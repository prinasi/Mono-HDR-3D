#!/bin/bash

# Check if sufficient arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <scene_name> <gpu_id>"
    exit 1
fi

# Assign input arguments to variables
SCENE_NAME=$1
GPU_ID=$2

# Define scene categories
SYNTHETIC_SCENES=(sponza sofa bear chair desk diningroom dog bathroom)
REAL_SCENES=(flower computer box luckycat)

# Loop to execute commands with additional parameters
for i in {0..2}; do
    if [[ " ${SYNTHETIC_SCENES[@]} " =~ " ${SCENE_NAME} " ]]; then
        python3 train_synthetic.py --config config/${SCENE_NAME}.yaml --gpu_id ${GPU_ID} --syn --eval --exps_index ${i} # --layers 1 ${@:3}
    elif [[ " ${REAL_SCENES[@]} " =~ " ${SCENE_NAME} " ]]; then
        python3 train_real.py --config config/${SCENE_NAME}.yaml --gpu_id ${GPU_ID} --eval --exps_index ${i} # ${@:3}
    else
        echo "Error: Invalid scene name '${SCENE_NAME}'."
        exit 1
    fi
done
