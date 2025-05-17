#!/bin/bash

# Check if sufficient arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <scene_name>"
    exit 1
fi

# clear;
for ((i=0; i<3; i++));
# do tail -n 1 output/$1_$i/testset_200000/eval_hdr2/eval_hdr.csv;
do tail -n 1 output/$1_$i/testset_200000/eval_*.csv;
done

