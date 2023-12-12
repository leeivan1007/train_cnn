#!/bin/bash

dataset="$1"
NUM="$2"

for (( i=1; i<=${NUM}; i++ )); do
    echo "i=${i}"
    python train_cnn.py --dataset ${dataset}
done
