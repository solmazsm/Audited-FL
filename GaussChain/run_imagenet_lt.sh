#!/bin/bash

# Default parameters
n_core=100
aggr_method="fedcls"  # or "fedic"
gauss_var=0.01
lr=0.01
momentum=0.9
weight_decay=1e-4
epochs=100
batch_size=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_core)
            n_core="$2"
            shift 2
            ;;
        --aggr_method)
            aggr_method="$2"
            shift 2
            ;;
        --gauss_var)
            gauss_var="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --momentum)
            momentum="$2"
            shift 2
            ;;
        --weight_decay)
            weight_decay="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Run the training
mpirun --use-hwthread-cpus --mca btl ^openib -np $n_core python src/gausschain_imagenet.py \
    --aggr_method $aggr_method \
    --gauss_var $gauss_var \
    --lr $lr \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --epochs $epochs \
    --batch_size $batch_size 