#!/bin/bash
for var in 0.0 0.04 0.16 0.36 0.64
do
  echo "Running FedIC with gauss_var=$var"
  ./GaussChain/run_imagenet_lt.sh \
    --aggr_method fedic \
    --model mobilenetv3 \
    --gauss_var $var \
    --n_round 10 \
    --n_epoch 5 \
    --batch_size 32 \
    --n_core 4
done
