#!/bin/bash -l

method=ours       # choose from: source, norm_test, tent, rpl, eta, eata, rdump, sar, cotta, rotta, adacontrast, lame, deyo, cmf, rmt, gtta, roid
setting=continual # choose from: continual, mixed_domains, correlated, mixed_domains_correlated, ccc
seeds=(1)         # to reproduce the benchmark results, use: (1 2 3 4 5)
options=()        # for RMT source-free without warm-up use: options=("RMT.LAMBDA_CE_SRC 0.0 RMT.NUM_SAMPLES_WARM_UP 0")

ccc=("cfgs/ccc/${method}.yaml")
cifar10=("cfgs/cifar10_c/${method}.yaml")
cifar100=("cfgs/cifar100_c/${method}.yaml")
imagenet_c=("cfgs/imagenet_c/${method}.yaml")
imagenet_others=("cfgs/imagenet_others/${method}.yaml")

# ---continual---
if [[ $setting = "continual" ]]
then
  for var in "${cifar10[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  for var in "${cifar100[@]}"; do
    for seed in ${seeds[*]}; do
      python test_time.py --cfg $var SETTING $setting RNG_SEED $seed $options
    done
  done
  architectures=(resnet50 swin_b vit_b_16)
  for arch in ${architectures[*]}; do
    for var in "${imagenet_c[@]}"; do
      for seed in ${seeds[*]}; do
        python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch RNG_SEED $seed $options
      done
    done
  done
  dataset=(imagenet_r imagenet_k imagenet_d109)
  for arch in ${architectures[*]}; do
    for ds in ${dataset[*]}; do
      for var in "${imagenet_others[@]}"; do
        for seed in ${seeds[*]}; do
          python test_time.py --cfg $var SETTING $setting MODEL.ARCH $arch CORRUPTION.DATASET $ds RNG_SEED $seed $options
        done
      done
    done
  done
fi
