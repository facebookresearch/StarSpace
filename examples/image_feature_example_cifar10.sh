#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET=cifar10

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/${DATASET}_resnext.train" ]
then
    echo "Downloading cifar-10 data with last layer feature from a trained ResNext model"
    wget -c "https://dl.fbaipublicfiles.com/starspace/cifar10-resnext.train" -O "${DATADIR}/${DATASET}_resnext.train"
    wget -c "https://dl.fbaipublicfiles.com/starspace/cifar10-resnext.test" -O "${DATADIR}/${DATASET}_resnext.test"
  fi
    
echo "Compiling StarSpace"

make

echo "Start to train on cifar-10 data:"

./starspace train \
  -trainFile "${DATADIR}"/"${DATASET}"_resnext.train \
  -model "${MODELDIR}"/cifar10_example \
  -useWeight true \
  -initRandSd 0.1 \
  -adagrad false \
  -lr 0.001 \
  -epoch 5 \
  -thread 40 \
  -dim 100 \
  -negSearchLimit 5 \
  -thread 40 \
  -dim 100 \
  -similarity "dot" \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -testFile "${DATADIR}"/"${DATASET}"_resnext.test \
  -model "${MODELDIR}"/cifar10_example \
  -thread 20 \
  -similarity "dot" \
  -verbose true
