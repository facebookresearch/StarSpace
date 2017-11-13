#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

DATASET=cifar10

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/${DATASET}_resnext.train" ]
then
    echo "Downloading cifar-10 data with last layer feature from a trained ResNext model"
    wget -c "https://s3.amazonaws.com/fair-data/starspace/cifar10-resnext.train" -O "${DATADIR}/${DATASET}_resnext.train"
    wget -c "https://s3.amazonaws.com/fair-data/starspace/cifar10-resnext.test" -O "${DATADIR}/${DATASET}_resnext.test"
  fi
    
echo "Compiling StarSpace"

make

echo "Start to train on cifar-10 data:"

./starspace train \
  -trainFile "${DATADIR}"/"${DATASET}"_resnext.train \
  -model "${MODELDIR}"/cifar10_example \
  -useWeight true \
  -initRandSd 0.01 \
  -adagrad false \
  -lr 0.0001 \
  -epoch 50 \
  -thread 40 \
  -dim 20 \
  -negSearchLimit 10 \
  -thread 40 \
  -dim 20 \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -testFile "${DATADIR}"/"${DATASET}"_resnext.test \
  -model "${MODELDIR}"/cifar10_example \
  -thread 20 \
  -verbose true
