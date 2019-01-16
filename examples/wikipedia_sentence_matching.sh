#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET=(
  wikipedia
)

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/${DATASET[i]}_train250k.txt" ]
then
    echo "Downloading wikipedia data"
    wget -c "https://dl.fbaipublicfiles.com/starspace/wikipedia_train250k.tgz" -O "${DATADIR}/${DATASET[0]}_train.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_train.tar.gz" -C "${DATADIR}"
    wget -c "https://dl.fbaipublicfiles.com/starspace/wikipedia_devtst.tgz" -O "${DATADIR}/${DATASET[0]}_test.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_test.tar.gz" -C "${DATADIR}"
    wget -c "https://dl.fbaipublicfiles.com/starspace/wikipedia_shuf_test_basedocs_tm3.txt" -O "${DATADIR}/${DATASET[0]}_test_basedocs_tm3.txt"
  fi
    
echo "Compiling StarSpace"

make

echo "Start to train on wikipedia data (small training set example version, not the same as the paper which takes longer to run on a bigger training set):"

./starspace train \
  -trainFile "${DATADIR}"/wikipedia_train250k.txt \
  -model "${MODELDIR}"/wikipedia_sentence_match \
  -trainMode 3 \
  -initRandSd 0.01 \
  -adagrad true \
  -ngrams 1 \
  -lr 0.05 \
  -epoch 5 \
  -thread 20 \
  -dim 100 \
  -negSearchLimit 10 \
  -fileFormat labelDoc \
  -similarity "cosine" \
  -minCount 5 \
  -normalizeText true \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -testFile "${DATADIR}"/wikipedia_test10k.txt \
  -basedoc "${DATADIR}"/wikipedia_test_basedocs_tm3.txt \
  -model "${MODELDIR}"/wikipedia_sentence_match \
  -thread 20 \
  -trainMode 3 \
  -normalizeText true \
  -verbose true
