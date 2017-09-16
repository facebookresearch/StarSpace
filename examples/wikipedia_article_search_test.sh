#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

DATASET=(
  wikipedia
)

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

echo "Downloading dataset ag_news"
if [ ! -f "${DATADIR}/${DATASET[i]}_train250k.txt" ]
then
    wget -c "https://s3.amazonaws.com/fair-data/starspace/wikipedia_train250k.tgz" -O "${DATADIR}/${DATASET[0]}.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}.tar.gz" -C "${DATADIR}"
  fi

echo "Compiling StarSpace"

make

echo "Start to train on wikipedia data (small version):"

./starspace test \
  -testFile "${DATADIR}"/wikipedia_test10k.txt \
  -basedoc "${DATADIR}"/wikipedia_test_basedocs.txt \
  -model "${MODELDIR}"/wikipedia_article_search \
  -thread 20 \
  -trainMode 2 \
  -verbose true
