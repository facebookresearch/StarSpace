#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

DATASET=(
  ag_news
)

ID=(
  0Bz8a_Dbh9QhbUDNpeUdjb0wxRms # ag_news
)

RESULTDIR=result
DATADIR=data

mkdir -p "${RESULTDIR}"
mkdir -p "${DATADIR}"

echo "Downloading dataset ag_news"
if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
then
    wget -c "https://drive.google.com/uc?export=download&id=${ID[0]}" -O "${DATADIR}/${DATASET[0]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[0]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[0]}.train"
    cat "${DATADIR}/${DATASET[0]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[0]}.test"
  fi

echo "Compiling StarSpace"

make starspace

echo "Start to train on ag_news data:"

./starspace train \
  -trainFile '/home/ledell/fspace/data/ag_news.train' \
  -model '/home/ledell/fspace/model/ag_news' \
  -initRandSd 0.01 \
  -adagrad false \
  -ngrams 1 \
  -lr 0.01 \
  -epoch 5 \
  -thread 20 \
  -dim 10 \
  -negSearchLimit 5 \
  -maxNegSamples 3 \
  -trainMode 0 \
  -label "__label__" \
  -similarity "dot" \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -model '/home/ledell/fspace/model/ag_news' \
  -testFile '/home/ledell/fspace/data/ag_news.test' \
  -ngrams 1 \
  -dim 10 \
  -label "__label__" \
  -thread 10 \
  -similarity "dot" \
  -trainmode 0 \
  -verbose true
