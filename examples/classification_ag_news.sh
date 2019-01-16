#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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

MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

echo "Downloading dataset ag_news"
if [ ! -f "${DATADIR}/${DATASET[i]}.train" ]
then
    wget -c "https://dl.fbaipublicfiles.com/starspace/ag_news_csv.tar.gz" -O "${DATADIR}/${DATASET[0]}_csv.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_csv.tar.gz" -C "${DATADIR}"
    cat "${DATADIR}/${DATASET[0]}_csv/train.csv" | normalize_text > "${DATADIR}/${DATASET[0]}.train"
    cat "${DATADIR}/${DATASET[0]}_csv/test.csv" | normalize_text > "${DATADIR}/${DATASET[0]}.test"
  fi

echo "Compiling StarSpace"

make

echo "Start to train on ag_news data:"

./starspace train \
  -trainFile "${DATADIR}"/ag_news.train \
  -model "${MODELDIR}"/ag_news \
  -initRandSd 0.01 \
  -adagrad false \
  -ngrams 1 \
  -lr 0.01 \
  -epoch 5 \
  -thread 20 \
  -dim 10 \
  -negSearchLimit 5 \
  -trainMode 0 \
  -label "__label__" \
  -similarity "dot" \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -model "${MODELDIR}"/ag_news \
  -testFile "${DATADIR}"/ag_news.test \
  -ngrams 1 \
  -dim 10 \
  -label "__label__" \
  -thread 10 \
  -similarity "dot" \
  -trainMode 0 \
  -verbose true

