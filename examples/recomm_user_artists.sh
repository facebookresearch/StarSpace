#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This scripts provides an example of user/artists recommendation on the
# Last.FM (http://www.lastfm.com) dataset


MODELDIR=/tmp/starspace/models
DATADIR=/tmp/starspace/data
DATASET=lastfm

mkdir -p "${MODELDIR}/${DATASET}"
mkdir -p "${DATADIR}/${DATASET}"

convert_data() {
    PREV_ID=0
    SET=""

    while read -r line 
    do
        read USER_ID ARTIST_ID COUNT <<< $line
        if [ $PREV_ID == $USER_ID ] 
        then
            SET="$SET A$ARTIST_ID"
        else
            if [ $PREV_ID != 0 ]
            then 
                echo $SET
            fi
            SET="A$ARTIST_ID"
            PREV_ID=$USER_ID
        fi
    done < "$1"
    echo $SET
}

echo "Downloading dataset lastFM"
if [ ! -f "${DATADIR}/${DATASET}/user_artists.train" ]
then
    wget -c "files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip" -O "${DATADIR}/${DATASET}/lastfm.zip"
    unzip "${DATADIR}/${DATASET}/lastfm.zip" -d "${DATADIR}/${DATASET}"

    echo "Converting data to StarSpace format ..."

    INPUT_FILE="${DATADIR}/${DATASET}/user_artists.dat"
    TEMP_FILE="${DATADIR}/${DATASET}/temp"
    OUTPUT_FILE="${DATADIR}/${DATASET}/user_artists"

    convert_data ${INPUT_FILE} > ${TEMP_FILE}
    split -d -l 1500 ${TEMP_FILE} ${OUTPUT_FILE}
    mv "${DATADIR}/${DATASET}/user_artists00" "${DATADIR}/${DATASET}/user_artists.train"
    mv "${DATADIR}/${DATASET}/user_artists01" "${DATADIR}/${DATASET}/user_artists.test"
fi

echo "Compiling StarSpace"

make

echo "Start to train on lastfm data:"

./starspace train \
  -trainFile ${DATADIR}/${DATASET}/user_artists.train \
  -model ${MODELDIR}/${DATASET}/user_artists \
  -initRandSd 0.01 \
  -lr 0.01 \
  -epoch 100 \
  -thread 40 \
  -dim 100 \
  -maxNegSamples 100 \
  -negSearchLimit 100 \
  -trainMode 1 \
  -label "A" \
  -verbose true

echo "Start to evaluate trained model:"

./starspace test \
  -model ${MODELDIR}/${DATASET}/user_artists \
  -testFile ${DATADIR}/${DATASET}/user_artists.test \
  -dim 100 \
  -label "A" \
  -thread 40 \
  -trainMode 1 \
  -K 10 \
  -verbose true \
  -predictionFile artist_recs
