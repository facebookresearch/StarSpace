/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <string>

namespace starspace {

class Args {
  public:
    Args();
    std::string trainFile;
    std::string validationFile;
    std::string testFile;
    std::string predictionFile;
    std::string model;
    std::string initModel;
    std::string fileFormat;
    std::string compressFile;
    std::string label;
    std::string basedoc;
    std::string loss;
    std::string similarity;

    char weightSep;
    double lr;
    double termLr;
    double norm;
    double margin;
    double initRandSd;
    double p;
    double dropoutLHS;
    double dropoutRHS;
    double wordWeight;
    size_t dim;
    int epoch;
    int ws;
    int maxTrainTime;
    int validationPatience;
    int thread;
    int maxNegSamples;
    int negSearchLimit;
    int minCount;
    int minCountLabel;
    int bucket;
    int ngrams;
    int trainMode;
    int K;
    int batchSize;
    int numGzFile;
    bool verbose;
    bool debug;
    bool adagrad;
    bool isTrain;
    bool normalizeText;
    bool saveEveryEpoch;
    bool saveTempModel;
    bool shareEmb;
    bool useWeight;
    bool trainWord;
    bool excludeLHS;

    void parseArgs(int, char**);
    void printHelp();
    void printArgs();
    void save(std::ostream& out);
    void load(std::istream& in);
    bool isTrue(std::string arg);
};

}
