/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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
    std::string model;
    std::string fileFormat;
    std::string label;
    std::string basedoc;
    std::string loss;
    std::string similarity;

    double lr;
    double termLr;
    double norm;
    double margin;
    double initRandSd;
    size_t dim;
    int epoch;
    int thread;
    int maxNegSamples;
    int negSearchLimit;
    int minCount;
    int minCountLabel;
    int bucket;
    int ngrams;
    int trainMode;
    bool verbose;
    bool debug;
    bool adagrad;
    bool isTrain;

    void parseArgs(int, char**);
    void printHelp();
    void printArgs();
    void save(std::ostream& out);
    void load(std::istream& in);
};

}
