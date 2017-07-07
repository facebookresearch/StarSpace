/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <iostream>
#include <string>
#include <cstring>
#include <assert.h>

using namespace std;

namespace starspace {

Args::Args() {
  lr = 0.01;
  termLr = 1e-9;
  norm = 1.0;
  margin = 0.05;
  initRandSd = 0.001;
  dim = 10;
  epoch = 5;
  thread = 10;
  maxNegSamples = 10;
  negSearchLimit = 50;
  minCount = 1;
  minCountLabel = 1;
  verbose = false;
  debug = false;
  adagrad = false;
  trainMode = 0;
  basedoc = "";
  fileFormat = "fastText";
  label = "__label__";
  bucket = 2000000;
  ngrams = 1;
  loss = "hinge";
  similarity = "cosine";
}

void Args::parseArgs(int argc, char** argv) {
  if (argc <= 1) {
    cerr << "Usage: need to specify whether it is train or test.\n";
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (strcmp(argv[1], "train") == 0) {
    isTrain = true;
  } else if (strcmp(argv[1], "test") == 0) {
    isTrain = false;
  } else {
    cerr << "Usage: the first argument should be either train or test.\n";
    printHelp();
    exit(EXIT_FAILURE);
  }
  int i = 2;
  while (i < argc) {
    if (argv[i][0] != '-') {
      cout << "Provided argument without a dash! Usage:" << endl;
      printHelp();
      exit(EXIT_FAILURE);
    }

    if (strcmp(argv[i], "-trainFile") == 0) {
      trainFile = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-validationFile") == 0) {
      validationFile = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-testFile") == 0) {
      testFile = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-basedoc") == 0) {
      basedoc = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-model") == 0) {
      model = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-fileFormat") == 0) {
      fileFormat = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-label") == 0) {
      label = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-loss") == 0) {
      loss = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-similarity") == 0) {
      similarity = string(argv[i + 1]);
    } else if (strcmp(argv[i], "-lr") == 0) {
      lr = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-termLr") == 0) {
      termLr = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-norm") == 0) {
      norm = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-margin") == 0) {
      margin = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-initRandSd") == 0) {
      initRandSd = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-dim") == 0) {
      dim = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-epoch") == 0) {
      epoch = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-thread") == 0) {
      thread = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-maxNegSamples") == 0) {
      maxNegSamples = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-negSearchLimit") == 0) {
      negSearchLimit = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-minCount") == 0) {
      minCount = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-minCountLabel") == 0) {
      minCountLabel = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-bucket") == 0) {
      bucket = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-ngrams") == 0) {
      ngrams = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-trainMode") == 0) {
      trainMode = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-verbose") == 0) {
      verbose = (string(argv[i + 1]) == "true");
    } else if (strcmp(argv[i], "-debug") == 0) {
      debug = (string(argv[i + 1]) == "true");
    } else if (strcmp(argv[i], "-adagrad") == 0) {
      adagrad = (string(argv[i + 1]) == "true");
    }
    i += 2;
  }
  if (isTrain) {
    if (trainFile.empty() || model.empty()) {
      cout << "Empty train file or output model path." << endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  } else {
    if (testFile.empty() || model.empty()) {
      cout << "Empty test file or model path." << endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
  // check for trainMode
  if (!(trainMode == 0 || trainMode == 1)) {
    cerr << "Uknown trainMode. trainMode should either be 0 or 1.\n";
    exit(EXIT_FAILURE);
  }
  // check for loss type
  if (!(loss == "hinge" || loss == "softmax")) {
    cerr << "Unsupported loss type: " << loss << endl;
    exit(EXIT_FAILURE);
  }
  // check for similarity type
  if (!(similarity == "cosine" || similarity == "dot")) {
    cerr << "Unsupported similarity type. Should be either dot or cosine.\n";
    exit(EXIT_FAILURE);
  }
  // check for file format
  if (!(fileFormat == "fastText" || fileFormat == "labelDoc")) {
    cerr << "Unsupported file format type. Should be either fastText or labelDoc.\n";
    exit(EXIT_FAILURE);
  }
}

void Args::printHelp() {
  cout << "\n"
       << "The following arguments are mandatory for train: \n"
       << "  -trainFile       training file path\n"
       << "  -model           output model file path\n\n"
       << "The following arguments are mandatory for eval: \n"
       << "  -testFile        test file path\n"
       << "  -model           model file path\n\n"
       << "The following arguments for the dictionary are optional:\n"
       << "  -minCount        minimal number of word occurences [" << minCount << "]\n"
       << "  -minCountLabel   minimal number of label occurences [" << minCountLabel << "]\n"
       << "  -ngrams          max length of word ngram [" << ngrams << "]\n"
       << "  -bucket          number of buckets [" << bucket << "]\n"
       << "  -label           labels prefix [" << label << "]\n"
       << "\nThe following arguments for training are optional:\n"
       << "  -trainMode       takes value in [0, 1], see Training Mode Section. [" << trainMode << "]\n"
       << "  -fileFormat      currently support 'fastText' and 'labelDoc', see File Format Section. [" << fileFormat << "]\n"
       << "  -lr              learning rate [" << lr << "]\n"
       << "  -dim             size of embedding vectors [" << dim << "]\n"
       << "  -epoch           number of epochs [" << epoch << "]\n"
       << "  -negiSearchLimit number of negatives sampled [" << negSearchLimit << "]\n"
       << "  -maxNegSamples   max number of negatives in a batch update [" << maxNegSamples << "]\n"
       << "  -loss            loss function {hinge, softmax} [hinge]\n"
       << "  -margin          margin parameter in hinge loss. It's only effective if hinge loss is used. [" << margin << "]\n"
       << "  -similarity      takes value in [cosine, dot]. Whether to use cosine or dot product as similarity function in  hinge loss.\n"
       << "                   It's only effective if hinge loss is used. [" << similarity << "]\n"
       << "  -thread          number of threads [" << thread << "]\n"
       << "  -adagrad         whether to use adagrad in training [" << adagrad << "]\n"
       <<  "\nThe following arguments are optional:\n"
       << "  -verbose         verbosity level [" << verbose << "]\n"
       << "  -debug           whether it's in debug mode [" << debug << "]\n"
       << std::endl;
}

void Args::printArgs() {
  cout << "Arguments: \n"
       << "lr: " << lr << endl
       << "dim: " << dim << endl
       << "epoch: " << epoch << endl
       << "loss: " << loss << endl
       << "margin: " << margin << endl
       << "similarity: " << similarity << endl
       << "maxNegSamples: " << maxNegSamples << endl
       << "negSearchLimit: " << negSearchLimit << endl
       << "thread: " << thread << endl
       << "minCount: " << minCount << endl
       << "minCountLabel: " << minCountLabel << endl
       << "label: " << label << endl
       << "ngrams: " << ngrams << endl
       << "bucket: " << bucket << endl
       << "adagrad: " << adagrad << endl
       << "trainMode: " << trainMode << endl
       << "fileFormat: " << fileFormat << endl;
}

void Args::save(std::ostream& out) {
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(minCountLabel), sizeof(int));
  out.write((char*) &(maxNegSamples), sizeof(int));
  out.write((char*) &(negSearchLimit), sizeof(int));
  out.write((char*) &(ngrams), sizeof(int));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(trainMode), sizeof(int));
  size_t size = fileFormat.size();
  out.write((char*) &(size), sizeof(size_t));
  out.write((char*) &(fileFormat[0]), size);
}

void Args::load(std::istream& in) {
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(minCountLabel), sizeof(int));
  in.read((char*) &(maxNegSamples), sizeof(int));
  in.read((char*) &(negSearchLimit), sizeof(int));
  in.read((char*) &(ngrams), sizeof(int));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(trainMode), sizeof(int));
  size_t size;
  in.read((char*) &(size), sizeof(size_t));
  fileFormat.resize(size);
  in.read((char*) &(fileFormat[0]), size);
}

}
