/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "starspace.h"
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;
using namespace starspace;

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();
  args->parseArgs(argc, argv);
  args->printArgs();

  StarSpace sp(args);
  if (args->isTrain) {
    if (!args->initModel.empty()) {
      sp.initFromSavedModel(args->initModel);
      cout << "------Loaded model args:\n";
      args->printArgs();
    } else {
      sp.init();
    }
    sp.train();
    sp.saveModel();
    sp.saveModelTsv();
  } else {
    if (boost::algorithm::ends_with(args->model, ".tsv")) {
      sp.initFromTsv();
    } else {
      sp.initFromSavedModel(args->model);
      cout << "------Loaded model args:\n";
      args->trainMode = 3;
      args->printArgs();
    }
    sp.evaluate();
  }

  return 0;
}
