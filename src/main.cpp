/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
      if (boost::algorithm::ends_with(args->initModel, ".tsv")) {
        sp.initFromTsv(args->initModel);
      } else {
        sp.initFromSavedModel(args->initModel);
        cout << "------Loaded model args:\n";
        args->printArgs();
      }
    } else {
      sp.init();
    }
    sp.train();
    sp.saveModel(args->model);
    sp.saveModelTsv(args->model + ".tsv");
  } else {
    if (boost::algorithm::ends_with(args->model, ".tsv")) {
      sp.initFromTsv(args->model);
    } else {
      sp.initFromSavedModel(args->model);
      cout << "------Loaded model args:\n";
      args->printArgs();
    }
    sp.evaluate();
  }

  return 0;
}
