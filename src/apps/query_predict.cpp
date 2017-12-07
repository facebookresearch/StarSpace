/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "../starspace.h"
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;
using namespace starspace;

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();
  if (argc < 3) {
    cerr << "usage: " << argv[0] << " <model> k [basedoc]\n";
    return 1;
  }
  std::string model(argv[1]);
  args->K = atoi(argv[2]);
  args->model = model;
  if (argc > 3) {
    args->fileFormat = "labelDoc";
    args->basedoc = argv[3];
  }

  StarSpace sp(args);
  if (boost::algorithm::ends_with(args->model, ".tsv")) {
    sp.initFromTsv(args->model);
  } else {
    sp.initFromSavedModel(args->model);
    cout << "------Loaded model args:\n";
    args->printArgs();
  }
  // Set dropout probability to 0 in test case.
  sp.args_->dropoutLHS = 0.0;
  sp.args_->dropoutRHS = 0.0;
  // Load basedocs which are set of possible things to predict.
  sp.loadBaseDocs();

  for(;;) {
    string input;
    cout << "Enter some text: ";
    if (!getline(cin, input) || input.size() == 0) break;
    // Do the prediction
    vector<Base> query_vec;
    sp.parseDoc(input, query_vec, " ");
    vector<Predictions> predictions;
    sp.predictOne(query_vec, predictions);
    for (int i = 0; i < predictions.size(); i++) {
      cout << i << "[" << predictions[i].first << "]: ";
      sp.printDoc(cout, sp.baseDocs_[predictions[i].second]);
    }
    cout << "\n";
  }

  return 0;
}
