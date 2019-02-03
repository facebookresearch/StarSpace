/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "../starspace.h"
#include <iostream>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;
using namespace starspace;

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();

  if (argc < 2) {
    cerr << "usage: " << argv[0] << " <model> [k]\n";
    return 1;
  }

  std::string model(argv[1]);
  args->model = model;

  int k = 5;
  if (argc > 2) {
    k = atoi(argv[2]);
  }
  StarSpace sp(args);
  if (boost::algorithm::ends_with(args->model, ".tsv")) {
    sp.initFromTsv(args->model);
  } else {
    sp.initFromSavedModel(args->model);
  }
  cout << "------Loaded model args:\n";
  args->printArgs();

  for(;;) {
    string input;
    cout << "Enter some text: ";
    if (!getline(cin, input) || input.size() == 0) break;
    sp.nearestNeighbor(input, k);
  }
  return 0;
}
