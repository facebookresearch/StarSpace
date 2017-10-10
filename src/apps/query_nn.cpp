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
  sp.initFromSavedModel(args->model);
  cout << "------Loaded model args:\n";
  args->printArgs();

  for(;;) {
    string input;
    cout << "enter some text: ";
    if (!getline(cin, input) || input.size() == 0) break;
    sp.nearestNeighbor(input, k);
  }
  return 0;
}
