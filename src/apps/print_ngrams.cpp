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

  StarSpace sp(args);
  sp.initFromSavedModel(args->model);
  if (args->ngrams == 1) {
    std::cerr << "Error: your provided model does not use ngram.\n";
    exit(EXIT_FAILURE);
  }

  string input;
  while (getline(cin, input)) {
    auto vec = sp.getNgramVector(input);
    cout << input;
    for (auto v : vec) { cout << "\t" << v; }
    cout << endl;
  }

  return 0;
}
