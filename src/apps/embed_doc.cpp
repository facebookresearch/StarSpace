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

// Read each sentence / document line by line,
// and output it's embedding vector
void embedDoc(StarSpace& sp, istream& fin) {
  string input;
  while (getline(fin, input)) {
    if (input.size() ==0) break;
    cout << input << endl;
    auto vec = sp.getDocVector(input);
    vec.forEachCell([&](Real r) { cout << r << ' '; });
    cout << endl;
  }
}

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();

  if (argc < 2) {
    cerr << "usage: " << argv[0] << " <model> [filename]\n";
    cerr << "if filename is specified, it reads each line from the file and"
         << "output corresponding vectors";
    return 1;
  }

  std::string model(argv[1]);
  args->model = model;

  StarSpace sp(args);
  sp.initFromSavedModel(args->model);
  // set useWeight by default.
  // use 1.0 for default weight if weight is not found
  args->useWeight = true;

  if (argc > 2) {
    std::string filename(argv[2]);
    ifstream fin(filename);
    if (!fin.is_open()) {
      std::cerr << "file cannot be opened for loading!" << std::endl;
      exit(EXIT_FAILURE);
    }
    embedDoc(sp, fin);
    fin.close();
  } else {
    cout << "Input your sentence / document now:\n";
    embedDoc(sp, cin);
  }

  return 0;
}
