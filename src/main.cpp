// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "starspace.h"
#include <iostream>

using namespace std;
using namespace starspace;

int main(int argc, char** argv) {
  shared_ptr<Args> args = make_shared<Args>();
  args->parseArgs(argc, argv);
  args->printArgs();

  StarSpace sp(args);
  if (args->isTrain) {
    sp.init();
    sp.train();
    sp.saveModel();
  } else {
    sp.initFromSavedModel();
    sp.evaluate();
  }

  return 0;
}
