// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "proj.h"
#include <gtest/gtest.h>

using namespace std;
using namespace starspace;

TEST(Proj, forward) {
  SparseLinear<float> sl({5, 1});
  vector<int> inputs = { 1 ,
                         4 };
  Matrix<float> output;
  sl.forward(inputs, output);
  EXPECT_FLOAT_EQ(output[0][0], sl[1][0] + sl[4][0]);
}

TEST(Proj, empty) {
  SparseLinear<float> sl({5, 1});
  vector<int> inputs = { };
  Matrix<float> output;
  sl.forward(inputs, output);
  output.forEachCell([&](float& f, int i, int j) {
    EXPECT_EQ(f, 0.0);
  });
}

