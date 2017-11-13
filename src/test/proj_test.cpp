/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "../proj.h"
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

TEST(Proj, weightedForward) {
  SparseLinear<float> sl({5, 1});
  vector<pair<int,float>> inputs = { {1, 0.5} ,
                                     {4, 1.5} };
  Matrix<float> output;
  sl.forward(inputs, output);
  EXPECT_FLOAT_EQ(output[0][0], sl[1][0] * 0.5 + sl[4][0] * 1.5);
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
