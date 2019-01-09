/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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

/**
* @brief  Main entry-point for this application, for the case of
*  running this test project standalone.
*/
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
