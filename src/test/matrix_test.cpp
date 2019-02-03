/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "../matrix.h"

using namespace starspace;

TEST(Matrix, init) {
  srand(12);
  Matrix<float> mtx {
    { { 0.01f, 2.23f, 3.34f },
      { 1.11f, -0.4f, 0.2f } } };
  EXPECT_EQ(mtx.numCols(), 3);
  EXPECT_EQ(mtx.numRows(), 2);
  float tot = 0.0;
  mtx.forRow(1, [&](float& f, int c) {
    ASSERT_TRUE(c == 0 || c == 1 || c == 2);
    if (c == 0) EXPECT_FLOAT_EQ(f, 1.11);
    if (c == 1) EXPECT_FLOAT_EQ(f, -0.4);
    if (c == 2) EXPECT_FLOAT_EQ(f, 0.2);
  });

  mtx.forCol(2, [&](float& f, int r) {
    ASSERT_TRUE(r == 0 || r == 1);
    if (r == 0) EXPECT_FLOAT_EQ(f, 3.34);
    if (r == 1) EXPECT_FLOAT_EQ(f, 0.2);
  });
}

TEST(Matrix, mulI) {
  Matrix<float> I4 {
    { { 1.0, 0.0, 0.0, 0.0, },
      { 0.0, 1.0, 0.0, 0.0, },
      { 0.0, 0.0, 1.0, 0.0, },
      { 0.0, 0.0, 0.0, 1.0 } } };

  for (int i = 0; i < 22; i++) {
    size_t otherDim = 1 + rand() % 17;
    Matrix<float> l({otherDim, 4});
    Matrix<float> result({otherDim, 4});
    Matrix<float>::mul(l, I4, result);
    result.forEachCell([&](float& f, int i, int j) {
      // EXPECT_FLOAT_EQ(result[i][j], l[i][j]);
    });
  }
}

TEST(Matrix, mulRand) {
  Matrix<double> A {
    { { -0.2, 0.3,  0.4 },
      { 0.2,  0.2,  -0.001 },
      { 0.3,  0.5,  1 },
      { 1,    2,    3 },
      { -2,   -1,   0 },
      { 0.3,  0.5,  1 },
      { 7,   -0.01, -7 } } };

  Matrix<double> B {
    { { 1,    2,  3, 4 },
      { -2,  -1,  0, 1 },
      { 0.01, 10, 0.3, 2} } };

  Matrix<double> C;
  Matrix<double> expectedC {
    { { -0.796,    3.3,  -0.48,    0.3   },
      { -0.20001,  0.19,  0.5997,  0.998 },
      { -0.69,    10.1,   1.2,     3.7   },
      { -2.97,    30.0,   3.9,    12.0   },
      {  0.0,     -3.0,  -6.0,    -9.0   },
      { -0.69,    10.1,   1.2,     3.7   },
      { 6.95,    -55.99,  18.9,   13.99   } } };
  Matrix<double>::mul(A, B, C);

  C.forEachCell([&](double d, int i, int j) {
    EXPECT_FLOAT_EQ(expectedC[i][j], d);
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
