/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The SparseLinear class implements the lookup tables used in starspace model.

#pragma once

#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <string.h>
#include <fstream>

namespace starspace {

template<typename Real = float>
struct SparseLinear : public Matrix<Real> {
  explicit SparseLinear(MatrixDims dims,
                        Real sd = 1.0) : Matrix<Real>(dims, sd) { }

  explicit SparseLinear(std::ifstream& in) : Matrix<Real>(in) { }

  void forward(int in, Matrix<Real>& mout) {
    using namespace boost::numeric::ublas;
    const auto c = this->numCols();
    mout.matrix.resize(1, c);
    memcpy(&mout[0][0], &(*this)[in][0], c * sizeof(Real));
  }

  void forward(const std::vector<int>& in, Matrix<Real>& mout) {
    using namespace boost::numeric::ublas;
    const auto c = this->numCols();
    mout.matrix = zero_matrix<Real>(1, c);
    auto outRow = mout.row(0);
    for (const auto& elt: in) {
      assert(elt < this->numRows());
      outRow += this->row(elt);
    }
  }

  void forward(const std::vector<std::pair<int, Real>>& in,
               Matrix<Real> &mout) {
    using namespace boost::numeric::ublas;
    const auto c = this->numCols();
    mout.matrix = zero_matrix<Real>(1, c);
    auto outRow = mout.row(0);
    for (const auto& pair: in) {
      assert(pair.first < this->numRows());
      outRow += this->row(pair.first) * pair.second;
    }
  }

  void backward(const std::vector<int>& in,
                const Matrix<Real>& mb, const Real alpha) {
    // Just update this racily and in-place.
    assert(mb.numRows() == 1);
    auto b = mb[0];
    for (const auto& elt: in) {
      auto row = (*this)[elt];
      for (int i = 0; i < this->numCols(); i++) {
        row[i] -= alpha * b[i];
      }
    }
  }

  Real* allocOutput() {
    Real* retval;
    auto val = posix_memalign((void**)&retval, Matrix<Real>::kAlign,
                              this->numCols() * sizeof(Real));
    if (val != 0) {
      perror("could not allocate output");
      throw this;
    }
    return retval;
  }
};

}
