/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

 /**
  * Mostly a collection of convenience routines around ublas.
  * We avoid doing any actual compute-intensive work in this file.
  */

#pragma once

#include <math.h>
#include <iostream>
#include <functional>
#include <random>
#include <thread>
#include <algorithm>
#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace starspace {

struct MatrixDims {
  size_t r, c;
  size_t numElts() const { return r * c; }
  bool operator==(const MatrixDims& rhs) {
    return r == rhs.r && c == rhs.c;
  }
};

template<typename Real = float>
struct Matrix {
  static const int kAlign = 64;
  boost::numeric::ublas::matrix<Real> matrix;

  explicit Matrix(MatrixDims dims,
                  Real sd = 1.0) :
    matrix(dims.r, dims.c)
  {
    assert(matrix.size1() == dims.r);
    assert(matrix.size2() == dims.c);
    if (sd > 0.0) {
      randomInit(sd);
    }
  }

  explicit Matrix(const std::vector<std::vector<Real>>& init) {
    size_t rows = init.size();
    size_t maxCols = 0;
    for (const auto& r : init) {
      maxCols = std::max(maxCols, r.size());
    }
    alloc(rows, maxCols);
    for (size_t i = 0; i < numRows(); i++) {
      size_t j;
      for (j = 0; j < init[i].size(); j++) {
        (*this)[i][j] = init[i][j];
      }
      for (; j < numCols(); j++) {
        (*this)[i][j] = 0.0;
      }
    }
  }

  explicit Matrix(std::istream& in) {
    in >> matrix;
  }

  Matrix() {
    alloc(0, 0);
  }

  Real* operator[](size_t i) {
    assert(i >= 0);
    assert(i < numRows());
    return &matrix(i, 0);
  }

  const Real* operator[](size_t i) const {
    assert(i >= 0);
    assert(i < numRows());
    return &matrix(i, 0);
  }

  Real& cell(size_t i, size_t j) {
    assert(i >= 0);
    assert(i < numRows());
    assert(j < numCols());
    assert(j >= 0);
    return matrix(i, j);
  }

  void add(const Matrix<Real>& rhs, Real scale = 1.0) {
    matrix += scale * rhs.matrix;
  }

  void forEachCell(std::function<void(Real&)> l) {
    for (size_t i = 0; i < numRows(); i++)
      for (size_t j = 0; j < numCols(); j++)
        l(matrix(i, j));
  }

  void forEachCell(std::function<void(Real)> l) const {
    for (size_t i = 0; i < numRows(); i++)
      for (size_t j = 0; j < numCols(); j++)
        l(matrix(i, j));
  }

  void forEachCell(std::function<void(Real&, size_t, size_t)> l) {
    for (size_t i = 0; i < numRows(); i++)
      for (size_t j = 0; j < numCols(); j++)
        l(matrix(i, j), i, j);
  }

  void forEachCell(std::function<void(Real, size_t, size_t)> l) const {
    for (size_t i = 0; i < numRows(); i++)
      for (size_t j = 0; j < numCols(); j++)
        l(matrix(i, j), i, j);
  }

  void sanityCheck() const {
#ifndef NDEBUG
    forEachCell([&](Real r, size_t i, size_t j) {
      assert(!std::isnan(r));
      assert(!std::isinf(r));
    });
#endif
  }

  void forRow(size_t r, std::function<void(Real&, size_t)> l) {
    for (size_t j = 0; j < numCols(); j++) l(matrix(r, j), j);
  }

  void forRow(size_t r, std::function<void(Real, size_t)> l) const {
    for (size_t j = 0; j < numCols(); j++) l(matrix(r, j), j);
  }

  void forCol(size_t c, std::function<void(Real&, size_t)> l) {
    for (size_t i = 0; i < numRows(); i++) l(matrix(i, c), i);
  }

  void forCol(size_t c, std::function<void(Real, size_t)> l) const {
    for (size_t i = 0; i < numRows(); i++) l(matrix(c, i), i);
  }

  static void mul(const Matrix& l, const Matrix& r, Matrix& dest) {
    dest.matrix = boost::numeric::ublas::prod(l.matrix, r.matrix);
  }

  void updateRow(size_t r, Matrix& addend, Real scale = 1.0) {
    using namespace boost::numeric::ublas;
    assert(addend.numRows() == 1);
    assert(addend.numCols() == numCols());
    row(r) += Row { addend.matrix, 0 } * scale;
  }

  typedef boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<Real>>
     Row;
  Row row(size_t r) { return Row{ matrix, r }; }

  /* implicit */ operator Row() {
    assert(numRows() == 1);
    return Row{ matrix, 0 };
  }

  size_t numElts() const { return numRows() * numCols(); }
  size_t numRows() const { return matrix.size1(); }
  size_t numCols() const { return matrix.size2(); }
  MatrixDims getDims() const { return { numRows(), numCols() }; }

  void reshape(MatrixDims dims) {
    if (dims == getDims()) return;
    alloc(dims.r, dims.c);
  }

  typedef size_t iterator;
  iterator begin() { return 0; }
  iterator end() { return numElts(); }

  void write(std::ostream& out) {
    out << matrix;
  }

  void randomInit(Real sd = 1.0) {
    if (numElts() > 0) {
      // Multi-threaded initialization brings debug init time down
      // from minutes to seconds.
      auto d = &matrix(0, 0);
      std::minstd_rand gen;
      auto nd = std::normal_distribution<Real>(0, sd);
      for (size_t i = 0; i < numElts(); i++) {
        d[i] = nd(gen);
      };
    }
  }

  private:
  void alloc(size_t r, size_t c) {
    matrix = boost::numeric::ublas::matrix<Real>(r, c);
  }
};

}
