/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "matrix.h"
#include "proj.h"
#include "dict.h"
#include "utils/normalize.h"
#include "utils/args.h"
#include "data.h"
#include "doc_data.h"

#include <fstream>
#include <boost/noncopyable.hpp>
#include <vector>


namespace starspace {

typedef float Real;
typedef boost::numeric::ublas::matrix_row<decltype(Matrix<Real>::matrix)>
  MatrixRow;
typedef boost::numeric::ublas::vector<Real> Vector;

/*
 * The model is basically two lookup tables: one for left hand side
 * (LHS) entities, one for right hand side (RHS) entities.
 */
struct EmbedModel : public boost::noncopyable {
public:
  explicit EmbedModel(std::shared_ptr<Args> args,
                      std::shared_ptr<Dictionary> dict);


  typedef std::vector<ParseResults> Corpus;
  float train(std::shared_ptr<InternDataHandler> data,
              int numThreads,
              std::chrono::time_point<std::chrono::high_resolution_clock> t_start,
	      int epochs_done,
              Real startRate,
	      Real endRate,
              bool verbose = true);

  float test(std::shared_ptr<InternDataHandler> data, int numThreads) {
    return this->train(data, numThreads,
		       std::chrono::high_resolution_clock::now(), 0,
		       0.0, 0.0, false);
  }

  float trainOneBatch(std::shared_ptr<InternDataHandler> data,
                 const std::vector<ParseResults>& batch_exs,
                 size_t negSearchLimits,
                 Real rate,
                 bool trainWord = false);

  float trainNLLBatch(std::shared_ptr<InternDataHandler> data,
                 const std::vector<ParseResults>& batch_exs,
                 int32_t negSearchLimit,
                 Real rate,
                 bool trainWord = false);

  void backward(const std::vector<ParseResults>& batch_exs,
                const std::vector<std::vector<Base>>& negLabels,
                std::vector<Matrix<Real>> gradW,
                std::vector<Matrix<Real>> lhs,
                const std::vector<int>& num_negs,
                Real rate_lhs,
                const std::vector<Real>& rate_rhsP,
                const std::vector<std::vector<Real>>& nRate);

  // Querying
  std::vector<std::pair<int32_t, Real>>
    kNN(std::shared_ptr<SparseLinear<Real>> lookup,
        Matrix<Real> point,
        int numSim);

  std::vector<std::pair<int32_t, Real>>
    findLHSLike(Matrix<Real> point, int numSim = 5) {
    return kNN(LHSEmbeddings_, point, numSim);
  }

  std::vector<std::pair<int32_t, Real>>
    findRHSLike(Matrix<Real> point, int numSim = 5) {
    return kNN(RHSEmbeddings_, point, numSim);
  }

  Matrix<Real> projectRHS(const std::vector<Base>& ws);
  Matrix<Real> projectLHS(const std::vector<Base>& ws);

  void projectLHS(const std::vector<Base>& ws, Matrix<Real>& retval);
  void projectRHS(const std::vector<Base>& ws, Matrix<Real>& retval);

  void loadTsv(std::istream& in, const std::string sep = "\t ");
  void loadTsv(const char* fname, const std::string sep = "\t ");
  void loadTsv(const std::string& fname, const std::string sep = "\t ") {
    return loadTsv(fname.c_str(), sep);
  }

  void saveTsv(std::ostream& out, const char sep = '\t') const;

  void save(std::ostream& out) const;

  void load(std::ifstream& in);

  const std::string& lookupLHS(int32_t idx) const {
    return dict_->getSymbol(idx);
  }
  const std::string& lookupRHS(int32_t idx) const {
    return dict_->getLabel(idx);
  }

  void loadTsvLine(std::string& line, int lineNum, int cols,
                   const std::string sep = "\t");

  std::shared_ptr<Dictionary> getDict() { return dict_; }

  std::shared_ptr<SparseLinear<Real>>& getLHSEmbeddings() {
    return LHSEmbeddings_;
  }
  const std::shared_ptr<SparseLinear<Real>>& getLHSEmbeddings() const {
    return LHSEmbeddings_;
  }
  std::shared_ptr<SparseLinear<Real>>& getRHSEmbeddings() {
    return RHSEmbeddings_;
  }
  const std::shared_ptr<SparseLinear<Real>>& getRHSEmbeddings() const {
    return RHSEmbeddings_;
  }

  void initModelWeights();

  Real similarity(const MatrixRow& a, const MatrixRow& b);
  Real similarity(Matrix<Real>& a, Matrix<Real>& b) {
    return similarity(asRow(a), asRow(b));
  }

  static Real cosine(const MatrixRow& a, const MatrixRow& b);
  static Real cosine(Matrix<Real>& a, Matrix<Real>& b) {
    return cosine(asRow(a), asRow(b));
  }

  static MatrixRow asRow(Matrix<Real>& m) {
    assert(m.numRows() == 1);
    return MatrixRow(m.matrix, 0);
  }

  static void normalize(Matrix<Real>::Row row, double maxNorm = 1.0);
  static void normalize(Matrix<Real>& m) { normalize(asRow(m)); }

private:
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<SparseLinear<Real>> LHSEmbeddings_;
  std::shared_ptr<SparseLinear<Real>> RHSEmbeddings_;
  std::shared_ptr<Args> args_;

  std::vector<Real> LHSUpdates_;
  std::vector<Real> RHSUpdates_;

#ifdef NDEBUG
  static const bool debug = false;
#else
  static const bool debug = false;
#endif

  static void check(const Matrix<Real>& m) {
    m.sanityCheck();
  }

  static void check(const boost::numeric::ublas::matrix<Real>& m) {
    if (!debug) return;
    for (unsigned int i = 0; i < m.size1(); i++) {
      for (unsigned int j = 0; j < m.size2(); j++) {
        assert(!std::isnan(m(i, j)));
        assert(!std::isinf(m(i, j)));
      }
    }
  }

};

}
