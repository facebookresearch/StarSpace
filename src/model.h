// Copyright 2004-, Facebook, Inc. All Rights Reserved.
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
typedef boost::numeric::ublas::matrix_row<typeof(Matrix<Real>::matrix)>
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

  // Learning
  typedef std::vector<ParseResults> Corpus;
  float train(std::shared_ptr<InternDataHandler> data,
              int numThreads,
              Real startRate, Real endRate,
              bool verbose = true);

  float test(std::shared_ptr<InternDataHandler> data, int numThreads) {
    return this->train(data, numThreads, 0.0, 0.0, false);
  }

  float trainOne(std::shared_ptr<InternDataHandler> data,
                 const std::vector<int32_t>& items,
                 const std::vector<int32_t>& labels,
                 size_t maxNegSamples,
                 Real rate);

  float trainNLL(std::shared_ptr<InternDataHandler> data,
                 const std::vector<int32_t>& items,
                 const std::vector<int32_t>& labels,
                 int32_t negSearchLimit,
                 Real rate);

  void backward(const std::vector<int32_t>& items,
                const std::vector<int32_t>& labels,
                const std::vector<std::vector<int32_t>>& negLabels,
                Matrix<Real>& gradW,
                Matrix<Real>& lhs,
                Real rate_lhs,
                Real rate_rhsP,
                const std::vector<Real>& rate_rhsN);

  // Querying
  std::vector<std::pair<int32_t, Real>>
    kNN(std::shared_ptr<SparseLinear<Real>> lookup,
        Matrix<Real> point,
        bool isLabel,
        int numSim);

  std::vector<std::pair<int32_t, Real>>
    findLHSLike(Matrix<Real> point, int numSim = 5) {
    return kNN(LHSEmbeddings_, point, false, numSim);
  }

  std::vector<std::pair<int32_t, Real>>
    findRHSLike(Matrix<Real> point, int numSim = 5) {
    return kNN(RHSEmbeddings_, point, true, numSim);
  }

  Matrix<Real>
  projectRHS(std::vector<int32_t> ws) {
    Matrix<Real> retval;
    RHSEmbeddings_->forward(ws, retval);
    if (ws.size()) retval.matrix /= ws.size();
    return retval;
  }

  Matrix<Real>
  projectLHS(std::vector<int32_t> ws) {
    Matrix<Real> retval;
    LHSEmbeddings_->forward(ws, retval);
    if (ws.size()) retval.matrix /= ws.size();
    return retval;
  }

  void projectLHS(std::vector<int32_t> ws, Matrix<Real>& retval) {
    LHSEmbeddings_->forward(ws, retval);
    if (ws.size()) retval.matrix /= ws.size();
  }

  void loadTsv(std::istream& in, const std::string sep = "\t");
  void loadTsv(const char* fname, const std::string sep = "\t");
  void loadTsv(const std::string& fname, const std::string sep = "\t") {
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
    for (int i = 0; i < m.size1(); i++) {
      for (int j = 0; j < m.size2(); j++) {
        assert(!isnan(m(i, j)));
        assert(!isinf(m(i, j)));
      }
    }
  }

};

}