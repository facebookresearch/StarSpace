/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thread>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

namespace starspace {

using namespace std;
using namespace boost::numeric;

EmbedModel::EmbedModel(
    shared_ptr<Args> args,
    shared_ptr<Dictionary> dict) {

  args_ = args;
  dict_ = dict;

  initModelWeights();
}

void EmbedModel::initModelWeights() {
  assert(dict_ != nullptr);
  size_t num_lhs = dict_->nwords() + dict_->nlabels();

  if (args_->ngrams > 1) {
    num_lhs += args_->bucket;
  }

  LHSEmbeddings_ =
    std::shared_ptr<SparseLinear<Real>>(
      new SparseLinear<Real>({num_lhs, args_->dim},args_->initRandSd)
    );

  if (args_->shareEmb) {
    RHSEmbeddings_ = LHSEmbeddings_;
  } else {
    RHSEmbeddings_ =
      std::shared_ptr<SparseLinear<Real>>(
        new SparseLinear<Real>({num_lhs, args_->dim},args_->initRandSd)
      );
  }

  if (args_->adagrad) {
    LHSUpdates_.resize(LHSEmbeddings_->numRows());
    RHSUpdates_.resize(RHSEmbeddings_->numRows());
  }

  if (args_->verbose) {
    cout << "Initialized model weights. Model size :\n"
         << "matrix : " << LHSEmbeddings_->numRows() << ' '
         << LHSEmbeddings_->numCols() << endl;
  }
}

Real dot(Matrix<Real>::Row a, Matrix<Real>::Row b) {
  const auto dim = a.size();
  assert(dim > 0);
  assert(a.size() == b.size());
  return ublas::inner_prod(a, b);
}

Real norm2(Matrix<Real>::Row a) {
  auto retval = norm_2(a);
  return std::max(std::numeric_limits<Real>::epsilon(), retval);
}

// consistent accessor methods for straight indices and index-weight pairs
int32_t index(int32_t idx) { return idx; }
int32_t index(std::pair<int32_t, Real> idxWeightPair) {
  return idxWeightPair.first;
}

constexpr float weight(int32_t idx) { return 1.0; }
float weight(std::pair<int32_t, Real> idxWeightPair) {
  return idxWeightPair.second;
}

Matrix<Real> EmbedModel::projectRHS(const std::vector<Base>& ws) {
  Matrix<Real> retval;
  projectRHS(ws, retval);
  return retval;
}

Matrix<Real> EmbedModel::projectLHS(const std::vector<Base>& ws) {
  Matrix<Real> retval;
  projectLHS(ws, retval);
  return retval;
}

void EmbedModel::projectLHS(const std::vector<Base>& ws, Matrix<Real>& retval) {
  LHSEmbeddings_->forward(ws, retval);
  if (ws.size()) {
    auto norm = (args_->similarity == "dot") ?
      pow(ws.size(), args_->p) : norm2(retval);
    retval.matrix /= norm;
  }
}

void EmbedModel::projectRHS(const std::vector<Base>& ws, Matrix<Real>& retval) {
  RHSEmbeddings_->forward(ws, retval);
  if (ws.size()) {
    auto norm = (args_->similarity == "dot") ?
      pow(ws.size(), args_->p) : norm2(retval);
    retval.matrix /= norm;
  }
}

Real EmbedModel::trainOneExample(
    shared_ptr<InternDataHandler> data,
    const ParseResults& s,
    int negSearchLimit,
    Real rate,
    bool trainWord) {

  if (s.RHSTokens.size() == 0 || s.LHSTokens.size() == 0) {
    return 0.0;
  }

  if (args_->debug) {
    auto printVec = [&](const vector<Base>& vec) {
      cout << "vec : ";
      for (auto v : vec) {cout << v.first << ':' << v.second << ' ';}
      cout << endl;
    };

    printVec(s.LHSTokens);
    printVec(s.RHSTokens);
    cout << endl;
  }

  Real wRate = s.weight * rate;
  if (args_->loss == "softmax") {
    return trainNLL(
      data,
      s.LHSTokens, s.RHSTokens,
      negSearchLimit, wRate,
      trainWord
    );
  } else {
    // default is hinge loss
    return trainOne(
      data,
      s.LHSTokens, s.RHSTokens,
      negSearchLimit, wRate,
      trainWord
    );
  }
}

Real EmbedModel::train(shared_ptr<InternDataHandler> data,
                       int numThreads,
		       std::chrono::time_point<std::chrono::high_resolution_clock> t_start,
		       int epochs_done,
                       Real rate,
                       Real finishRate,
                       bool verbose) {
  assert(rate >= finishRate);
  assert(rate >= 0.0);

  // Use a layer of indirection when accessing the corpus to allow shuffling.
  auto numSamples = data->getSize();
  vector<int> indices(numSamples);
  {
    int i = 0;
    for (auto& idx: indices) idx = i++;
  }
  std::random_shuffle(indices.begin(), indices.end());

  // Compute word negatives
  if (args_->trainMode == 5 || args_->trainWord) {
    data->initWordNegatives();
  }

  // If we decrement after *every* sample, precision causes us to lose the
  // update.
  const int kDecrStep = 1000;
  auto decrPerKSample = (rate - finishRate) / (numSamples / kDecrStep);
  const Real negSearchLimit = std::min(numSamples,
                                       size_t(args_->negSearchLimit));

  numThreads = std::max(numThreads, 2);
  numThreads -= 1; // Withold one thread for the norm thread.
  numThreads = std::min(numThreads, int(numSamples));
  vector<Real> losses(numThreads);
  vector<long> counts(numThreads);

  auto trainThread = [&](int idx,
                         vector<int>::const_iterator start,
                         vector<int>::const_iterator end) {
    assert(start >= indices.begin());
    assert(end >= start);
    assert(end <= indices.end());
    bool amMaster = idx == 0;
    int64_t elapsed;
    auto t_epoch_start = std::chrono::high_resolution_clock::now();
    losses[idx] = 0.0;
    counts[idx] = 0;
    for (auto ip = start; ip < end; ip++) {
      auto i = *ip;
      float thisLoss = 0.0;
      if (args_->trainMode == 5 || args_->trainWord) {
        vector<ParseResults> exs;
        data->getWordExamples(i, exs);
        for (auto ex : exs) {
          thisLoss = trainOneExample(data, ex, negSearchLimit, rate, true);
          assert(thisLoss >= 0.0);
          counts[idx]++;
          losses[idx] += thisLoss;
        }
      }
      if (args_->trainMode != 5) {
        ParseResults ex;
        data->getExampleById(i, ex);
        thisLoss = trainOneExample(data, ex, negSearchLimit, rate, false);
        assert(thisLoss >= 0.0);
        counts[idx]++;
        losses[idx] += thisLoss;
      }

      // update rate racily.
      if ((i % kDecrStep) == (kDecrStep - 1)) {
        rate -= decrPerKSample;
      }
      if (amMaster && ((ip - indices.begin()) % 100 == 99 || (ip + 1) == end)) {
        auto t_end = std::chrono::high_resolution_clock::now();
        auto t_epoch_spent =
          std::chrono::duration<double>(t_end-t_epoch_start).count();
        double ex_done_this_epoch = ip - indices.begin();
        int ex_left = ((end - start) * (args_->epoch - epochs_done))
                      - ex_done_this_epoch;
        double ex_done = epochs_done * (end - start) + ex_done_this_epoch;
        double time_per_ex = double(t_epoch_spent) / ex_done_this_epoch;
        int eta = int(time_per_ex * double(ex_left));
        auto tot_spent = std::chrono::duration<double>(t_end-t_start).count();
        if (tot_spent > args_->maxTrainTime) {
          break;
        }
        double epoch_progress = ex_done_this_epoch / (end - start);
        double progress = ex_done / (ex_done + ex_left);
        if (eta > args_->maxTrainTime - tot_spent) {
          eta = args_->maxTrainTime - tot_spent;
          progress = tot_spent / (eta + tot_spent);
        }
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        int etas = (eta - etah * 3600 - etam * 60);
        int toth = int(tot_spent) / 3600;
        int totm = (tot_spent - toth * 3600) / 60;
        int tots = (tot_spent - toth * 3600 - totm * 60);
        std::cerr << std::fixed;
        std::cerr << "\rEpoch: " << std::setprecision(1) << 100 * epoch_progress << "%";
        std::cerr << "  lr: " << std::setprecision(6) << rate;
        std::cerr << "  loss: " << std::setprecision(6) << losses[idx] / counts[idx];
        if (eta < 60) {
          std::cerr << "  eta: <1min ";
        } else {
          std::cerr << "  eta: " << std::setprecision(3) << etah << "h" << etam << "m";
        }
        std::cerr << "  tot: " << std::setprecision(3) << toth << "h" << totm << "m"  << tots << "s ";
        std::cerr << " (" << std::setprecision(1) << 100 * progress << "%)";
        std::cerr << std::flush;
      }
    }
  };

  vector<thread> threads;
  bool doneTraining = false;
  size_t numPerThread = ceil(numSamples / numThreads);
  assert(numPerThread > 0);
  for (size_t i = 0; i < numThreads; i++) {
    auto start = i * numPerThread;
    auto end = std::min(start + numPerThread, numSamples);
    assert(end >= start);
    assert(end <= numSamples);
    auto b = indices.begin() + start;
    auto e = indices.begin() + end;
    assert(b >= indices.begin());
    assert(e >= b);
    assert(e <= indices.end());
    threads.emplace_back(thread([=] {
      trainThread(i, b, e);
    }));
  }

  // .. and a norm truncation thread. It's not worth it to slow
  // down every update with truncation, so just work our way through
  // truncating as needed on a separate thread.
  std::thread truncator([&] {
    auto trunc = [](Matrix<Real>::Row row, double maxNorm) {
      auto norm = norm2(row);
      if (norm > maxNorm) {
        row *= (maxNorm / norm);
      }
    };
    for (int i = 0; !doneTraining; i++) {
      auto wIdx = i % LHSEmbeddings_->numRows();
      trunc(LHSEmbeddings_->row(wIdx), args_->norm);
    }
  });
  for (auto& t: threads) t.join();
  // All done. Shut the truncator down.
  doneTraining = true;
  truncator.join();

  Real totLoss = std::accumulate(losses.begin(), losses.end(), 0.0);
  long totCount = std::accumulate(counts.begin(), counts.end(), 0);
  return totLoss / totCount;
}

void EmbedModel::normalize(Matrix<float>::Row row, double maxNorm) {
  auto norm = norm2(row);
  if (norm != maxNorm) { // Not all of them are updated.
    if (norm == 0.0) { // Unlikely!
      norm = 0.01;
    }
    row *= (maxNorm / norm);
  }
}

float EmbedModel::trainOne(shared_ptr<InternDataHandler> data,
                           const vector<Base>& items,
                           const vector<Base>& labels,
                           size_t negSearchLimit,
                           Real rate0,
                           bool trainWord) {
  if (items.size() == 0) return 0.0; // nothing to learn.

  using namespace boost::numeric::ublas;
  // Keep all the activations on the stack so we can asynchronously
  // update.

  Matrix<Real> lhs, rhsP, rhsN;

  projectLHS(items, lhs);
  check(lhs);
  auto cols = lhs.numCols();

  projectRHS(labels, rhsP);
  check(rhsP);

  const auto posSim = similarity(lhs, rhsP);
  Real negSim = std::numeric_limits<Real>::min();

  // Some simple helpers to characterize the current triple we're
  // considering.
  auto tripleLoss = [&] (Real posSim, Real negSim) {
    auto val = args_->margin - posSim + negSim;
    assert(!isnan(posSim));
    assert(!isnan(negSim));
    assert(!isinf(posSim));
    assert(!isinf(negSim));
    // We want the max representable loss to have some wiggle room to
    // compute with.
    const auto kMaxLoss = 10e7;
    auto retval = std::max(std::min(val, kMaxLoss), 0.0);
    return retval;
  };

  // Select negative examples
  Real loss = 0.0;
  std::vector<Matrix<Real>> negs;
  std::vector<std::vector<Base>> negLabelsBatch;
  Matrix<Real> negMean;
  negMean.matrix = zero_matrix<Real>(1, cols);

  for (int i = 0; i < negSearchLimit &&
                  negs.size() < args_->maxNegSamples; i++) {

    std::vector<Base> negLabels;
    do {
      if (trainWord) {
        data->getRandomWord(negLabels);
      } else {
        data->getRandomRHS(negLabels);
      }
    } while (negLabels == labels);

    projectRHS(negLabels, rhsN);

    check(rhsN);
    auto thisLoss = tripleLoss(posSim, similarity(lhs, rhsN));
    if (thisLoss > 0.0) {
      loss += thisLoss;
      negs.emplace_back(rhsN);
      negLabelsBatch.emplace_back(negLabels);
      negMean.add(rhsN);
      assert(loss >= 0.0);
    }
  }
  loss /= negSearchLimit;
  negMean.matrix /= negs.size();

  // Couldn't find a negative example given reasonable effort, so
  // give up.
  if (negs.size() == 0) return 0.0;
  assert(!std::isinf(loss));
  if (rate0 == 0.0) return loss;

  // Let w be the average of the input features, t+ be the positive
  // example and t- be the average of the negative examples.
  // Our error E is:
  //
  //    E = k - dot(w, t+) + dot(w, t-)
  //
  // Differentiating term-by-term we get:
  //
  //     dE / dw  = t- - t+
  //     dE / dt- = w
  //     dE / dt+ = -w
  //
  // This is the innermost loop, so cache misses count. Please do some perf
  // testing if you end up modifying it.

  // gradW = \sum_i t_i- - t+. We're done with negMean, so reuse it.
  auto gradW = negMean;
  gradW.add(rhsP, -1);
  auto nRate = rate0 / negs.size();
  std::vector<Real> negRate(negs.size());
  std::fill(negRate.begin(), negRate.end(), nRate);

  backward(items, labels, negLabelsBatch,
           gradW, lhs,
           rate0, -rate0, negRate);

  return loss;
}

float EmbedModel::trainNLL(shared_ptr<InternDataHandler> data,
                           const vector<Base>& items,
                           const vector<Base>& labels,
                           int32_t negSearchLimit,
                           Real rate0,
                           bool trainWord) {
  if (items.size() == 0) return 0.0; // nothing to learn.
  Matrix<Real> lhs, rhsP, rhsN;

  using namespace boost::numeric::ublas;

  projectLHS(items, lhs);
  check(lhs);

  projectRHS(labels, rhsP);
  check(rhsP);

  // label is treated as class 0
  auto numClass = args_->negSearchLimit + 1;
  std::vector<Real> prob(numClass);
  std::vector<Matrix<Real>> negClassVec;
  std::vector<std::vector<Base>> negLabelsBatch;

  prob[0] = dot(lhs, rhsP);
  Real max = prob[0];

  for (int i = 1; i < numClass; i++) {
    std::vector<Base> negLabels;
    do {
      if (trainWord) {
        data->getRandomWord(negLabels);
      } else {
        data->getRandomRHS(negLabels);
      }
    } while (negLabels == labels);
    projectRHS(negLabels, rhsN);
    check(rhsN);
    negClassVec.push_back(rhsN);
    negLabelsBatch.push_back(negLabels);

    prob[i] = dot(lhs, rhsN);
    max = std::max(prob[i], max);
  }

  Real base = 0;
  for (int i = 0; i < numClass; i++) {
    prob[i] = exp(prob[i] - max);
    base += prob[i];
  }

  // normalize the probabilities
  for (int i = 0; i < numClass; i++) { prob[i] /= base; };

  Real loss = - log(prob[0]);

  // Let w be the average of the words in the post, t+ be the
  // positive example (the tag the post has) and t- be the average
  // of the negative examples (the tags we searched for with submarginal
  // separation above).
  // Our error E is:
  //
  //    E = - log P(t+)
  //
  // Where P(t) = exp(dot(w, t)) / (\sum_{t'} exp(dot(w, t')))
  //
  // Differentiating term-by-term we get:
  //
  //    dE / dw = t+ (P(t+) - 1)
  //    dE / dt+ = w (P(t+) - 1)
  //    dE / dt- = w P(t-)

  auto gradW = rhsP;
  gradW.matrix *= (prob[0] - 1);
  for (int i = 0; i < numClass - 1; i++) {
    gradW.add(negClassVec[i], prob[i + 1]);
  }

  std::vector<Real> negRate(numClass - 1);
  for (int i = 0; i < negRate.size(); i++) {
    negRate[i] = prob[i + 1] * rate0;
  }

  backward(items, labels, negLabelsBatch,
           gradW, lhs,
           rate0, (prob[0] - 1 ) * rate0, negRate);

  return loss;
}

void EmbedModel::backward(
    const vector<Base>& items,
    const vector<Base>& labels,
    const vector<vector<Base>>& negLabels,
    Matrix<Real>& gradW,
    Matrix<Real>& lhs,
    Real rate_lhs,
    Real rate_rhsP,
    const vector<Real>& rate_rhsN) {

  using namespace boost::numeric::ublas;
  auto cols = lhs.numCols();

  typedef
    std::function<void(MatrixRow&, const MatrixRow&, Real, Real, std::vector<Real>&, int32_t)>
    UpdateFn;
  auto updatePlain   = [&] (MatrixRow& dest, const MatrixRow& src,
                            Real rate,
                            Real weight,
                            std::vector<Real>& adagradWeight,
                            int32_t idx) {
    dest -= (rate * src);
  };
  auto updateAdagrad = [&] (MatrixRow& dest, const MatrixRow& src,
                            Real rate,
                            Real weight,
                            std::vector<Real>& adagradWeight,
                            int32_t idx) {
    assert(idx < adagradWeight.size());
    adagradWeight[idx] += weight / cols;
    rate /= sqrt(adagradWeight[idx] + 1e-6);
    updatePlain(dest, src, rate, weight, adagradWeight, idx);
  };

  auto update = args_->adagrad ?
    UpdateFn(updateAdagrad) : UpdateFn(updatePlain);

  Real n1 = 0, n2 = 0;
  if (args_->adagrad) {
    n1 = dot(gradW, gradW);
    n2 = dot(lhs, lhs);
  }

  // Update input items.
  for (auto w : items) {
    auto row = LHSEmbeddings_->row(index(w));
    update(row, gradW, rate_lhs * weight(w), n1, LHSUpdates_, index(w));
  }

  // Update positive example.
  for (auto la : labels) {
    auto row = RHSEmbeddings_->row(index(la));
    update(row, lhs, rate_rhsP * weight(la), n2, RHSUpdates_, index(la));
  }

  // Update negative example.
  for (size_t i = 0; i < negLabels.size(); i++) {
    for (auto la : negLabels[i]) {
      auto row = RHSEmbeddings_->row(index(la));
      update(row, lhs, rate_rhsN[i] * weight(la), n2, RHSUpdates_, index(la));
    }
  }
}

Real EmbedModel::similarity(const MatrixRow& a, const MatrixRow& b) {
  auto retval = (args_->similarity == "dot") ? dot(a, b) : cosine(a, b);
  assert(!isnan(retval));
  assert(!isinf(retval));
  return retval;
}

Real EmbedModel::cosine(const MatrixRow& a, const MatrixRow& b) {
  auto normA = dot(a, a), normB = dot(b, b);
  if (normA == 0.0 || normB == 0.0) {
    return 0.0;
  }
  return dot(a, b) / sqrt(normA * normB);
}

vector<pair<int32_t, Real>>
EmbedModel::kNN(shared_ptr<SparseLinear<Real>> lookup,
                Matrix<Real> point,
                int numSim) {

    typedef pair<int32_t, Real> Cand;
    int  maxn = dict_->nwords() + dict_->nlabels();

    vector<Cand> mostSimilar(std::min(numSim, maxn));
    for (auto& s: mostSimilar) {
      s = { -1, -1.0 };
    }
    auto resort = [&] {
      std::sort(mostSimilar.begin(), mostSimilar.end(),
               [&](Cand a, Cand b) { return a.second > b.second; });
    };
    Matrix<Real> contV;

    for (int i = 0; i < maxn; i++) {
      lookup->forward(i, contV);
      Real sim = (args_->similarity == "dot") ?
          dot(point, contV) : cosine(point, contV);
      if (sim > mostSimilar.back().second) {
        mostSimilar.back() = { i, sim };
        resort();
      }
    }
    for (auto r : mostSimilar) {
      if (r.first == -1 || r.second == -1.0) {
        abort();
      }
    }
    return mostSimilar;
}

void EmbedModel::loadTsvLine(string& line, int lineNum,
                             int cols, const string sep) {
  vector<string> pieces;
  static const string zero = "0.0";
  // Strip trailing spaces
  while (line.size() && isspace(line[line.size() - 1])) {
    line.resize(line.size() - 1);
  }
  boost::split(pieces, line, boost::is_any_of(sep));
  if (pieces.size() > cols + 1) {
    cout << "Hmm, truncating long (" << pieces.size() <<
        ") record at line " << lineNum;
    if (true) {
      for (size_t i = cols; i < pieces.size(); i++) {
        cout << "Warning excess fields " << pieces[i]
                      << "; misformatted file?";
      }
    }
    pieces.resize(cols + 1);
  }
  if (pieces.size() == cols) {
    cout << "Missing record at line " << lineNum <<
      "; assuming empty string";
    pieces.insert(pieces.begin(), "");
  }
  while (pieces.size() < cols + 1) {
    cout << "Zero-padding short record at line " << lineNum;
    pieces.push_back(zero);
  }
  auto idx = dict_->getId(pieces[0]);
  if (idx == -1) {
    if (pieces[0].size() > 0) {
      cerr << "Failed to insert record: " << line << "\n";
    }
    return;
  }
  auto row = LHSEmbeddings_->row(idx);
  for (int i = 0; i < cols; i++) {
    row(i) = boost::lexical_cast<Real>(pieces[i + 1].c_str());
  }
}

void EmbedModel::loadTsv(const char* fname, const string sep) {
  cout << "Loading model from file " << fname << endl;
  auto cols = args_->dim;

  std::ifstream ifs(fname);
  auto filelen = [&](ifstream& f) {
    auto pos = f.tellg();
    f.seekg(0, ios_base::end);
    auto retval = f.tellg();
    f.seekg(pos, ios_base::beg);
    return retval;
  };

  auto len = filelen(ifs);
  auto numThreads = sysconf(_SC_NPROCESSORS_ONLN);
  vector<off_t> partitions(numThreads + 1);
  partitions[0] = 0;
  partitions[numThreads] = len;

  string unused;
  for (int i = 1; i < numThreads; i++) {
    ifs.seekg((len / numThreads) * i);
    getline(ifs, unused);
    partitions[i] = ifs.tellg();
  }

  // It's possible that the ranges in partitions overlap; consider,
  // e.g., a machine with 100 hardware threads and only 99 lines
  // in the file. In this case, we'll do some excess work but loadTsvLine
  // is idempotent, so it is ok.
  std::vector<thread> threads;
  for (int i = 0; i < numThreads; i++) {
    auto body = [this, fname, cols, sep, i, &partitions]() {
      // Get our own seek pointer.
      ifstream ifs(fname);
      ifs.seekg(partitions[i]);
      string line;
      while (ifs.tellg() < partitions[i + 1] && getline(ifs, line)) {
        // We don't know the line number. Super-bummer.
        loadTsvLine(line, -1, cols, sep);
      }
    };
    threads.emplace_back(body);
  }
  for (auto& t: threads) {
    t.join();
  }

  cout << "Model loaded.\n";
}

void EmbedModel::loadTsv(istream& in, const string sep) {
  auto cols = LHSEmbeddings_->numCols();
  assert(RHSEmbeddings_->numCols() == cols);

  string line;
  int lineNum = 0;
  while (getline(in, line)) {
    lineNum++;
    loadTsvLine(line, lineNum, cols, sep);
  }
}

void EmbedModel::saveTsv(ostream& out, const char sep) const {
  auto dumpOne = [&](shared_ptr<SparseLinear<Real>> emb) {
    auto size =  dict_->nwords() + dict_->nlabels();
    for (size_t i = 0; i < size; i++) {
      // Skip invalid IDs.
      string symbol = dict_->getSymbol(i);
      out << symbol;
      emb->forRow(i,
                 [&](Real r, size_t j) {
        out << sep << r;
      });
      out << "\n";
    }
  };
  dumpOne(LHSEmbeddings_);
}

void EmbedModel::save(ostream& out) const {
  LHSEmbeddings_->write(out);
  if (!args_->shareEmb) {
    RHSEmbeddings_->write(out);
  }
}

void EmbedModel::load(ifstream& in) {
  LHSEmbeddings_.reset(new SparseLinear<Real>(in));
  if (args_->shareEmb) {
    RHSEmbeddings_ = LHSEmbeddings_;
  } else {
    RHSEmbeddings_.reset(new SparseLinear<Real>(in));
  }
}

}
