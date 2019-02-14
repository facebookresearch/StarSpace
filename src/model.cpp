/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
#include <numeric>


#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

int getNumberOfCores() {
#ifdef WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#elif MACOS
  int nm[2];
  size_t len = 4;
  uint32_t count;

  nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
  sysctl(nm, 2, &count, &len, NULL, 0);

  if (count < 1) {
    nm[1] = HW_NCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);
    if (count < 1) { count = 1; }
  }
  return count;
#else
  return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

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
  assert(a.size() > 0);
  assert(a.size() == b.size());
  return ublas::inner_prod(a, b);
}

Real norm2(Matrix<Real>::Row a) {
  auto retval = norm_2(a);
  return (std::max)(std::numeric_limits<Real>::epsilon(), retval);
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
  const Real negSearchLimit = (std::min)(numSamples,
                                       size_t(args_->negSearchLimit));

  numThreads = (std::max)(numThreads, 2);
  numThreads -= 1; // Withold one thread for the norm thread.
  numThreads = (std::min)(numThreads, int(numSamples));
  vector<Real> losses(numThreads);
  vector<long> counts(numThreads);

  auto trainThread = [&](int idx,
                         vector<int>::const_iterator start,
                         vector<int>::const_iterator end) {
    assert(start >= indices.begin());
    assert(end >= start);
    assert(end <= indices.end());
    bool amMaster = idx == 0;
    auto t_epoch_start = std::chrono::high_resolution_clock::now();
    losses[idx] = 0.0;
    counts[idx] = 0;

    unsigned int batch_sz = args_->batchSize;
    vector<ParseResults> examples;
    for (auto ip = start; ip < end; ip++) {
      auto i = *ip;
      float thisLoss = 0.0;
      if (args_->trainMode == 5 || args_->trainWord) {
        vector<ParseResults> exs;
        data->getWordExamples(i, exs);
        vector<ParseResults> word_exs;
        for (unsigned int i = 0; i < exs.size(); i++) {
          word_exs.push_back(exs[i]);
          if (word_exs.size() >= batch_sz || i == exs.size() - 1) {
            if (args_->loss == "softmax") {
              thisLoss = trainNLLBatch(data, word_exs, negSearchLimit, rate, true);
            } else {
              thisLoss = trainOneBatch(data, word_exs, negSearchLimit, rate, true);
            }
            word_exs.clear();
            assert(thisLoss >= 0.0);
            counts[idx]++;
            losses[idx] += thisLoss;
          }
        }
      }
      if (args_->trainMode != 5) {
        ParseResults ex;
        data->getExampleById(i, ex);
        if (ex.LHSTokens.size() == 0 or ex.RHSTokens.size() == 0) {
          continue;
        }
        examples.push_back(ex);
        if (examples.size() >= batch_sz || (ip + 1) == end) {
          if (args_->loss == "softmax") {
            thisLoss = trainNLLBatch(data, examples, negSearchLimit, rate, false);
          } else {
            thisLoss = trainOneBatch(data, examples, negSearchLimit, rate, false);
          }
          examples.clear();

          assert(thisLoss >= 0.0);
          counts[idx]++;
          losses[idx] += thisLoss;
        }
      }

      // update rate racily.
      if ((i % kDecrStep) == (kDecrStep - 1)) {
        rate -= decrPerKSample;
      }
      auto t_end = std::chrono::high_resolution_clock::now();
      auto tot_spent = std::chrono::duration<double>(t_end-t_start).count();
      if (tot_spent > args_->maxTrainTime) {
        break;
      }
      if (amMaster && ((ip - indices.begin()) % 100 == 99 || (ip + 1) == end)) {

        auto t_epoch_spent =
          std::chrono::duration<double>(t_end-t_epoch_start).count();
        double ex_done_this_epoch = ip - indices.begin();
        int ex_left = ((end - start) * (args_->epoch - epochs_done))
                      - ex_done_this_epoch;
        double ex_done = epochs_done * (end - start) + ex_done_this_epoch;
        double time_per_ex = double(t_epoch_spent) / ex_done_this_epoch;
        int eta = int(time_per_ex * double(ex_left));
        double epoch_progress = ex_done_this_epoch / (end - start);
        double progress = ex_done / (ex_done + ex_left);
        if (eta > args_->maxTrainTime - tot_spent) {
          eta = args_->maxTrainTime - tot_spent;
          progress = tot_spent / (eta + tot_spent);
        }
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
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
  for (size_t i = 0; i < (size_t)numThreads; i++) {
    auto start = i * numPerThread;
    auto end = (std::min)(start + numPerThread, numSamples);
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

float EmbedModel::trainOneBatch(shared_ptr<InternDataHandler> data,
                           const vector<ParseResults>& batch_exs,
                           size_t negSearchLimit,
                           Real rate0,
                           bool trainWord) {

  using namespace boost::numeric::ublas;
  // Keep all the activations on the stack so we can asynchronously
  // update.

  int batch_sz = batch_exs.size();
  std::vector<Matrix<Real>> lhs(batch_sz), rhsP(batch_sz);
  std::vector<Real> posSim(batch_sz);
  std::vector<Real> labelRate(batch_sz, -rate0);

  auto cols = args_->dim;
  for (auto i = 0; i < batch_sz; i++) {
    const auto& items = batch_exs[i].LHSTokens;
    const auto& labels = batch_exs[i].RHSTokens;
    projectLHS(items, lhs[i]);
    check(lhs[i]);

    projectRHS(labels, rhsP[i]);
    check(rhsP[i]);
    posSim[i] = similarity(lhs[i], rhsP[i]);
  }

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
    auto retval = (std::max)((std::min)(val, kMaxLoss), 0.0);
    return retval;
  };

  // Get a random batch of negatives
  std::vector<Matrix<Real>> rhsN(negSearchLimit);
  std::vector<std::vector<Base>> batch_negLabels;

  for (unsigned int i = 0; i < negSearchLimit; i++) {
    std::vector<Base> negLabels;
    if (trainWord) {
      data->getRandomWord(negLabels);
    } else {
      data->getRandomRHS(negLabels);
    }
    projectRHS(negLabels, rhsN[i]);;
    check(rhsN[i]);
    batch_negLabels.push_back(negLabels);
  }

  // Select negative examples
  Real total_loss = 0.0;
  std::vector<Real> loss(batch_sz);
  std::vector<Matrix<Real>> negMean(batch_sz);
  std::vector<int> num_negs(batch_sz);
  std::vector<std::vector<Real>> nRate(batch_sz);

  std::vector<std::vector<bool>> update_flag;
  update_flag.resize(batch_sz);

  for (int i = 0; i < batch_sz; i++) {
    num_negs[i] = 0;
    loss[i] = 0.0;
    negMean[i].matrix = zero_matrix<Real>(1, cols);
    update_flag[i].resize(negSearchLimit, false);
    nRate[i].resize(negSearchLimit, 0);

    for (unsigned int j = 0; j < negSearchLimit; j++) {
      nRate[i][j] = 0.0;
      if (batch_exs[i].RHSTokens == batch_negLabels[j]) {
        continue;
      }
      auto thisLoss = tripleLoss(posSim[i], similarity(lhs[i], rhsN[j]));
      if (thisLoss > 0.0) {
        num_negs[i]++;
        loss[i] += thisLoss;
        negMean[i].add(rhsN[j]);
        assert(loss[i] >= 0.0);
        update_flag[i][j] = true;
        if (num_negs[i] == args_->maxNegSamples) {
          break;
        }
      }
    }
    if (num_negs[i] == 0) {
      continue;
    }
    loss[i] /= negSearchLimit;
    negMean[i].matrix /= num_negs[i];
    total_loss += loss[i];
    // gradW for i
    negMean[i].add(rhsP[i], -1);
    for (unsigned int j = 0; j < negSearchLimit; j++) {
      if (update_flag[i][j]) {
        nRate[i][j] = rate0 / num_negs[i];
      }
    }
  }

  // Couldn't find a negative example given reasonable effort, so
  // give up.
  if (total_loss == 0.0) return 0.0;
  assert(!std::isinf(total_loss));
  if (rate0 == 0.0) return total_loss;

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
  // gradW = \sum_i t_i- - t+. We're done with negMean, so reuse it.

  backward(batch_exs, batch_negLabels,
           negMean, lhs, num_negs,
           rate0, labelRate, nRate);

  return total_loss;
}

void EmbedModel::backward(
    const vector<ParseResults>& batch_exs,
    const vector<vector<Base>>& batch_negLabels,
    vector<Matrix<Real>> gradW,
    vector<Matrix<Real>> lhs,
    const vector<int>& num_negs,
    Real rate_lhs,
    const vector<Real>& rate_rhsP,
    const vector<vector<Real>>& nRate) {

  using namespace boost::numeric::ublas;
  auto cols = args_->dim;

  typedef
    std::function<void(MatrixRow&, const MatrixRow&, Real, Real, std::vector<Real>&, int32_t)>
    UpdateFn;
  std::function<void(MatrixRow&, const MatrixRow&, Real, Real, std::vector<Real>&, int32_t)> updatePlain =
    [&] (MatrixRow& dest,
         const MatrixRow& src,
         Real rate,
         Real weight,
         std::vector<Real>& adagradWeight,
         int32_t idx) {
    dest -= (rate * src);
  };
  std::function<void(MatrixRow&, const MatrixRow&, Real, Real, std::vector<Real>&, int32_t)> updateAdagrad =
    [&] (MatrixRow& dest,
         const MatrixRow& src,
         Real rate,
         Real weight,
         std::vector<Real>& adagradWeight,
         int32_t idx) {
    assert(idx < adagradWeight.size());
    adagradWeight[idx] += weight / cols;
    rate /= sqrt(adagradWeight[idx] + 1e-6);
    updatePlain(dest, src, rate, weight, adagradWeight, idx);
  };

  UpdateFn* update = args_->adagrad ?
    (UpdateFn*)(&updateAdagrad) : (UpdateFn*)(&updatePlain);

  auto batch_sz = batch_exs.size();
  std::vector<Real> n1(batch_sz, 0.0);
  std::vector<Real> n2(batch_sz, 0.0);
  if (args_->adagrad) {
    for (unsigned int i = 0; i < batch_sz; i++) if (num_negs[i] > 0) {
      n1[i] = dot(gradW[i], gradW[i]);
      n2[i] = dot(lhs[i], lhs[i]);
    }
  }
  // Update input items.
  // Update positive example.
  for (unsigned int i = 0; i < batch_sz; i++) if (num_negs[i] > 0) {
    const auto& items = batch_exs[i].LHSTokens;
    const auto& labels = batch_exs[i].RHSTokens;
    for (auto w : items) {
      auto row = LHSEmbeddings_->row(index(w));
      (*update)(row, gradW[i], rate_lhs * weight(w), n1[i], LHSUpdates_, index(w));
    }
    for (auto la : labels) {
      auto row = RHSEmbeddings_->row(index(la));
      (*update)(row, lhs[i], rate_rhsP[i] * weight(la), n2[i], RHSUpdates_, index(la));
    }
  }

  // Update negative example
  for (unsigned int j = 0; j < batch_negLabels.size(); j++) {
    for (unsigned int i = 0; i < batch_sz; i++) if (fabs(nRate[i][j]) > 1e-8) {
      for (auto la : batch_negLabels[j]) {
        auto row = RHSEmbeddings_->row(index(la));
        (*update)(row, lhs[i], nRate[i][j] * weight(la), n2[i], RHSUpdates_, index(la));
      }
    }
  }
}

float EmbedModel::trainNLLBatch(
    shared_ptr<InternDataHandler> data,
    const vector<ParseResults>& batch_exs,
    int32_t negSearchLimit,
    Real rate0,
    bool trainWord) {

  auto batch_sz = batch_exs.size();
  std::vector<Matrix<Real>> lhs(batch_sz), rhsP(batch_sz), rhsN(negSearchLimit);

  using namespace boost::numeric::ublas;

  for (int i = 0; i < batch_sz; i++) {
    const auto& items = batch_exs[i].LHSTokens;
    const auto& labels = batch_exs[i].RHSTokens;
    projectLHS(items, lhs[i]);
    check(lhs[i]);

    projectRHS(labels, rhsP[i]);
    check(rhsP[i]);
  }

  std::vector<std::vector<Real>> prob(batch_sz);
  std::vector<std::vector<Base>> batch_negLabels;
  std::vector<Matrix<Real>> gradW(batch_sz);
  std::vector<Real> loss(batch_sz);

  std::vector<std::vector<Real>> nRate(batch_sz);
  std::vector<int> num_negs(batch_sz, 0);
  std::vector<Real> labelRate(batch_sz);

  Real total_loss = 0.0;

  for (int i = 0; i < negSearchLimit; i++) {
    std::vector<Base> negLabels;
    if (trainWord) {
      data->getRandomWord(negLabels);
    } else {
      data->getRandomRHS(negLabels);
    }
    projectRHS(negLabels, rhsN[i]);
    check(rhsN[i]);
    batch_negLabels.push_back(negLabels);
  }

  for (int i = 0; i < batch_sz; i++) {
    nRate[i].resize(negSearchLimit);
    std::vector<int> index;
    index.clear();

    int cls_cnt = 1;
    prob[i].clear();
    prob[i].push_back(dot(lhs[i], rhsP[i]));
    Real max = prob[i][0];

    for (int j = 0; j < negSearchLimit; j++) {
      nRate[i][j] = 0.0;
      if (batch_negLabels[j] == batch_exs[i].RHSTokens) {
        continue;
      }
      prob[i].push_back(dot(lhs[i], rhsN[j]));
      max = (std::max)(prob[i][0], prob[i][cls_cnt]);
      index.push_back(j);
      cls_cnt += 1;
    }
    loss[i] = 0.0;

    // skip, failed to find any negatives
    if (cls_cnt == 1) {
      continue;
    }

    num_negs[i] = cls_cnt - 1;
    Real base = 0;
    for (int j = 0; j < cls_cnt; j++) {
      prob[i][j] = exp(prob[i][j] - max);
      base += prob[i][j];
    }

    // normalize probabilities
    for (int j = 0; j < cls_cnt; j++) {
      prob[i][j] /= base;
    }

    loss[i] = -log(prob[i][0]);
    total_loss += loss[i];

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

    gradW[i] = rhsP[i];
    gradW[i].matrix *= (prob[i][0] - 1);

    for (int j = 1; j < cls_cnt; j++) {
      auto inj = index[j - 1];
      gradW[i].add(rhsN[inj], prob[i][j]);
      nRate[i][inj] = prob[i][j] * rate0;
    }
    labelRate[i] = (prob[i][0] - 1) * rate0;
  }

  backward(
      batch_exs, batch_negLabels,
      gradW, lhs, num_negs,
      rate0, labelRate, nRate);

  return total_loss;
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

    vector<Cand> mostSimilar((std::min)(numSim, maxn));
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
  if (pieces.size() > (unsigned int)(cols + 1)) {
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
  if (pieces.size() == (unsigned int)cols) {
    cout << "Missing record at line " << lineNum <<
      "; assuming empty string";
    pieces.insert(pieces.begin(), "");
  }
  while (pieces.size() < (unsigned int)(cols + 1)) {
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
  auto numThreads = getNumberOfCores();
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
    for (size_t i = 0; i < (size_t)size; i++) {
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
