/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "doc_data.h"
#include "utils/utils.h"
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>

using namespace std;

namespace starspace {

LayerDataHandler::LayerDataHandler(shared_ptr<Args> args) :
  InternDataHandler(args) {
}

void LayerDataHandler::loadFromFile(
  const string& fileName,
  shared_ptr<DataParser> parser) {

  ifstream fin(fileName);
  if (!fin.is_open()) {
    std::cerr << fileName << " cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  fin.close();

  cout << "Loading data from file : " << fileName << endl;
  vector<Corpus> corpora(args_->thread);
  foreach_line(
    fileName,
    [&](std::string& line) {
      auto& corpus = corpora[getThreadID()];
      ParseResults example;
      if (parser->parse(line, example)) {
        corpus.push_back(example);
      }
    },
    args_->thread
  );
  // Glue corpora together.
  auto totalSize = std::accumulate(corpora.begin(), corpora.end(), size_t(0),
                     [](size_t l, Corpus& r) { return l + r.size(); });
  size_t destCursor = examples_.size();
  examples_.resize(totalSize + examples_.size());
  for (const auto &subcorp: corpora) {
    std::copy(subcorp.begin(), subcorp.end(), examples_.begin() + destCursor);
    destCursor += subcorp.size();
  }
  cout << "Total number of examples loaded : " << examples_.size() << endl;
  size_ = examples_.size();
  if (size_ == 0) {
    std::cerr << "ERROR: File '" << fileName << "' is empty." << std::endl;
    exit(EXIT_FAILURE);
  }
}

void LayerDataHandler::insert(
    vector<int32_t>& rslt,
    const vector<int32_t>& ex,
    float dropout) const {

  if (dropout < 1e-8) {
    // if dropout is not enabled, copy all elements
    rslt.insert(rslt.end(), ex.begin(), ex.end());
  } else {
    // dropout enabled
    auto rnd = [&] {
      static __thread unsigned int rState;
      return rand_r(&rState);
    };
    for (const auto& it : ex) {
      auto p = (double)(rnd()) / RAND_MAX;
      if (p > dropout) {
        rslt.push_back(it);
      }
    }
  }
}

void LayerDataHandler::getWordExamples(
    int idx,
    vector<ParseResults>& rslts) const {

  assert(idx < size_);
  const auto& example = examples_[idx];
  assert(example.RHSFeatures.size() > 0);

  // take one random sentence and train on word
  auto r = rand() % example.RHSFeatures.size();
  InternDataHandler::getWordExamples(example.RHSFeatures[r], rslts);
}

void LayerDataHandler::convert(
    const ParseResults& example,
    ParseResults& rslt) const {

  rslt.LHSTokens.clear();
  rslt.RHSTokens.clear();

  if (args_->trainMode == 0) {
    assert(example.LHSTokens.size() > 0);
    assert(example.RHSFeatures.size() > 0);
    insert(rslt.LHSTokens, example.LHSTokens, args_->dropoutLHS);
    auto idx = rand() % example.RHSFeatures.size();
    insert(rslt.RHSTokens, example.RHSFeatures[idx], args_->dropoutRHS);
  } else {
    assert(example.RHSFeatures.size() > 1);
    if (args_->trainMode == 1) {
      // pick one random rhs as label, the rest becomes lhs features
      auto idx = rand() % example.RHSFeatures.size();
      for (int i = 0; i < example.RHSFeatures.size(); i++) {
        if (i == idx) {
          insert(rslt.RHSTokens, example.RHSFeatures[i], args_->dropoutRHS);
        } else {
          insert(rslt.LHSTokens, example.RHSFeatures[i], args_->dropoutLHS);
        }
      }
    } else
    if (args_->trainMode == 2) {
      // pick one random rhs as lhs, the rest becomes rhs features
      auto idx = rand() % example.RHSFeatures.size();
      for (int i = 0; i < example.RHSFeatures.size(); i++) {
        if (i == idx) {
          insert(rslt.LHSTokens, example.RHSFeatures[i], args_->dropoutLHS);
        } else {
          insert(rslt.RHSTokens, example.RHSFeatures[i], args_->dropoutRHS);
        }
      }
    } else
    if (args_->trainMode == 3) {
      // pick one random rhs as input
      auto idx = rand() % example.RHSFeatures.size();
      insert(rslt.LHSTokens, example.RHSFeatures[idx], args_->dropoutLHS);
      // pick another random rhs as label
      int idx2;
      do {
        idx2 = rand() % example.RHSFeatures.size();
      } while (idx == idx2);
      insert(rslt.RHSTokens, example.RHSFeatures[idx2], args_->dropoutRHS);
    } else
    if (args_->trainMode == 4) {
      // the first one as lhs and the second one as rhs
      insert(rslt.LHSTokens, example.RHSFeatures[0], args_->dropoutLHS);
      insert(rslt.RHSTokens, example.RHSFeatures[1], args_->dropoutRHS);
    }
  }
}

void LayerDataHandler::getRandomRHS(vector<int32_t>& result) const {
  assert(size_ > 0);
  auto& ex = examples_[rand() % size_];
  int r = rand() % ex.RHSFeatures.size();

  result.clear();
  if (args_->trainMode == 2) {
    // pick one random, the rest is rhs features
    for (int i = 0; i < ex.RHSFeatures.size(); i++) {
      if (i != r) {
        insert(result, ex.RHSFeatures[i], args_->dropoutRHS);
      }
    }
  } else
  if (args_->trainMode == 5) {
    // pick random word
    int wid = rand() % ex.RHSFeatures[r].size();
    result.push_back(ex.RHSFeatures[r][wid]);
  } else {
    insert(result, ex.RHSFeatures[r], args_->dropoutRHS);
  }
}

void LayerDataHandler::save(ostream& out) {
  for (auto example : examples_) {
    out << "lhs: ";
    for (auto t : example.LHSTokens) {
      out << t << ' ';
    }
    out << "\nrhs: ";
    for (auto feat : example.RHSFeatures) {
      for (auto r : feat) { cout << r << ' '; }
      out << "\t";
    }
    out << endl;
  }
}

} // namespace starspace
