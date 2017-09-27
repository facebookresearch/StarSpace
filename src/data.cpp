/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "data.h"
#include "utils/utils.h"
#include <string>
#include <vector>
#include <fstream>
#include <assert.h>

using namespace std;

namespace starspace {

InternDataHandler::InternDataHandler(shared_ptr<Args> args) {
  size_ = 0;
  idx_ = -1;
  examples_.clear();
  args_= args;
}

void InternDataHandler::loadFromFile(
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
}

// Convert an example for training/testing if needed.
// In the case of trainMode=1, a random label from r.h.s will be selected
// as label, and the rest of labels from r.h.s. will be input features
void InternDataHandler::convert(
    const ParseResults& example,
    ParseResults& rslt) const {

  rslt.LHSTokens.clear();
  rslt.RHSTokens.clear();

  rslt.LHSTokens.insert(rslt.LHSTokens.end(),
      example.LHSTokens.begin(), example.LHSTokens.end());

  if (args_->trainMode == 0) {
    // lhs is the same, pick one random label as rhs
    assert(example.LHSTokens.size() > 0);
    assert(example.RHSTokens.size() > 0);
    auto idx = rand() % example.RHSTokens.size();
    rslt.RHSTokens.push_back(example.RHSTokens[idx]);
  } else {
    assert(example.RHSTokens.size() > 1);
    if (args_->trainMode == 1) {
      // pick one random label as rhs and the rest is lhs
      auto idx = rand() % example.RHSTokens.size();
      for (int i = 0; i < example.RHSTokens.size(); i++) {
        auto tok = example.RHSTokens[i];
        if (i == idx) {
          rslt.RHSTokens.push_back(tok);
        } else {
          rslt.LHSTokens.push_back(tok);
        }
      }
    } else
    if (args_->trainMode == 2) {
      // pick one random label as lhs and the rest is rhs
      auto idx = rand() % example.RHSTokens.size();
      for (int i = 0; i < example.RHSTokens.size(); i++) {
        auto tok = example.RHSTokens[i];
        if (i == idx) {
          rslt.LHSTokens.push_back(tok);
        } else {
          rslt.RHSTokens.push_back(tok);
        }
      }
    } else
    if (args_->trainMode == 3) {
      // pick two random labels, one as lhs and the other as rhs
      auto idx = rand() % example.RHSTokens.size();
      int idx2;
      do {
        idx2 = rand() % example.RHSTokens.size();
      } while (idx2 == idx);
      rslt.LHSTokens.push_back(example.RHSTokens[idx]);
      rslt.RHSTokens.push_back(example.RHSTokens[idx2]);
    } else
    if (args_->trainMode == 4) {
      // the first one as lhs and the second one as rhs
      rslt.LHSTokens.push_back(example.RHSTokens[0]);
      rslt.RHSTokens.push_back(example.RHSTokens[1]);
    }
  }
}

void InternDataHandler::getWordExamples(
    const vector<int32_t>& doc,
    vector<ParseResults>& rslts) const {

  rslts.clear();
  for (int widx = 0; widx < doc.size(); widx++) {
    ParseResults rslt;
    rslt.LHSTokens.clear();
    rslt.RHSTokens.clear();
    rslt.RHSTokens.push_back(doc[widx]);
    for (int i = max(widx - args_->ws, 0);
         i < min(size_t(widx + args_->ws), doc.size()); i++) {
      if (i != widx) {
        rslt.LHSTokens.push_back(doc[i]);
      }
    }
    rslts.emplace_back(rslt);
  }
}

void InternDataHandler::getWordExamples(
    int idx,
    vector<ParseResults>& rslts) const {

  assert(idx < size_);
  const auto& example = examples_[idx];
  getWordExamples(example.LHSTokens, rslts);
}

void InternDataHandler::addExample(const ParseResults& example) {
  examples_.push_back(example);
  size_++;
}

void InternDataHandler::getExampleById(int32_t idx, ParseResults& rslt) const {
  assert(idx < size_);
  convert(examples_[idx], rslt);
}

void InternDataHandler::getNextExample(ParseResults& rslt) {
  assert(size_ > 0);
  idx_ = idx_ + 1;
  // go back to the beginning of the examples if we reach the end
  if (idx_ >= size_) {
    idx_ = idx_ - size_;
  }
  convert(examples_[idx_], rslt);
}

void InternDataHandler::getRandomExample(ParseResults& rslt) const {
  assert(size_ > 0);
  int32_t idx = rand() % size_;
  convert(examples_[idx], rslt);
}

void InternDataHandler::getKRandomExamples(int K, vector<ParseResults>& c) {
  auto kSamples = min(K, size_);
  for (int i = 0; i < kSamples; i++) {
    ParseResults example;
    getRandomExample(example);
    c.push_back(example);
  }
}

void InternDataHandler::getNextKExamples(int K, vector<ParseResults>& c) {
  auto kSamples = min(K, size_);
  for (int i = 0; i < kSamples; i++) {
    idx_ = (idx_ + 1) % size_;
    ParseResults example;
    convert(examples_[idx_], example);
    c.push_back(example);
  }
}

// Randomly sample one example and randomly sample a label from this example
// The result is usually used as negative samples in training
void InternDataHandler::getRandomRHS(vector<int32_t>& results) const {
  assert(size_ > 0);
  results.clear();
  auto& ex = examples_[rand() % size_];
  if (args_->trainMode == 5) {
    int r = rand() % ex.LHSTokens.size();
    results.push_back(ex.LHSTokens[r]);
  } else {
    int r = rand() % ex.RHSTokens.size();
    if (args_->trainMode == 2) {
      for (int i = 0; i < ex.RHSTokens.size(); i++) {
        if (i != r) {
          results.push_back(ex.RHSTokens[i]);
        }
      }
    } else {
      results.push_back(ex.RHSTokens[r]);
    }
  }
}

void InternDataHandler::save(std::ostream& out) {
  out << "data size : " << size_ << endl;
  for (auto& example : examples_) {
    out << "lhs : ";
    for (auto t : example.LHSTokens) {out << t << ' ';}
    out << endl;
    out << "rhs : ";
    for (auto t : example.RHSTokens) {out << t << ' ';}
    out << endl;
  }
}

} // unamespace starspace
