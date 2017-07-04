// Copyright 2004-, Facebook, Inc. All Rights Reserved.

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

  rslt.RHSTokens.clear();
  rslt.LHSTokens.resize(example.LHSTokens.size());
  std::copy(example.LHSTokens.begin(), example.LHSTokens.end(), rslt.LHSTokens.begin());

  assert(example.RHSTokens.size() > 0);
  auto idx = rand() % example.RHSTokens.size();
  if (args_->trainMode == 1) {
    // pick one random rhs as label, and the rest as input
    for (int i = 0; i < example.RHSTokens.size(); i++) {
      auto tok = example.RHSTokens[i];
      if (i == idx) {
        rslt.RHSTokens.push_back(tok);
      } else {
        rslt.LHSTokens.push_back(tok);
      }
    }
  } else {
    // pick one random rhs as label
    rslt.RHSTokens.push_back(example.RHSTokens[idx]);
  }
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
  auto rnd = [&] {
    static __thread unsigned int rState;
    return rand_r(&rState);
  };

  auto& ex = examples_[rnd() % size_];
  int r = rnd() % ex.RHSTokens.size();
  results.clear();
  results.push_back(ex.RHSTokens[r]);
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

} // namespace starspace
