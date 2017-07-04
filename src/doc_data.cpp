// Copyright 2004-, Facebook, Inc. All Rights Reserved.

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

  cout << "Loading layered data from file : " << fileName << endl;
  vector<Corpus> corpora(args_->thread);
  foreach_line(
    fileName,
    [&](std::string& line) {
      auto& corpus = corpora[getThreadID()];
      ParseResults example;
      vector<vector<int32_t>> RHSFeatures;
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

void LayerDataHandler::convert(
    const ParseResults& example,
    ParseResults& rslt) const {

  assert(example.RHSFeatures.size() > 0);
  rslt.LHSTokens.resize(example.LHSTokens.size());
  std::copy(example.LHSTokens.begin(), example.LHSTokens.end(), rslt.LHSTokens.begin());

  // pick a random rhs as rhs
  auto idx = rand() % example.RHSFeatures.size();
  auto& res = example.RHSFeatures[idx];
  rslt.RHSTokens.resize(res.size());
  std::copy(res.begin(), res.end(), rslt.RHSTokens.begin());
  if (args_->trainMode == 1) {
    // the rest becomes lhs feature
    for (int i = 0; i < example.RHSFeatures.size(); i++) {
      if (i == idx) {continue; }
      auto& res = example.RHSFeatures[i];
      rslt.LHSTokens.insert(rslt.LHSTokens.end(), res.begin(), res.end());
    }
  }
}

void LayerDataHandler::getRandomRHS(vector<int32_t>& result) const {
  assert(size_ > 0);
  auto& ex = examples_[rand() % size_];
  int r = rand() % ex.RHSFeatures.size();

  auto& res = ex.RHSFeatures[r];
  result.resize(res.size());
  std::copy(res.begin(), res.end(), result.begin());
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
