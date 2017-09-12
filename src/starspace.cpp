/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "starspace.h"
#include <iostream>
#include <queue>
#include <unordered_set>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace starspace {

StarSpace::StarSpace(shared_ptr<Args> args)
  : args_(args)
  , dict_(nullptr)
  , parser_(nullptr)
  , trainData_(nullptr)
  , validData_(nullptr)
  , testData_(nullptr)
  , model_(nullptr)
  {}

void StarSpace::initParser() {
  if (args_->fileFormat == "fastText") {
    parser_ = make_shared<DataParser>(dict_, args_);
  } else if (args_->fileFormat == "labelDoc") {
    parser_ = make_shared<LayerDataParser>(dict_, args_);
  } else {
    cerr << "Unsupported file format. Currently support: fastText or labelDoc.\n";
    exit(EXIT_FAILURE);
  }
}

void StarSpace::initDataHandler() {
  if (args_->isTrain) {
    trainData_ = initData();
    trainData_->loadFromFile(args_->trainFile, parser_);
    // set validation data
    if (!args_->validationFile.empty()) {
      validData_ = initData();
      validData_->loadFromFile(args_->validationFile, parser_);
    }
  } else {
    if (args_->testFile != "") {
      testData_ = initData();
      testData_->loadFromFile(args_->testFile, parser_);
    }
  }
}

shared_ptr<InternDataHandler> StarSpace::initData() {
  if (args_->fileFormat == "fastText") {
    return make_shared<InternDataHandler>(args_);
  } else if (args_->fileFormat == "labelDoc") {
    return make_shared<LayerDataHandler>(args_);
  } else {
    cerr << "Unsupported file format. Currently support: fastText or labelDoc.\n";
    exit(EXIT_FAILURE);
  }
  return nullptr;
}

// initialize dict and load data
void StarSpace::init() {
  cout << "Start to initialize starspace model.\n";
  assert(args_ != nullptr);

  // build dict
  initParser();
  dict_ = make_shared<Dictionary>(args_);
  auto filename = args_->trainFile;
  dict_->readFromFile(filename, parser_);
  parser_->resetDict(dict_);
  if (args_->debug) {dict_->save(cout);}

  // init train data class
  trainData_ = initData();
  trainData_->loadFromFile(args_->trainFile, parser_);

  // init model with args and dict
  model_ = make_shared<EmbedModel>(args_, dict_);

  // set validation data
  if (!args_->validationFile.empty()) {
    validData_ = initData();
    validData_->loadFromFile(args_->validationFile, parser_);
  }
}

void StarSpace::initFromSavedModel(const string& filename) {
  cout << "Start to load a trained starspace model.\n";
  std::ifstream in(filename, std::ifstream::binary);
  if (!in.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  string magic;
  char c;
  while ((c = in.get()) != 0) {
    magic.push_back(c);
  }
  cout << magic << endl;
  if (magic != kMagic) {
    std::cerr << "Magic signature does not match!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // load args
  args_->load(in);

  // init and load dict
  dict_ = make_shared<Dictionary>(args_);
  dict_->load(in);

  // init and load model
  model_ = make_shared<EmbedModel>(args_, dict_);
  model_->load(in);
  cout << "Model loaded.\n";

  // init data parser
  initParser();
  initDataHandler();
}

void StarSpace::initFromTsv(const string& filename) {
  cout << "Start to load a trained embedding model in tsv format.\n";
  assert(args_ != nullptr);
  ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in.close();

  // build dict
  dict_ = make_shared<Dictionary>(args_);
  dict_->loadDictFromModel(filename);
  if (args_->debug) {dict_->save(cout);}

  // load Model
  model_ = make_shared<EmbedModel>(args_, dict_);
  model_->loadTsv(filename, "\t ");

  // init data parser
  initParser();
  initDataHandler();
}

void StarSpace::train() {
  float rate = args_->lr;
  float decrPerEpoch = (rate - 1e-9) / args_->epoch;

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < args_->epoch; i++) {
    cout << "Training epoch " << i << ": " << rate << ' ' << decrPerEpoch << endl;
    auto err = model_->train(trainData_, args_->thread,
			     t_start,  i,
			     rate, rate - decrPerEpoch);
    printf("\n ---+++ %20s %4d Train error : %3.8f +++--- %c%c%c\n",
           "Epoch", i, err,
           0xe2, 0x98, 0x83);
    if (validData_ != nullptr) {
      auto valid_err = model_->test(validData_, args_->thread);
      cout << "Validation error: " << valid_err << endl;
    }
    rate -= decrPerEpoch;

    auto t_end = std::chrono::high_resolution_clock::now();
    auto tot_spent = std::chrono::duration<double>(t_end-t_start).count();
    if (tot_spent >args_->maxTrainTime) {
      cout << "MaxTrainTime exceeded." << endl;
      break;
    }
  }
}

void StarSpace::parseDoc(
    const string& line,
    vector<int32_t>& ids,
    const string& sep) {

  vector<string> tokens;
  boost::split(tokens, line, boost::is_any_of(string(sep)));
  parser_->parse(tokens, ids);
}

Matrix<Real> StarSpace::getDocVector(const string& line, const string& sep) {
  vector<int32_t> ids;
  parseDoc(line, ids, sep);
  return model_->projectRHS(ids);
}

void StarSpace::nearestNeighbor(const string& line, int k) {
  auto vec = getDocVector(line, " ");
  auto preds = model_->findLHSLike(vec, k);
  for (auto n : preds) {
    cout << dict_->getSymbol(n.first) << ' ' << n.second << endl;
  }
}

void StarSpace::loadBaseDocs() {
  if (args_->basedoc.empty()) {
    if (args_->fileFormat == "labelDoc") {
      std::cerr << "Must provide base labels when label is featured.\n";
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dict_->nlabels(); i++) {
      baseDocs_.push_back({ i + dict_->nwords() });
      baseDocVectors_.push_back(model_->projectRHS({i + dict_->nwords()}));
    }
  } else {
    cout << "Loading base docs from file : " << args_->basedoc << endl;
    ifstream fin(args_->basedoc);
    if (!fin.is_open()) {
      std::cerr << "Base doc file cannot be opened for loading!" << std::endl;
      exit(EXIT_FAILURE);
    }
    string line;
    while (getline(fin, line)) {
      vector<int32_t> ids;
      parseDoc(line, ids, "\t ");
      baseDocs_.push_back(ids);
      auto docVec = model_->projectRHS(ids);
      baseDocVectors_.push_back(docVec);
    }
    fin.close();
    cout << "Finished loading base docs.\n";
  }
}

Metrics StarSpace::evaluateOne(
    const vector<int32_t>& lhs,
    const vector<int32_t>& rhs,
    vector<Predictions>& pred) {

  std::priority_queue<Predictions> heap;

  auto lhsM = model_->projectLHS(lhs);
  auto rhsM = model_->projectRHS(rhs);
  // Our evaluation function currently assumes there is only one correct label.
  // TODO: generalize this to the multilabel case.
  auto score = model_->similarity(lhsM, rhsM);

  int rank = 1;
  heap.push({ score, 0 });

  for (int i = 0; i < baseDocVectors_.size(); i++) {
    // in case base labels are not provided, basedoc is all label
    if ((args_->basedoc.empty()) && (i == rhs[0] - dict_->nwords())) {
      continue;
    }
    auto cur_score = model_->similarity(lhsM, baseDocVectors_[i]);
    if (cur_score > score) {
      rank++;
    }
    heap.push({ cur_score, i + 1 });
  }

  // get the first K predictions
  int i = 0;
  while (i < args_->K && heap.size() > 0) {
    pred.push_back(heap.top());
    heap.pop();
    i++;
  }

  Metrics s;
  s.clear();
  s.update(rank);
  return s;
}

void StarSpace::printDoc(ofstream& ofs, const vector<int32_t>& tokens) {
  for (auto t : tokens) {
    ofs << dict_->getSymbol(t) << ' ';
  }
  ofs << endl;
}

void StarSpace::evaluate() {
  // set dropout probability to 0 in test case
  args_->dropoutLHS = 0.0;
  args_->dropoutRHS = 0.0;

  loadBaseDocs();
  int N = testData_->getSize();

  auto numThreads = args_->thread;
  vector<thread> threads;
  vector<Metrics> metrics(numThreads);
  vector<vector<Predictions>> predictions(N);
  int numPerThread = ceil(N / numThreads);
  assert(numPerThread > 0);

  vector<ParseResults> examples;
  testData_->getNextKExamples(N, examples);

  auto evalThread = [&] (int idx, int start, int end) {
    metrics[idx].clear();
    for (int i = start; i < end; i++) {
      auto s = evaluateOne(examples[i].LHSTokens, examples[i].RHSTokens, predictions[i]);
      metrics[idx].add(s);
    }
  };

  for (int i = 0; i < numThreads; i++) {
    auto start = i * numPerThread;
    auto end = std::min(start + numPerThread, N);
    assert(end >= start);
    threads.emplace_back(thread([=] {
      evalThread(i, start, end);
    }));
  }
  for (auto& t : threads) t.join();

  Metrics result;
  result.clear();
  for (int i = 0; i < numThreads; i++) {
    if (args_->debug) { metrics[i].print(); }
    result.add(metrics[i]);
  }
  result.average();
  result.print();

  if (!args_->predictionFile.empty()) {
    // print out prediction results to file
    ofstream ofs(args_->predictionFile);
    for (int i = 0; i < N; i++) {
      ofs << "Example " << i << ":\nLHS:\n";
      printDoc(ofs, examples[i].LHSTokens);
      ofs << "RHS: \n";
      printDoc(ofs, examples[i].RHSTokens);
      ofs << "Predictions: \n";
      for (auto pred : predictions[i]) {
        if (pred.second == 0) {
          ofs << "(++) [" << pred.first << "]\t";
          printDoc(ofs, examples[i].RHSTokens);
        } else {
          ofs << "(--) [" << pred.first << "]\t";
          printDoc(ofs, baseDocs_[pred.second - 1]);
        }
      }
      ofs << "\n";
    }
    ofs.close();
  }
}

void StarSpace::saveModel() {
  cout << "Saving model to file : " << args_->model << endl;
  std::string filename(args_->model);
  std::ofstream ofs(filename, std::ofstream::binary);
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // sign model
  ofs.write(kMagic.data(), kMagic.size() * sizeof(char));
  ofs.put(0);
  args_->save(ofs);
  dict_->save(ofs);
  model_->save(ofs);
  ofs.close();
}

void StarSpace::saveModelTsv() {
  cout << "Saving model in tsv format : " << args_->model + ".tsv" << endl;
  ofstream fout(args_->model + ".tsv");
  model_->saveTsv(fout, '\t');
  fout.close();
}

} // starspace
