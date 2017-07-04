// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "starspace.h"
#include <iostream>
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

// initialize dict and load data
void StarSpace::init() {
  cout << "Start to initlaize starspace model.\n";
  assert(args_ != nullptr);
  args_->printArgs();

  // build dict
  dict_ = make_shared<Dictionary>(args_);
  auto filename = args_->trainFile;
  dict_->readFromFile(filename);
  if (args_->debug) {dict_->save(cout);}

  // init parser and laod trian data
  if (!args_->isLabelFeatured) {
    parser_ = make_shared<DataParser>(dict_, args_);
    trainData_ = make_shared<InternDataHandler>(args_);
  } else {
    parser_ = make_shared<LayerDataParser>(dict_, args_);
    trainData_ = make_shared<LayerDataHandler>(args_);
  }
  trainData_->loadFromFile(args_->trainFile, parser_);
  if (args_->debug) {
    trainData_->save(cout);
  }

  // init model with args and dict
  model_ = make_shared<EmbedModel>(args_, dict_);

  // set validation data
  if (!args_->validationFile.empty()) {
    validData_ = args_->isLabelFeatured ?
      make_shared<LayerDataHandler>(args_) : make_shared<InternDataHandler>(args_);
    validData_->loadFromFile(args_->validationFile, parser_);
  } else {
    validData_ = nullptr;
  }
}

void StarSpace::initFromSavedModel() {
  cout << "Start to load a trained starspace model.\n";
  std::ifstream in(args_->model, std::ifstream::binary);
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

  // init data parser
  if (!args_->isLabelFeatured) {
    parser_ = make_shared<DataParser>(dict_, args_);
    testData_ = make_shared<InternDataHandler>(args_);
  } else {
    parser_ = make_shared<LayerDataParser>(dict_, args_);
    testData_ = make_shared<LayerDataHandler>(args_);
  }
}

void StarSpace::initFromTsv() {
  cout << "Start to load a trained embedding model in tsv format.\n";
  assert(args_ != nullptr);

  // build dict
  dict_ = make_shared<Dictionary>(args_);
  dict_->loadDictFromModel(args_->model + ".tsv");
  if (args_->debug) {dict_->save(cout);}

  // load Model
  model_ = make_shared<EmbedModel>(args_, dict_);
  model_->loadTsv(args_->model + ".tsv");

  // init data parser
  if (!args_->isLabelFeatured) {
    parser_ = make_shared<DataParser>(dict_, args_);
    testData_ = make_shared<InternDataHandler>(args_);
  } else {
    parser_ = make_shared<LayerDataParser>(dict_, args_);
    testData_ = make_shared<LayerDataHandler>(args_);
  }
}

void StarSpace::train() {
  float rate = args_->lr;
  float decrPerEpoch = (rate - 1e-9) / args_->epoch;

  for (int i = 0; i < args_->epoch; i++) {
    cout << "begining training : " << rate << ' ' << decrPerEpoch << endl;
    auto err = model_->train(trainData_, args_->thread, rate, rate - decrPerEpoch);
    printf("\n ---+++ %20s %4d Train error : %3.8f +++--- %c%c%c\n",
           "Epoch", i, err,
           0xe2, 0x98, 0x83);
    if (validData_ != nullptr) {
      auto valid_err = model_->test(validData_, args_->thread);
      cout << "Validation error: " << valid_err << endl;
    }
    rate -= decrPerEpoch;
    saveModel();
  }

  // save to tsv format as well
  saveModelTsv();
}

Matrix<Real> StarSpace::getDocVector(const string& line, const string& sep) {
  vector<string> tokens;
  boost::split(tokens, line, boost::is_any_of(string(sep)));
  vector<int32_t> ids;
  parser_->parse(tokens, ids);
  return model_->projectRHS(ids);
}

void StarSpace::loadBaseDocs() {
  if (args_->basedoc.empty()) {
    if (args_->isLabelFeatured) {
      std::cerr << "Must provide base labels when label is featured.\n";
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < dict_->nlabels(); i++) {
      baseDocs_.push_back(model_->projectRHS({i}));
    }
  } else {
    cout << "Loading base docs from file : " << args_->basedoc << endl;
    ifstream fin(args_->basedoc);
    string line;
    while (getline(fin, line)) {
      baseDocs_.push_back(getDocVector(line, " "));
    }
    fin.close();
    cout << "Finished loading base docs.\n";
  }
}

Metrics StarSpace::evaluateOne(
    const vector<int32_t>& lhs,
    const vector<int32_t>& rhs) {

  typedef pair<int32_t, Real> Cand;
  vector<Cand> result;

  auto lhsM = model_->projectLHS(lhs);
  auto rhsM = model_->projectRHS(rhs);
  // here similarity is dot product
  result.push_back({0, model_->similarity(lhsM, rhsM)});

  for (int i = 0; i < baseDocs_.size(); i++) {
    // in case base labels are not provided, basedoc is all label
    if ((args_->basedoc.empty()) && (i == rhs[0])) {
      continue;
    }
    result.push_back({i + 1, model_->similarity(lhsM, baseDocs_[i])});
  }

  std::sort(result.begin(), result.end(),
           [&](Cand a, Cand b) {return a.second > b.second; });

  Metrics s;
  s.clear();
  for (int i = 0; i < result.size(); i++) {
    if (result[i].first == 0) {
      s.update(i + 1);
      break;
    }
  }
  return s;
}

void StarSpace::evaluate() {
  testData_->loadFromFile(args_->testFile, parser_);
  if (args_->debug) {testData_->save(cout);}
  loadBaseDocs();
  int N = testData_->getSize();

  auto numThreads = args_->thread;
  vector<thread> threads;
  vector<Metrics> metrics(numThreads);
  int numPerThread = ceil(N / numThreads);
  assert(numPerThread > 0);

  auto evalThread = [&] (int idx, int start, int end) {
    metrics[idx].clear();
    for (int i = start; i < end; i++) {
      ParseResults ex;
      testData_->getExampleById(i, ex);
      auto s = evaluateOne(ex.LHSTokens, ex.RHSTokens);
      metrics[idx].add(s);
    }
  };

  for (int i = 0; i < numThreads; i++) {
    auto start = i * numPerThread;
    auto end = std::min(start + numPerThread, N);
    assert(end >= start);
    assert(end <= N);
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