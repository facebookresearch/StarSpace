/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dict.h"
#include "parser.h"

#include <assert.h>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace boost::iostreams;

namespace starspace {

const std::string Dictionary::EOS = "</s>";
const uint32_t Dictionary::HASH_C = 116049371;

Dictionary::Dictionary(shared_ptr<Args> args) : args_(args),
  hashToIndex_(MAX_VOCAB_SIZE, -1), size_(0), nwords_(0), nlabels_(0),
  ntokens_(0)
  {
    entryList_.clear();
  }

// hash trick from fastText
uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

int32_t Dictionary::find(const std::string& w) const {
  int32_t h = hash(w) % MAX_VOCAB_SIZE;
  while (hashToIndex_[h] != -1 && entryList_[hashToIndex_[h]].symbol != w) {
    h = (h + 1) % MAX_VOCAB_SIZE;
  }
  return h;
}

int32_t Dictionary::getId(const string& symbol) const {
  int32_t h = find(symbol);
  return hashToIndex_[h];
}

const std::string& Dictionary::getSymbol(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return entryList_[id].symbol;
}

const std::string& Dictionary::getLabel(int32_t lid) const {
  assert(lid >= 0);
  assert(lid < nlabels_);
  return entryList_[lid + nwords_].symbol;
}

entry_type Dictionary::getType(int32_t id) const {
  assert(id >= 0);
  assert(id < size_);
  return entryList_[id].type;
}

entry_type Dictionary::getType(const string& w) const {
  return (w.find(args_->label) == 0)? entry_type::label : entry_type::word;
}

void Dictionary::insert(const string& symbol) {
  int32_t h = find(symbol);
  ntokens_++;
  if (hashToIndex_[h] == -1) {
    entry e;
    e.symbol = symbol;
    e.count = 1;
    e.type = getType(symbol);
    entryList_.push_back(e);
    hashToIndex_[h] = size_++;
  } else {
    entryList_[hashToIndex_[h]].count++;
  }
}

void Dictionary::save(std::ostream& out) const {
  out.write((char*) &size_, sizeof(int32_t));
  out.write((char*) &nwords_, sizeof(int32_t));
  out.write((char*) &nlabels_, sizeof(int32_t));
  out.write((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    entry e = entryList_[i];
    out.write(e.symbol.data(), e.symbol.size() * sizeof(char));
    out.put(0);
    out.write((char*) &(e.count), sizeof(int64_t));
    out.write((char*) &(e.type), sizeof(entry_type));
  }
}

void Dictionary::load(std::istream& in) {
  entryList_.clear();
  std::fill(hashToIndex_.begin(), hashToIndex_.end(), -1);
  in.read((char*) &size_, sizeof(int32_t));
  in.read((char*) &nwords_, sizeof(int32_t));
  in.read((char*) &nlabels_, sizeof(int32_t));
  in.read((char*) &ntokens_, sizeof(int64_t));
  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = in.get()) != 0) {
      e.symbol.push_back(c);
    }
    in.read((char*) &e.count, sizeof(int64_t));
    in.read((char*) &e.type, sizeof(entry_type));
    entryList_.push_back(e);
    hashToIndex_[find(e.symbol)] = i;
  }
}

/* Build dictionary from file.
 * In dictionary building process, if the current dictionary is at 75% capacity,
 * it automatically increases the threshold for both word and label.
 * At the end the -minCount and -minCountLabel from arguments will be applied
 * as thresholds.
 */
void Dictionary::readFromFile(
    const std::string& file,
    shared_ptr<DataParser> parser) {

  int64_t minThreshold = 1;
  size_t lines_read = 0;

  auto readFromInputStream = [&](std::istream& in) {
    string line;
    while (getline(in, line, '\n')) {
      vector<string> tokens;
      parser->parseForDict(line, tokens);
      lines_read++;
      for (auto token : tokens) {
        insert(token);
        if ((ntokens_ % 1000000 == 0) && args_->verbose) {
          std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
        }
        if (size_ > 0.75 * MAX_VOCAB_SIZE) {
          minThreshold++;
          threshold(minThreshold, minThreshold);
        }
      }
    }
  };

#ifdef COMPRESS_FILE
  if (args_->compressFile == "gzip") {
    cout << "Build dict from compressed input file.\n";
    for (int i = 0; i < args_->numGzFile; i++) {
      filtering_istream in;
      auto str_idx = boost::str(boost::format("%02d") % i);
      auto fname = file + str_idx + ".gz";
      ifstream ifs(fname);
      if (!ifs.good()) {
        continue;
      }
      in.push(gzip_decompressor());
      in.push(ifs);
      readFromInputStream(in);
      ifs.close();
    }
  } else {
    cout << "Build dict from input file : " << file << endl;
    ifstream fin(file);
    if (!fin.is_open()) {
      cerr << "Input file cannot be opened!" << endl;
      exit(EXIT_FAILURE);
    }
    readFromInputStream(fin);
    fin.close();
  }
#else
  cout << "Build dict from input file : " << file << endl;
  ifstream fin(file);
  if (!fin.is_open()) {
    cerr << "Input file cannot be opened!" << endl;
    exit(EXIT_FAILURE);
  }
  readFromInputStream(fin);
  fin.close();
#endif

  threshold(args_->minCount, args_->minCountLabel);

  std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::endl;
  std::cerr << "Number of words in dictionary:  " << nwords_ << std::endl;
  std::cerr << "Number of labels in dictionary: " << nlabels_ << std::endl;
  if (lines_read == 0) {
    std::cerr << "ERROR: Empty file." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (size_ == 0) {
    std::cerr << "Empty vocabulary. Try a smaller -minCount value."
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Sort the dictionary by [word, label] order and by number of occurance.
// Removes word / label that does not pass respective threshold.
void Dictionary::threshold(int64_t t, int64_t tl) {
  sort(entryList_.begin(), entryList_.end(), [](const entry& e1, const entry& e2) {
        if (e1.type != e2.type) return e1.type < e2.type;
        return e1.count > e2.count;
      });
  entryList_.erase(remove_if(entryList_.begin(), entryList_.end(), [&](const entry& e) {
        return (e.type == entry_type::word && e.count < t) ||
               (e.type == entry_type::label && e.count < tl);
      }), entryList_.end());

  entryList_.shrink_to_fit();

  computeCounts();
}

void Dictionary::computeCounts() {
  size_ = 0;
  nwords_ = 0;
  nlabels_ = 0;
  std::fill(hashToIndex_.begin(), hashToIndex_.end(), -1);
  for (auto it = entryList_.begin(); it != entryList_.end(); ++it) {
    int32_t h = find(it->symbol);
    hashToIndex_[h] = size_++;
    if (it->type == entry_type::word) nwords_++;
    if (it->type == entry_type::label) nlabels_++;
  }
}

// Given a model saved in .tsv format, build the dictionary from model.
void Dictionary::loadDictFromModel(const string& modelfile) {
  cout << "Loading dict from model file : " << modelfile << endl;
  ifstream fin(modelfile);
  string line;
  while (getline(fin, line)) {
    string symbol;
    stringstream ss(line);
    ss >> symbol;
    insert(symbol);
  }
  fin.close();
  computeCounts();

  std::cout << "Number of words in dictionary:  " << nwords_ << std::endl;
  std::cout << "Number of labels in dictionary: " << nlabels_ << std::endl;
}

} // namespace
