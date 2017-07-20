/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


#include "parser.h"
#include "utils/normalize.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace starspace {

void chomp(std::string& line, char toChomp = '\n') {
  auto sz = line.size();
  if (sz >= 1 && line[sz - 1] == toChomp) {
    line.resize(sz - 1);
  }
}

DataParser::DataParser(
    shared_ptr<Dictionary> dict,
    shared_ptr<Args> args) {
  dict_ = dict;
  args_ = args;
}

bool DataParser::parse(
    std::string& s,
    ParseResults& rslts,
    const string& sep) {

  chomp(s);
  vector<string> toks;
  boost::split(toks, s, boost::is_any_of(string(sep)));

  return parse(toks, rslts);
}

void DataParser::parseForDict(
    string& line,
    vector<string>& tokens,
    const string& sep) {

  chomp(line);
  boost::split(tokens, line, boost::is_any_of(sep));
}

// check wether it is a valid example
bool DataParser::check(const ParseResults& example) {
  if (args_->trainMode == 0) {
    // require lhs and rhs
    return !example.RHSTokens.empty() && !example.LHSTokens.empty();
  } else {
    // lhs is not required, but rhs should contain at least 2 example
    return example.RHSTokens.size() > 1;
  }
}

void DataParser::addNgrams(
    const std::vector<std::string>& tokens,
    std::vector<int32_t>& line,
    int n) {

  vector<int32_t> hashes;

  for (auto token: tokens) {
    int32_t wid = dict_->getId(token);
    entry_type type = dict_->getType(token);
    if (type == entry_type::word) {
      hashes.push_back(dict_->hash(token));
    }
  }

  for (int32_t i = 0; i < hashes.size(); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < hashes.size() && j < i + n; j++) {
      h = h * 116049371 + hashes[j];
      int64_t id = h % args_->bucket;
      line.push_back(dict_->nwords() + dict_->nlabels() + id);
    }
  }
}

bool DataParser::parse(
    const std::vector<std::string>& tokens,
    ParseResults& rslts) {

  for (auto &token: tokens) {
    auto t = token;
    normalize_text(t);
    int32_t wid = dict_->getId(t);
    if (wid < 0) {
      continue;
    }

    entry_type type = dict_->getType(wid);
    if (type == entry_type::word) {
      rslts.LHSTokens.push_back(wid);
    }
    if (type == entry_type::label) {
      rslts.RHSTokens.push_back(wid);
    }
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, rslts.LHSTokens, args_->ngrams);
  }
  return check(rslts);
}

bool DataParser::parse(
    const std::vector<std::string>& tokens,
    vector<int32_t>& rslts) {

  for (auto &token: tokens) {
    auto t = token;
    normalize_text(t);
    int32_t wid = dict_->getId(t);
    if (wid < 0) {
      continue;
    }

    entry_type type = dict_->getType(wid);
    if (type == entry_type::word) {
      rslts.push_back(wid);
    }
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, rslts, args_->ngrams);
  }
  return rslts.size() > 0;
}

} // namespace starspace
