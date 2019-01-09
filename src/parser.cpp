/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
  vector<string> toks;
  boost::split(toks, line, boost::is_any_of(string(sep)));
  for (unsigned int i = 0; i < toks.size(); i++) {
    string token = toks[i];
    if (args_->useWeight) {
      std::size_t pos = toks[i].find(args_->weightSep);
      if (pos != std::string::npos) {
        token = toks[i].substr(0, pos);
      }
    }
    if (args_->normalizeText) {
      normalize_text(token);
    }
    if (token.find("__weight__") == std::string::npos) {
      tokens.push_back(token);
    }
  }
}

// check wether it is a valid example
bool DataParser::check(const ParseResults& example) {
  if (args_->trainMode == 0) {
    // require lhs and rhs
    return !example.RHSTokens.empty() && !example.LHSTokens.empty();
  } if (args_->trainMode == 5) {
    // only requires lhs.
    return !example.LHSTokens.empty();
  } else {
    // lhs is not required, but rhs should contain at least 2 example
    return example.RHSTokens.size() > 1;
  }
}

void DataParser::addNgrams(
    const std::vector<std::string>& tokens,
    std::vector<Base>& line,
    int n) {

  vector<int32_t> hashes;

  for (auto token: tokens) {
    entry_type type = dict_->getType(token);
    if (type == entry_type::word) {
      hashes.push_back(dict_->hash(token));
    }
  }

  for (int32_t i = 0; i < (int32_t)(hashes.size()); i++) {
    uint64_t h = hashes[i];
    for (int32_t j = i + 1; j < (int32_t)(hashes.size()) && j < i + n; j++) {
      h = h * Dictionary::HASH_C + hashes[j];
      int64_t id = h % args_->bucket;
      line.push_back(make_pair(dict_->nwords() + dict_->nlabels() + id, 1.0));
    }
  }
}

bool DataParser::parse(
    const std::vector<std::string>& tokens,
    ParseResults& rslts) {

  for (auto &token: tokens) {
    if (token.find("__weight__") != std::string::npos) {
      std::size_t pos = token.find(args_->weightSep);
      if (pos != std::string::npos) {
        rslts.weight = atof(token.substr(pos + 1).c_str());
      }
      continue;
    }
    string t = token;
    float weight = 1.0;
    if (args_->useWeight) {
      std::size_t pos = token.find(args_->weightSep);
      if (pos != std::string::npos) {
        t = token.substr(0, pos);
        weight = atof(token.substr(pos + 1).c_str());
      }
    }

    if (args_->normalizeText) {
      normalize_text(t);
    }
    int32_t wid = dict_->getId(t);
    if (wid < 0) {
      continue;
    }

    entry_type type = dict_->getType(wid);
    if (type == entry_type::word) {
      rslts.LHSTokens.push_back(make_pair(wid, weight));
    }
    if (type == entry_type::label) {
      rslts.RHSTokens.push_back(make_pair(wid, weight));
    }
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, rslts.LHSTokens, args_->ngrams);
  }

  return check(rslts);
}

bool DataParser::parse(
    const std::vector<std::string>& tokens,
    vector<Base>& rslts) {

  for (auto &token: tokens) {
    auto t = token;
    float weight = 1.0;
    if (args_->useWeight) {
      std::size_t pos = token.find(args_->weightSep);
      if (pos != std::string::npos) {
        t = token.substr(0, pos);
        weight = atof(token.substr(pos + 1).c_str());
      }
    }

    if (args_->normalizeText) {
      normalize_text(t);
    }
    int32_t wid = dict_->getId(t);
    if (wid < 0) {
      continue;
    }

    //entry_type type = dict_->getType(wid);
    rslts.push_back(make_pair(wid, weight));
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, rslts, args_->ngrams);
  }
  return rslts.size() > 0;
}

} // namespace starspace
