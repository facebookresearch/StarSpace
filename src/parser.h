// Copyright 2004-, Facebook, Inc. All Rights Reserved.

/* This is the basic class of data parsing.
 * It provides essential functions as follows:
 * - parse(input, output):
 *   takes input as a line of string (or a vector of string tokens)
 *   and return output result which is one example contains l.h.s. features
 *   and r.h.s. features.
 *
 * - check(example):
 *   checks whether the example is a valid example.
 *
 * - addNgrams(input, output):
 *   add ngrams from input as output.
 *
 * One can write different parsers for data with different format.
 */

#pragma once

#include "dict.h"
#include <string>
#include <vector>

namespace starspace {

struct ParseResults {
  std::vector<int32_t> LHSTokens;
  std::vector<int32_t> RHSTokens;
  std::vector<std::vector<int32_t>> RHSFeatures;
};

typedef std::vector<ParseResults> Corpus;

class DataParser {
public:
  explicit DataParser(
    std::shared_ptr<Dictionary> dict,
    std::shared_ptr<Args> args);

  virtual bool parse(
      std::string& s,
      ParseResults& rslt,
      const std::string& sep="\t ");

  bool parse(
      const std::vector<std::string>& tokens,
      ParseResults& rslt);

  bool parse(
      const std::vector<std::string>& tokens,
      std::vector<int32_t>& rslt);

  bool check(const ParseResults& example);

  void addNgrams(
      const std::vector<std::string>& tokens,
      std::vector<int32_t>& line,
      int32_t n);

  std::shared_ptr<Dictionary> getDict() { return dict_; };

protected:
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Args> args_;
};

}