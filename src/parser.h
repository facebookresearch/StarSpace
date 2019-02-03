/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/**
 * This is the basic class of data parsing.
 * It provides essential functions as follows:
 * - parse(input, output):
 *   takes input as a line of string (or a vector of string tokens)
 *   and return output result which is one example contains l.h.s. features
 *   and r.h.s. features.
 *
 * - parseForDict(input, tokens):
 *   takes input as a line of string, output tokens to be added for building
 *   the dictionary.
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

typedef std::pair<int32_t, float> Base;

struct ParseResults {
  float weight = 1.0;
  std::vector<Base> LHSTokens;
  std::vector<Base> RHSTokens;
  std::vector<std::vector<Base>> RHSFeatures;
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

  virtual void parseForDict(
      std::string& s,
      std::vector<std::string>& tokens,
      const std::string& sep="\t ");

  bool parse(
      const std::vector<std::string>& tokens,
      std::vector<Base>& rslt);

  bool parse(
      const std::vector<std::string>& tokens,
      ParseResults& rslt);

  bool check(const ParseResults& example);

  void addNgrams(
      const std::vector<std::string>& tokens,
      std::vector<Base>& line,
      int32_t n);

  std::shared_ptr<Dictionary> getDict() { return dict_; };

  void resetDict(std::shared_ptr<Dictionary> dict) { dict_ = dict; };

protected:
  std::shared_ptr<Dictionary> dict_;
  std::shared_ptr<Args> args_;
};

}
