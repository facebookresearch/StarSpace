/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


/**
 * This is the data parser implemented for freebase data.
 * It takes the input format of the freebase15k dataset, and convert
 * it to examples consumed by starspace.
 * Specifally, each training example (head, relation, tail) from input file becomes
 * two examples: (head, relation, tail) and (tail, reverse_relation, head).
 */

#pragma once

#include "dict.h"
#include "parser.h"
#include <string>
#include <vector>

namespace starspace {

class FreebaseDataParser : public DataParser {
public:
  FreebaseDataParser(
    std::shared_ptr<Dictionary> dict,
    std::shared_ptr<Args> args);

  void parseForDict(
      std::string& line,
      std::vector<std::string>& tokens,
      const std::string& sep=" \t") override;

  void parse(
      std::string& line,
      std::vector<ParseResults>& rslt,
      const std::string& sep=" \t") override;

private:
  const std::string REVERSE_PREFIX = "_reverse_";
  std::string reverse(const std::string& s);

};

}
