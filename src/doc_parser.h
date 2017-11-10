/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */


/**
 * This is the parser class for the case where we have features
 * to represent labels. It overrides a few key functions such as
 * parse(input, output) and check(example) in the basic Parser class.
 */

#pragma once

#include "dict.h"
#include "parser.h"
#include <string>
#include <vector>

namespace starspace {

class LayerDataParser : public DataParser {
public:
  LayerDataParser(
    std::shared_ptr<Dictionary> dict,
    std::shared_ptr<Args> args);

  bool parse(
      std::string& line,
      std::vector<Base>& rslt,
      const std::string& sep=" ");

  bool parse(
      std::string& line,
      ParseResults& rslt,
      const std::string& sep="\t") override;

};

}
