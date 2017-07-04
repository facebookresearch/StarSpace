// Copyright 2004-, Facebook, Inc. All Rights Reserved.

/* This is the parser class for the case where we have features
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
      std::vector<int32_t>& rslt,
      const std::string& sep=" ");

  bool parse(
      std::string& line,
      ParseResults& rslt,
      const std::string& sep="\t") override;

};

}
