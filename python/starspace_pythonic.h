/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include <starspace.h>
#include <parser.h>

namespace starspace {
  /**
   * This class wraps StarSpace class to make functions more python like.
   *
   * C++11 allows returning complex types by value in fast manner thus dancing
   * around with passing vectors by reference to function is not needed.
   *
   * The functions used here are not the same as in StarSpace class, thus
   * no virtual modifiers for them are needed.
   */

  class StarSpacePythonic : public StarSpace
  {
  public:
    explicit StarSpacePythonic(std::shared_ptr<Args> args);

    std::vector<Base> parseDoc(
        const std::string& line,
        const std::string& sep);

    // Let use fasttext compatible signature of predict function
    std::vector<Predictions> predict(const std::vector<Base>& input, int k);

    // Render response to string instead using ofstream
    std::vector<std::vector<std::string>> renderTokens(const std::vector<Predictions>& tokens);
  };
}
