/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * This is the internal data handler class for the case where we
 * have features to represent labels. It overrides a few key functions
 * in DataHandler class in order to return label features for training/testing
 * instead of label ids.
 */

#pragma once

#include "dict.h"
#include "data.h"
#include "doc_parser.h"
#include <string>
#include <vector>
#include <fstream>

namespace starspace {

class LayerDataHandler : public InternDataHandler {
public:
  explicit LayerDataHandler(std::shared_ptr<Args> args);

  void convert(const ParseResults& example, ParseResults& rslts) const override;

  void getWordExamples(int idx, std::vector<ParseResults>& rslts) const override;

  void loadFromFile(const std::string& file,
                    std::shared_ptr<DataParser> parser) override;

  void getRandomRHS(std::vector<Base>& results) const override;

  void save(std::ostream& out) override;

private:
  Base genRandomWord() const override;

  void insert(
      std::vector<Base>& rslt,
      const std::vector<Base>& ex,
      float dropout = 0.0) const;

};

}
