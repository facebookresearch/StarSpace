/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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

  void convert(const ParseResults& example, ParseResults& rslt) const override;

  void loadFromFile(const std::string& file,
                    std::shared_ptr<DataParser> parser) override;

  void getRandomRHS(std::vector<int32_t>& results) const override;

  void save(std::ostream& out) override;

private:
  void insert(
      std::vector<int32_t>& rslt,
      const std::vector<int32_t>& ex,
      float dropout = 0.0) const;

};

}
