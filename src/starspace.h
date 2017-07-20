/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "utils/args.h"
#include "dict.h"
#include "matrix.h"
#include "parser.h"
#include "doc_parser.h"
#include "model.h"
#include "utils/utils.h"

namespace starspace {

class StarSpace {
  public:
    explicit StarSpace(std::shared_ptr<Args> args);

    void init();
    void initFromTsv();
    void initFromSavedModel(const std::string& filename);

    void train();
    void evaluate();

    Matrix<Real> getDocVector(const std::string& line, const std::string& sep);
    void nearestNeighbor(const std::string& line, int k);

    void saveModel();
    void saveModelTsv();

    const std::string kMagic = "STARSPACE-2017-1";

  private:
    void initParser();
    std::shared_ptr<InternDataHandler> initData();
    void loadBaseDocs();

    Metrics evaluateOne(
        const std::vector<int32_t>& lhs,
        const std::vector<int32_t>& rhs);

    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<DataParser> parser_;
    std::shared_ptr<InternDataHandler> trainData_;
    std::shared_ptr<InternDataHandler> validData_;
    std::shared_ptr<InternDataHandler> testData_;
    std::shared_ptr<EmbedModel> model_;

    std::vector<Matrix<Real>> baseDocs_;
};

}
