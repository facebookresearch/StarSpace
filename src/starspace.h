/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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

typedef std::pair<Real, int32_t> Predictions;

class StarSpace {
  public:
    explicit StarSpace(std::shared_ptr<Args> args);

    void init();
    void initFromTsv(const std::string& filename);
    void initFromSavedModel(const std::string& filename);

    void train();
    void evaluate();

    MatrixRow getNgramVector(const std::string& phrase);
    Matrix<Real> getDocVector(
        const std::string& line,
        const std::string& sep = " \t");
    void parseDoc(
        const std::string& line,
        std::vector<Base>& ids,
        const std::string& sep);

    void nearestNeighbor(const std::string& line, int k);


    std::unordered_map<std::string, float> predictTags(const std::string& line, int k);
    std::string printDocStr(const std::vector<Base>& tokens); 
    
    void saveModel(const std::string& filename);
    void saveModelTsv(const std::string& filename);
    void printDoc(std::ostream& ofs, const std::vector<Base>& tokens);

    const std::string kMagic = "STARSPACE-2018-2";


    void loadBaseDocs();

    void predictOne(
        const std::vector<Base>& input,
        std::vector<Predictions>& pred);

    std::shared_ptr<Args> args_;
    std::vector<std::vector<Base>> baseDocs_;
  private:
    void initParser();
    void initDataHandler();
    std::shared_ptr<InternDataHandler> initData();
    Metrics evaluateOne(
        const std::vector<Base>& lhs,
        const std::vector<Base>& rhs,
        std::vector<Predictions>& pred,
        bool excludeLHS);

    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<DataParser> parser_;
    std::shared_ptr<InternDataHandler> trainData_;
    std::shared_ptr<InternDataHandler> validData_;
    std::shared_ptr<InternDataHandler> testData_;
    std::shared_ptr<EmbedModel> model_;

    std::vector<Matrix<Real>> baseDocVectors_;
};

}
