/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "dict.h"
#include "parser.h"
#include "utils/utils.h"
#include <string>
#include <vector>
#include <fstream>

namespace starspace {

class InternDataHandler {
public:
  explicit InternDataHandler(std::shared_ptr<Args> args);

  virtual void loadFromFile(const std::string& file,
                            std::shared_ptr<DataParser> parser);

  virtual void convert(const ParseResults& example, ParseResults& rslt) const;

  virtual void getRandomRHS(std::vector<Base>& results)
    const;

  virtual void save(std::ostream& out);

  virtual void getWordExamples(int idx, std::vector<ParseResults>& rslt) const;

  void getWordExamples(
      const std::vector<Base>& doc,
      std::vector<ParseResults>& rslt) const;

  void addExample(const ParseResults& example);

  void getExampleById(int32_t idx, ParseResults& rslt) const;

  void getNextExample(ParseResults& rslt);

  void getRandomExample(ParseResults& rslt) const;

  void getKRandomExamples(int K, std::vector<ParseResults>& c);

  void getNextKExamples(int K, std::vector<ParseResults>& c);

  size_t getSize() const { return size_; };

  void errorOnZeroExample(const std::string& fileName);

  void initWordNegatives();
  void getRandomWord(std::vector<Base>& result);


protected:
  virtual Base genRandomWord() const;

  static const int32_t MAX_VOCAB_SIZE = 10000000;
  static const int32_t MAX_WORD_NEGATIVES_SIZE = 10000000;

  std::shared_ptr<Args> args_;
  std::vector<ParseResults> examples_;

  int32_t idx_ = -1;
  int32_t size_ = 0;

  int32_t word_iter_;
  std::vector<Base> word_negatives_;
};

}
