/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "doc_parser.h"
#include "utils/normalize.h"
#include <string>
#include <vector>
#include <fstream>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace starspace {

LayerDataParser::LayerDataParser(
    shared_ptr<Dictionary> dict,
    shared_ptr<Args> args)
: DataParser(dict, args) {};

bool LayerDataParser::parse(
    string& s,
    vector<Base>& feats,
    const string& sep) {

  // split each part into tokens
  vector<string> tokens;
  boost::split(tokens, s, boost::is_any_of(string(sep)));

  int start_idx = 0;
  float ex_weight = 1.0;
  if (tokens[0].find("__weight__") != std::string::npos) {
    std::size_t pos = tokens[0].find(args_->weightSep);
    if (pos != std::string::npos) {
        ex_weight = atof(tokens[0].substr(pos + 1).c_str());
    }
    start_idx = 1;
  }

  for (unsigned int i = start_idx; i < tokens.size(); i++) {
    string t = tokens[i];
    float weight = 1.0;
    if (args_->useWeight) {
      std::size_t pos = tokens[i].find(args_->weightSep);
      if (pos != std::string::npos) {
        t = tokens[i].substr(0, pos);
        weight = atof(tokens[i].substr(pos + 1).c_str());
      }
    }

    if (args_->normalizeText) {
      normalize_text(t);
    }
    int32_t wid = dict_->getId(t);
    if (wid != -1)  {
      feats.push_back(make_pair(wid, weight * ex_weight));
    }
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, feats, args_->ngrams);
  }

  return feats.size() > 0;
}

bool LayerDataParser::parse(
    string& line,
    ParseResults& rslt,
    const string& sep) {

  vector<string> parts;
  boost::split(parts, line, boost::is_any_of("\t"));
  int start_idx = 0;

  if (args_->trainMode == 0) {
    // the first part is input features
    parse(parts[start_idx], rslt.LHSTokens);
    start_idx += 1;
  }
  for (unsigned int i = start_idx; i < parts.size(); i++) {
    vector<Base> feats;
    if (parse(parts[i], feats)) {
      rslt.RHSFeatures.push_back(feats);
    }
  }

  bool isValid;
  if (args_->trainMode == 0) {
    isValid = (rslt.LHSTokens.size() > 0) && (rslt.RHSFeatures.size() > 0);
  } else {
    // need to have at least two examples
    isValid = rslt.RHSFeatures.size() > 1;
  }

  return isValid;
}

} // namespace starspace
