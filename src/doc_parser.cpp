/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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

  for (auto token : tokens) {
    string t = token;
    float weight = 1.0;
    if (args_->useWeight) {
      std::size_t pos = token.find(":");
      if (pos != std::string::npos) {
        t = token.substr(0, pos);
        weight = atof(token.substr(pos + 1).c_str());
      }
    }

    if (args_->normalizeText) {
      normalize_text(t);
    }
    int32_t wid = dict_->getId(t);
    if (wid != -1)  {
      feats.push_back(make_pair(wid, weight));
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
    parse(parts[0], rslt.LHSTokens);
    start_idx = 1;
  }
  for (int i = start_idx; i < parts.size(); i++) {
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
