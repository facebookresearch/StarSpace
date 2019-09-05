/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <boost/algorithm/string.hpp>

#include "starspace_pythonic.h"

using namespace std;

namespace starspace {

StarSpacePythonic::StarSpacePythonic(std::shared_ptr<Args> args) :
  StarSpace(args)
{
}

vector<Base> StarSpacePythonic::parseDoc(
        const string& line,
        const string& sep)
{
    vector<Base> ids;
    StarSpace::parseDoc(line, ids, sep);
    return ids;
}

vector<Predictions> StarSpacePythonic::predict(const vector<Base>& input, int k)
{
  vector<Predictions> out;
  StarSpace::predict(input, out, k);
  return out;
}

vector<vector<string>> StarSpacePythonic::renderTokens(const vector<Predictions>& predictions)
{
  vector<vector<string>> out;

  for (auto p: predictions) {
    vector<string> current;
    auto tokens = baseDocs_[p.second];

    for (auto t : tokens)
      if (t.first < dict_->size())
        current.push_back(dict_->getSymbol(t.first));

    out.push_back(current);
  }

  return out;
}

} // namespace starspace end
