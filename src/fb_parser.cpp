/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fb_parser.h"
#include "utils/normalize.h"
#include <string>
#include <vector>
#include <fstream>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace starspace {

FreebaseDataParser::FreebaseDataParser(
    shared_ptr<Dictionary> dict,
    shared_ptr<Args> args)
: DataParser(dict, args) {};

string FreebaseDataParser::reverse(
    const string& rel) {
  return REVERSE_PREFIX + rel;
}

void FreebaseDataParser::parseForDict(
    string& s,
    vector<string>& tokens,
    const string& sep) {

  // split each part into tokens
  boost::split(tokens, s, boost::is_any_of(string(sep)));

  assert(tokens.size() == 3);
  tokens.push_back(reverse(tokens[1]));
}

// For each input line of form (head, relation, tail)
// We generate two examples:
// 1. lhs: (head, relation); rhs: tail.
// 2. lhs: (tail, reverse_relation); rhs: head.
void FreebaseDataParser::parse(
    string& s,
    vector<ParseResults>& rslts,
    const string& sep) {

  rslts.clear();

  vector<string> tokens;
  boost::split(tokens, s, boost::is_any_of(string(sep)));

  assert(tokens.size() == 3);
  ParseResults ex1, ex2;

  auto head_id = dict_->getId(tokens[0]);
  auto rel_id = dict_->getId(tokens[1]);
  auto tail_id = dict_->getId(tokens[2]);
  auto reverse_rel_id = dict_->getId(reverse(tokens[1]));

  if (head_id == -1 || rel_id == -1 || tail_id == -1 || reverse_rel_id == -1) {
    return ;
  }

  ex1.LHSTokens.push_back(head_id);
  ex1.LHSTokens.push_back(rel_id);
  ex1.RHSTokens.push_back(tail_id);
  rslts.emplace_back(ex1);

  ex2.LHSTokens.push_back(tail_id);
  ex2.LHSTokens.push_back(rel_id);
  ex2.RHSTokens.push_back(head_id);
  rslts.emplace_back(ex2);
}

} // namespace starspace
