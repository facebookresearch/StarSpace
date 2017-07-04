// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#include "normalize.h"

#include <algorithm>
#include <ctype.h>
#include <assert.h>
#include <string>

namespace starspace {

void normalize_text(std::string& str) {
  /*
   * Don't normalize super-short strings; "747" means "747", NwA stays
   * NwA, etc.
   */
  if (str.size() <= 3) return;

  /*
   * We categorize longer strings into the following buckets:
   *
   * 1. All punctuation-and-numeric. Things in this bucket get
   *    their numbers flattened, to prevent combinatorial explosions.
   *    They might be specific numbers, prices, etc.
   *
   * 2. All letters: case-flattened.
   *
   * 3. Mixed letters and numbers: a product ID? Flatten case and leave
   *    numbers alone.
   *
   * The case-normalization is state-machine-driven.
   */
  bool allNumeric = true;
  bool containsDigits = false;

  for (char c: str) {
    assert(c); // don't shove binary data through this.
    containsDigits |= isdigit(c);
    if (!isascii(c)) {
      allNumeric = false;
      continue;
    }
    if (!isalpha(c)) continue;
    bool isUpper = isupper(c);
    allNumeric = false;
  }

  bool flattenCase = true;
  bool flattenNum = allNumeric && containsDigits;
  if (!flattenNum && !flattenCase) return;

  std::transform(str.begin(), str.end(), str.begin(),
    [&](char c) {
      if (flattenNum && isdigit(c)) return '0';
      if (isalpha(c)) return char(tolower(c));
      return c;
  });
}

}
