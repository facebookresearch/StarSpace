// Copyright 2004-, Facebook, Inc. All Rights Reserved.
#pragma once

#include <string>

namespace starspace {

// In-place normalization of UTF-8 strings.
extern void normalize_text(std::string& buf);

}
