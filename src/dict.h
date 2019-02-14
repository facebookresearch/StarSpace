/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * The implementation of dictionary here is very similar to the dictionary used
 * in fastText (https://github.com/facebookresearch/fastText).
 */

#pragma once

#include "utils/args.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <random>
#include <memory>
#include <boost/format.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#ifdef COMPRESS_FILE
  #include <boost/iostreams/filter/zlib.hpp>
  #include <boost/iostreams/filter/gzip.hpp>
#endif

namespace starspace {

class DataParser;

enum class entry_type : int8_t {word=0, label=1};

struct entry {
  std::string symbol;
  int64_t count;
  entry_type type;
};

class Dictionary {
  public:
    static const std::string EOS;
    static const uint32_t HASH_C;

    explicit Dictionary(std::shared_ptr<Args>);
    int32_t size() const { return size_; };
    int32_t nwords() const { return nwords_; };
    int32_t nlabels() const { return nlabels_; };
    int32_t ntokens() const { return ntokens_; };
    int32_t getId(const std::string&) const;
    entry_type getType(int32_t) const;
    entry_type getType(const std::string&) const;
    const std::string& getSymbol(int32_t) const;
    const std::string& getLabel(int32_t) const;

    uint32_t hash(const std::string& str) const;
    void insert(const std::string&);

    void load(std::istream&);
    void save(std::ostream&) const;
    void readFromFile(const std::string&, std::shared_ptr<DataParser>);
    bool readWord(std::istream&, std::string&) const;

    void threshold(int64_t, int64_t);
    void computeCounts();
    void loadDictFromModel(const std::string& model);

  private:
    static const int32_t MAX_VOCAB_SIZE = 30000000;

    int32_t find(const std::string&) const;

    void addNgrams(
        std::vector<int32_t>& line,
        const std::vector<int32_t>& hashes,
        int32_t n) const;

    std::shared_ptr<Args> args_;
    std::vector<entry> entryList_;
    std::vector<int32_t> hashToIndex_;

    int32_t size_;
    int32_t nwords_;
    int32_t nlabels_;
    int64_t ntokens_;
};

}
