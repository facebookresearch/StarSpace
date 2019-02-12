/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <iostream>
#include <thread>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#ifdef COMPRESS_FILE
  #include <boost/iostreams/filter/gzip.hpp>
#endif

namespace starspace {

struct Metrics {
  float hit1, hit10, hit20, hit50, rank;
  int32_t count;

  void clear() {
    hit1 = 0;
    hit10 = 0;
    hit20 = 0;
    hit50 = 0;
    rank = 0;
    count = 0;
  };

  void add(const Metrics& b) {
    hit1 += b.hit1;
    hit10 += b.hit10;
    hit20 += b.hit20;
    hit50 += b.hit50;
    rank += b.rank;
    count += b.count;
  };

  void average() {
    if (count == 0) {
      return ;
    }
    hit1 /= count;
    hit10 /= count;
    hit20 /= count;
    hit50 /= count;
    rank /= count;
  }

  void print() {
    std::cout << "Evaluation Metrics : \n"
         << "hit@1: " << hit1
         << " hit@10: " << hit10
         << " hit@20: " << hit20
         << " hit@50: " << hit50
         << " mean ranks : " << rank
         << " Total examples : " << count << "\n";
  }

  void update(int cur_rank) {
    if (cur_rank == 1) { hit1++; }
    if (cur_rank <= 10) { hit10++; }
    if (cur_rank <= 20) { hit20++; }
    if (cur_rank <= 50) { hit50++; }
    rank += cur_rank;
    count++;
  }

};


namespace detail {
extern thread_local int id;
}

namespace {
inline int getThreadID() {
  return detail::id;
}
}

namespace {
template<typename Stream>
void reset(Stream& s, std::streampos pos) {
  s.clear();
  s.seekg(pos, std::ios_base::beg);
}

template<typename Stream>
std::streampos tellg(Stream& s) {
  auto retval = s.tellg();
  return retval;
}
}

// Apply a closure pointwise to every line of a file.
template<typename String=std::string,
         typename Lambda>
void foreach_line(const String& fname,
                  Lambda f,
                  int numThreads = 1) {
  using namespace std;

  auto filelen = [&](ifstream& f) {
    f.seekg(0, ios_base::end);
    return tellg(f);
  };

  ifstream ifs(fname);
  if (!ifs.good()) {
    throw runtime_error(string("error opening ") + fname);
  }
  auto len = filelen(ifs);
  // partitions[i],partitions[i+1] will be the bytewise boundaries for the i'th
  // thread.
  std::vector<off_t> partitions(numThreads + 1);
  partitions[0] = 0;
  partitions[numThreads] = len;

  // Seek to bytewise partition boundaries, and read one line forward.
  string unused;
  for (int i = 1; i < numThreads; i++) {
    reset(ifs, (len / numThreads) * i);
    getline(ifs, unused);
    partitions[i] = tellg(ifs);
  }

  // It's possible that the ranges in partitions overlap; consider,
  // e.g., a machine with 100 hardware threads and only 99 lines
  // in the file. In this case, we'll do some excess work, so we ask
  // that f() be idempotent.
  vector<thread> threads;
  for (int i = 0; i < numThreads; i++) {
    threads.emplace_back([i, f, &fname, &partitions] {
      detail::id = i;
      // Get our own seek pointer.
      ifstream ifs2(fname);
      ifs2.seekg(partitions[i]);
      string line;
      while (tellg(ifs2) < partitions[i + 1] && getline(ifs2, line)) {
        // We don't know the line number. Super-bummer.
        f(line);
      }
    });
  }
  for (auto &t: threads) {
    t.join();
  }
}

template<typename String=std::string,
         typename Lambda>
void foreach_line_gz(
    const String& fname,
    int numFiles,
    Lambda f,
    int numThreads = 1) {

  using namespace std;
  using namespace boost::iostreams;

  vector<thread> threads;
  numThreads = std::min(numFiles, numThreads);

#ifdef COMPRESS_FILE
  for (int i = 0; i < numFiles; i++) {
    auto thread_id = i % numThreads;
    threads.emplace_back([thread_id, i, f, &fname] {
      detail::id = thread_id;
      auto fname_t = fname + boost::str(boost::format("%02d") % i) + ".gz";
      ifstream ifs2(fname_t);
      if (!ifs2.good()) {
        return;
      }

      cout << "Reading file from " << fname_t << endl;
      filtering_istream in;
      in.push(gzip_decompressor());
      in.push(ifs2);
      std::string line;
      while (getline(in, line, '\n')) {
        f(line);
      }
    });
  }
  for (auto &t: threads) {
    t.join();
  }
#endif
}

} // namespace
