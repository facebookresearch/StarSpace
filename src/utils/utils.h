// Copyright 2004-, Facebook, Inc. All Rights Reserved.

#pragma once
#include <iostream>
#include <thread>
#include <fstream>
#include <string>
#include <algorithm>

namespace starspace {

struct Metrics {
  float p1, p10, p20, p50, rank;
  int32_t count;

  void clear() {
    p1 = 0;
    p10 = 0;
    p20 = 0;
    p50 = 0;
    rank = 0;
    count = 0;
  };

  void add(const Metrics& b) {
    p1 += b.p1;
    p10 += b.p10;
    p20 += b.p20;
    p50 += b.p50;
    rank += b.rank;
    count += b.count;
  };

  void average() {
    if (count == 0) {
      return ;
    }
    p1 /= count;
    p10 /= count;
    p20 /= count;
    p50 /= count;
    rank /= count;
  }

  void print() {
    std::cout << "Evaluation Metrics : \n"
         << "p@1: " << p1
         << " p@10: " << p10
         << " p@20: " << p20
         << " p@50: " << p50
         << " mean ranks : " << rank
         << " Total examples : " << count << "\n";
  }

  void update(int cur_rank) {
    if (cur_rank == 1) { p1++; }
    if (cur_rank <= 10) { p10++; }
    if (cur_rank <= 20) { p20++; }
    if (cur_rank <= 50) { p50++; }
    rank += cur_rank;
    count++;
  }

};


namespace detail {
extern __thread int id;
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
    auto pos = tellg(f);
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
  off_t partitions[numThreads + 1];
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

} // namespace

