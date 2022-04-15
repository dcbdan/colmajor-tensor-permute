#pragma once

#include <vector>
#include <tuple>

using std::vector;
using std::tuple;

#include <iostream>
#define DCB01(x) // std::cout << x << std::endl

template <typename T>
std::ostream& operator<<(std::ostream& os, vector<T> const& xs) {
  if(xs.size() == 0) {
    os << "[]";
    return os;
  }

  os << "[" << xs[0];
  for(int i = 1; i < xs.size(); ++i) {
    os << "," << xs[i];
  }
  os << "]";

  return os;
}



struct permute_t {
  permute_t(int min_block_size): min_block_size(min_block_size) {}

  void operator()(
    vector<int> const& dims,
    vector<int> const& perm,
    float* inn,
    float* out) const
  {
    // TODO: fuse all of the adjacent permutations

    // This is a "batched" permutation if
    // the last indices are unpermuted... That is,
    //   perm = {1,0,2}   has a batch size of dims[2],
    //   perm = {1,0,2,3] has a batch size of dims[2]*dims[3].
    int num_batch_dims = 0;
    int batch_size     = 1;
    for(int i = perm.size() - 1; i >= 0; --i) {
      if(perm[i] == i) {
        num_batch_dims++;
        batch_size *= dims[i];
      } else {
        break;
      }
    }

    // In this case, there is no permutation
    // and so it is just a copy.
    // For example,
    //   perm might equal {0,1,2,3,4,5}
    if(num_batch_dims == perm.size()) {
      DCB01("JUST A COPY");
      std::copy(inn, inn + batch_size, out);
      return;
    }

    // In this case, there are no batch dimensions.
    // (This would be correct even if there were
    //  batch dimensions.)
    if(num_batch_dims == 0) {
      vector<tuple<int,int>> rngs;
      rngs.reserve(dims.size());
      for(auto const& n: dims) {
        rngs.emplace_back(0, n);
      }
      DCB01("NO BATCH");
      recurse(rngs, dims, perm, inn, out);
      return;
    }

    // This is a batched permutation; do each batch separately.
    // The idea being that doint this in batches will increase cache hits the most.
    //   TODO: test this, maybe add a flag.

    vector<int> batch_dims(dims.size() - num_batch_dims);
    std::copy(dims.begin(), dims.begin() + batch_dims.size(), batch_dims.begin());

    vector<int> batch_perm(batch_dims.size());
    std::copy(perm.begin(), perm.begin() + batch_dims.size(), batch_perm.begin());

    vector<tuple<int,int>> batch_rngs;
    int offset = 1;
    batch_rngs.reserve(batch_dims.size());
    for(auto const& n: batch_dims) {
      batch_rngs.emplace_back(0, n);
      offset *= n;
    }

    DCB01("ALL THE BATCHES " << batch_size << " ... " << offset);
    for(int which_batch = 0; which_batch != batch_size; ++which_batch) {
      recurse(batch_rngs, batch_dims, batch_perm, inn, out);
      inn += offset;
      out += offset;
    }
  }

private:
  void recurse(
    vector<tuple<int,int>> rngs,
    vector<int> const& dims,
    vector<int> const& perm,
    float* inn, float* out) const
  {
    // Traverse over rngs to determine two things:
    //
    // 1. What is the block size being written to?
    //     > use this to see if this is the base case
    //
    // 2. Which rank has the largest remaining dimension?
    //     > if not the base case, recurse on this rank

    int block_size = 1;
    int which_recurse = 0;
    int largest_remaining = 0;
    for(int i = 0; i != dims.size(); ++i) {
      auto const& [beg, end] = rngs[i];
      int remaining = end - beg;
      block_size *= remaining;

      if(remaining > largest_remaining) {
        largest_remaining = remaining;
        which_recurse = i;
      }
    }

    if(block_size < min_block_size) {
      int offset_inn;
      int offset_out;
      indexer_t indexer(rngs, dims, perm);
      do {
        //auto [offset_inn, offset_out] = indexer();
        indexer(offset_inn, offset_out);

        out[offset_out] = inn[offset_inn];

      } while(indexer.increment());

      return;
    }

    auto [beg, end] = rngs[which_recurse];
    int half = beg + ((end-beg) / 2);

    rngs[which_recurse] = {beg, half};
    recurse(rngs, dims, perm, inn, out);

    rngs[which_recurse] = {half,end};
    recurse(rngs, dims, perm, inn, out);
  }

private:
  struct indexer_t {
    indexer_t(
      vector<tuple<int,int>> const& rngs,
      vector<int>            const& dims,
      vector<int>            const& perm):
        rngs(rngs), dims(dims), perm(perm)
    {
      idx.reserve(rngs.size());
      for(auto const& [beg,_]: rngs) {
        idx.push_back(beg);
      }
    }

    inline bool increment() {
      for(int i = 0; i < idx.size(); ++i) {
        if(idx[i] + 1 == std::get<1>(rngs[i])) {
          idx[i] = std::get<0>(rngs[i]);
        } else {
          idx[i] += 1;
          return true;
        }
      }
      return false;
    }

    inline void operator()(int& m_inn, int& m_out) const {
      m_inn = 1;
      m_out = 1;

      int ret_inn = 0;
      int ret_out = 0;

      for(int i = 0; i != idx.size(); ++i) {
        ret_inn += m_inn*idx[     i ];
        ret_out += m_out*idx[perm[i]];

        m_inn *= dims[     i ];
        m_out *= dims[perm[i]];
      }
    }

  private:
    vector<tuple<int,int>> const& rngs;
    vector<int>            const& dims;
    vector<int>            const& perm;

    vector<int>                   idx;
  };

private:
  int min_block_size;
};
