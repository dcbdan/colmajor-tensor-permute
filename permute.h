#pragma once

#include <vector>
#include <tuple>

using std::vector;
using std::tuple;

#include <iostream>
#define DCB01(x)  // std::cout << x << std::endl

#define __FST(x) std::get<0>(x)
#define __SND(x) std::get<1>(x)

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
      indexer_t indexer(rngs, dims, perm);

      vector<int> str_inn = indexer.str_inn;
      vector<int> str_out = indexer.str_out;

      // Here, we directly dispatch the four loops based off of how many
      // dimensions there are.
      //
      // Doing for loops is way faster than doing the indexer thing.
      //
      if(rngs.size() == 2) {
        for(int i1 = __FST(rngs[1]); i1 != __SND(rngs[1]); ++i1) {
        for(int i0 = __FST(rngs[0]); i0 != __SND(rngs[0]); ++i0) {
          out[i0*str_out[0] + i1*str_out[1]] =
          inn[i0*str_inn[0] + i1*str_inn[1]] ;
        }}
      } else
      if(rngs.size() == 3) {
        for(int i2 = __FST(rngs[2]); i2 != __SND(rngs[2]); ++i2) {
        for(int i1 = __FST(rngs[1]); i1 != __SND(rngs[1]); ++i1) {
        for(int i0 = __FST(rngs[0]); i0 != __SND(rngs[0]); ++i0) {
          out[i0*str_out[0] + i1*str_out[1] + i2*str_out[2]] =
          inn[i0*str_inn[0] + i1*str_inn[1] + i2*str_inn[2]] ;
        }}}
      } else
      if(rngs.size() == 4) {
        for(int i3 = __FST(rngs[3]); i3 != __SND(rngs[3]); ++i3) {
        for(int i2 = __FST(rngs[2]); i2 != __SND(rngs[2]); ++i2) {
        for(int i1 = __FST(rngs[1]); i1 != __SND(rngs[1]); ++i1) {
        for(int i0 = __FST(rngs[0]); i0 != __SND(rngs[0]); ++i0) {
          out[i0*str_out[0] + i1*str_out[1] + i2*str_out[2] + i3*str_out[3]] =
          inn[i0*str_inn[0] + i1*str_inn[1] + i2*str_inn[2] + i3*str_inn[3]] ;
        }}}}
      } else
      if(rngs.size() == 5) {
        for(int i4 = __FST(rngs[4]); i4 != __SND(rngs[4]); ++i4) {
        for(int i3 = __FST(rngs[3]); i3 != __SND(rngs[3]); ++i3) {
        for(int i2 = __FST(rngs[2]); i2 != __SND(rngs[2]); ++i2) {
        for(int i1 = __FST(rngs[1]); i1 != __SND(rngs[1]); ++i1) {
        for(int i0 = __FST(rngs[0]); i0 != __SND(rngs[0]); ++i0) {
          out[i0*str_out[0] + i1*str_out[1] + i2*str_out[2] + i3*str_out[3] + i4*str_out[4]] =
          inn[i0*str_inn[0] + i1*str_inn[1] + i2*str_inn[2] + i3*str_inn[3] + i4*str_inn[4]] ;
        }}}}}
      } else {
        do {
          out[indexer.offset_out()] = inn[indexer.offset_inn()];
        } while(indexer.increment());
      }

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
        rngs(rngs), str_inn(dims.size()), str_out(dims.size()),
        off_inn(0), off_out(0)
    {
      // set the strides
      int m_inn = 1;
      int m_out = 1;
      for(int i = 0; i != rngs.size(); ++i) {
        str_inn[     i ] = m_inn;
        str_out[perm[i]] = m_out;

        m_inn *= dims[     i ];
        m_out *= dims[perm[i]];
      }

      idx.reserve(rngs.size());
      for(int i = 0; i != rngs.size(); ++i) {
        auto const& [beg, _] = rngs[i];
        idx.push_back(beg);
        off_inn += beg * str_inn[i];
        off_out += beg * str_out[i];
      }
    }

    // 0 0
    //        incrment 0
    // 1 0
    //        increment 0
    // 2 0
    //        reset 0
    //        increment 1
    // 0 1
    //        increment 0
    // 1 1
    //        increment 0
    // 2 1
    //        reset 0
    //        reset 1
    //        fail
    inline bool increment() {
      for(int i = 0; i < idx.size(); ++i) {
        if(idx[i] + 1 == std::get<1>(rngs[i])) {
          // Let rngs[i] = (3,9)
          // Then idx[i] = 8
          //      diff = 5
          //      idx[i] -= 5 => idx[i] = 3
          // Moreover, the offsets had to have been incremented 5 times as well,
          // so subtract that from the offsets
          int diff = std::get<1>(rngs[i]) - std::get<0>(rngs[i]) - 1;
          idx[i]  -=            diff;
          off_inn -= str_inn[i]*diff;
          off_out -= str_out[i]*diff;
        } else {
          idx[i] += 1;
          off_inn += str_inn[i];
          off_out += str_out[i];
          return true;
        }
      }
      return false;
    }

    inline int const& offset_inn() const { return off_inn; }
    inline int const& offset_out() const { return off_out; }

    vector<tuple<int,int>> const& rngs;

    vector<int> idx;

    vector<int> str_inn;
    vector<int> str_out;

    int off_inn;
    int off_out;
  };

private:
  int min_block_size;
};
