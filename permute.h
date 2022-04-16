#pragma once

#include <vector>
#include <tuple>

using std::vector;
using std::tuple;

#include <iostream>
#define DCB01(x) // std::cout << x << std::endl

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
    vector<int> dims,
    vector<int> perm,
    float* inn,
    float* out) const
  {
    // Some extra tensor-permute optimizations:
    // 1. fuse adjacent dimensions...
    //      so if perm is [2,0,1], fuse [0,1] yielding [1,0]
    // 2. remove dimensions of size 1
    //
    // (There should be at most a handful of fuse and singletons,
    //  so don't worrry about efficiency here)
    DCB01("BEFORE dims, perm " << dims << ", " << perm);

    while(
      dims.size() > 1 &&
      (has_fuse(dims, perm) || has_singleton(dims, perm)))
    {}

    DCB01("AFTER dims, perm " << dims << ", " << perm);

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
      auto const [str_inn, str_out] = build_strides(dims, perm);

      vector<tuple<int,int>> rngs;
      rngs.reserve(dims.size());
      for(auto const& n: dims) {
        rngs.emplace_back(0, n);
      }
      DCB01("NO BATCH");
      recurse(rngs, str_inn, str_out, inn, out);
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

    auto const [str_inn, str_out] = build_strides(batch_dims, batch_perm);

    DCB01("ALL THE BATCHES " << batch_size << " ... " << offset);
    for(int which_batch = 0; which_batch != batch_size; ++which_batch) {
      recurse(batch_rngs, str_inn, str_out, inn, out);
      inn += offset;
      out += offset;
    }
  }

private:
  inline void recurse(
    vector<tuple<int,int>>& rngs,
    vector<int> const& str_inn,
    vector<int> const& str_out,
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
    for(int i = 0; i != rngs.size(); ++i) {
      auto const& [beg, end] = rngs[i];
      int remaining = end - beg;
      block_size *= remaining;

      if(remaining > largest_remaining) {
        largest_remaining = remaining;
        which_recurse = i;
      }
    }

    if(block_size < min_block_size) {

      // Here, directly dispatch the four loops based off of how many
      // dimensions there are.
      //
      // Doing for loops is way faster than using indexer.
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
        // This works for all dimension sizes, but is slower
        indexer_t indexer(rngs, str_inn, str_out);
        do {
          out[indexer.offset_out()] = inn[indexer.offset_inn()];
        } while(indexer.increment());
      }

      return;
    }

    auto [beg, end] = rngs[which_recurse];
    int half = beg + ((end-beg) / 2);

    rngs[which_recurse] = {beg, half};
    recurse(rngs, str_inn, str_out, inn, out);

    rngs[which_recurse] = {half,end};
    recurse(rngs, str_inn, str_out, inn, out);

    // Note: rngs is passed by reference, so on the recursion out,
    //       we have set rngs back to the way it was
    // (Bassing by copy is less to think about, but not as efficient)
    rngs[which_recurse] = {beg, end};
  }

  static
  tuple<
    vector<int>,
    vector<int>>
      build_strides(
        vector<int> const& dims,
        vector<int> const& perm)
  {
    using vec = vector<int>;
    tuple<vec,vec> ret(vec(dims.size()), vec(dims.size()));
    auto& [str_inn, str_out] = ret;

    // set the strides
    int m_inn = 1;
    int m_out = 1;
    for(int i = 0; i != dims.size(); ++i) {
      str_inn[     i ] = m_inn;
      str_out[perm[i]] = m_out;

      m_inn *= dims[     i ];
      m_out *= dims[perm[i]];
    }

    return ret;
  }

private:
  bool has_fuse(vector<int>& dims, vector<int>& perm) const {
    for(int i = 0; i < perm.size()-1; ++i) {
      if(perm[i] + 1 == perm[i+1]) {
        int which = perm[i];
        dims[which] = dims[which] * dims[which+1];
        remove(which+1, dims, perm);
        return true;
      }
    }
    return false;
  }

  bool has_singleton(vector<int>& dims, vector<int>& perm) const {
    for(int i = 0; i < dims.size()-1; ++i) {
      if(dims[i] == 1) {
        remove(i, dims, perm);
        return true;
      }
    }
    return false;
  }

  void remove(int i, vector<int>& dims, vector<int>& perm) const {
    // i = 1
    // [d0,d1,d2,d3,d4]
    // [d0,d2,d3,d4]     <- copy over
    // [d0,d2,d3]        <- resize
    for(int x = i; x < dims.size()-1; ++x) {
      dims[x] = dims[x+1];
    }
    dims.resize(dims.size()-1);

    // [3,1,2,4,0]
    // [3,2,4,0,0] <- removed
    // [3,2,4,0]   <- resized
    // [2,1,3,0]   <- decremented

    // find where x lives
    int x = 0;
    for(; x != perm.size(); ++x) {
      if(perm[x] == i) {
        break;
      }
    }

    // shift to the left and resize
    for(; x < perm.size()-1; ++x) {
      perm[x] = perm[x+1];
    }
    perm.resize(perm.size()-1);

    // decrement things greater than i
    for(auto& p: perm) {
      if(p > i) {
        p--;
      }
    }
  }


  struct indexer_t {
    indexer_t(
      vector<tuple<int,int>> const& rngs,
      vector<int>            const& str_inn,
      vector<int>            const& str_out):
        rngs(rngs), str_inn(str_inn), str_out(str_out),
        off_inn(0), off_out(0)
    {
      idx.reserve(rngs.size());
      for(int i = 0; i != rngs.size(); ++i) {
        auto const& [beg, _] = rngs[i];
        idx.push_back(beg);
        off_inn += beg * str_inn[i];
        off_out += beg * str_out[i];
      }
    }

    // 0 0 0 -> increment i0, which = 0
    // 1 0 0 -> increment i0, which = 1
    // 2 0 0 -> reset i0, increment i1, which = 0
    // 0 1 0 -> increment i0, which = 0
    // 1 1 0 -> increment i0, which = 1
    // 2 1 0 -> reset i0, increment i1, which = 0
    // 0 2 0 -> increment i0, which = 0
    // 1 2 0 -> increment i0, which = 2
    // 2 2 0 -> reset i0, reset i1, increment i2, which = 0
    // 0 0 1 -> increment i0, which = 0
    // 1 0 1 -> ....

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
