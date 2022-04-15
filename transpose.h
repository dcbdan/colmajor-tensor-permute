#pragma once

void naive_hit_inn(int ni, int nj, float* inn, float* out) {
  for(int j = 0; j != nj; ++j) {
  for(int i = 0; i != ni; ++i) {
    out[j + nj*i] = inn[i + ni*j];
  }}
}

void naive_hit_out(int ni, int nj, float* inn, float* out) {
  for(int i = 0; i != ni; ++i) {
  for(int j = 0; j != nj; ++j) {
    out[j + nj*i] = inn[i + ni*j];
  }}
}

struct with_blocks_t {
  with_blocks_t(int block_size): block_size(block_size) {}

  void operator()(int ni, int nj, float* inn, float* out) const {
    int num_block_j = nj / block_size;
    int num_block_i = ni / block_size;

    // Do the portions covered by the blocks
    for(int block_j = 0; block_j != num_block_j; ++block_j) {
    for(int block_i = 0; block_i != num_block_i; ++block_i) {
      int end_j = (block_j+1)*block_size;
      int end_i = (block_i+1)*block_size;
      for(int j = block_j*block_size; j != end_j; ++j) {
      for(int i = block_i*block_size; i != end_i; ++i) {
        out[j + nj*i] = inn[i + ni*j];
      }}
    }}

    // Now there are three more portions..

    // This covers two of those portions.

    // If the blocks covered all of i, the if prevents the
    // outer loop from happening
    int beg_i = num_block_i * block_size;
    if(beg_i != ni) {
      for(int j = 0; j != nj; ++j) {
        for(int i = beg_i; i != ni; ++i) {
          out[j + nj*i] = inn[i + ni*j];
        }
      }
    }

    // This covers the last portion.

    // If the blocks covered all of j, the for outer for
    // loop exits immediately
    int end_i = num_block_i * block_size;
    for(int j = num_block_j * block_size; j != nj; ++j) {
      for(int i = 0; i != end_i; ++i) {
        out[j + nj*i] = inn[i + ni*j];
      }
    }
  }
private:
  int block_size;
};

// This is the cache oblivious algorithm, as exemplified in
//   https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
struct recursive_t {
  recursive_t(int min_block_size): min_block_size(min_block_size) {}

  void operator()(int ni, int nj, float* inn, float* out) const {
    recurse(0, ni, ni, 0, nj, nj, inn, out);
  }
private:
  void recurse(
      int beg_i, int end_i, int const& total_i,
      int beg_j, int end_j, int const& total_j,
      float* inn, float* out) const
  {
    int const& ni = total_i;
    int const& nj = total_j;

    // 1. Check the base case of the recursion
    int remaining_j = end_j - beg_j;
    int remaining_i = end_i - beg_i;

    if(remaining_i <= min_block_size && remaining_j <= min_block_size) {
      for(int j = beg_j; j != end_j; ++j) {
      for(int i = beg_i; i != end_i; ++i) {
        out[j + nj*i] = inn[i + ni*j];
      }}

      return;
    }

    // 2. Pick the larger dimension and recurse
    if(remaining_i > remaining_j) {
      int half_i = beg_i + ((end_i - beg_i) / 2);
      recurse(beg_i, half_i, total_i,
              beg_j, end_j,  total_j,
              inn, out);

      return recurse(half_i, end_i, total_i,
                     beg_j,  end_j, total_j,
                     inn, out);
    } else {
      int half_j = beg_j + ((end_j - beg_j) / 2);
      recurse(beg_i, end_i, total_i,
              beg_j, half_j,  total_j,
              inn, out);

      return recurse(beg_i,  end_i, total_i,
                     half_j, end_j, total_j,
                     inn, out);
    }
  }

private:
  int min_block_size;
};

