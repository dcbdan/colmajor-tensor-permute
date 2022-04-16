#include <iostream>
#include <vector>
#include <chrono>
#include <tuple>
#include <string>
#include <functional>

#include "transpose.h"
#include "permute.h"

using std::vector;
using std::tuple;
using std::string;
using std::function;


struct raii_timer_t {
  raii_timer_t(std::string msg): msg(msg) {
    start = std::chrono::high_resolution_clock::now();
  }

  ~raii_timer_t() {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << msg << ": " << duration.count() << "ms" << std::endl;
  }

  decltype(std::chrono::high_resolution_clock::now()) start;
  std::string msg;
};

template <typename T>
T product(vector<T> const& ts) {
  T ret = 1;
  for(T const& t: ts) {
    ret *= t;
  }
  return ret;
}

struct tensor_t {
  tensor_t(std::vector<int> dims, float* data):
    dims(dims), own(false), data(data)
  {}

  tensor_t(std::vector<int> dims):
    dims(dims), own(true)
  {
    int sz = product(dims);
    data = new float[sz];
    std::fill(data, data + sz, 0.0);
  }

  float& operator[](std::vector<int> const& idx) {
    int p = 1;
    int total = 0;
    for(int i = 0; i != dims.size(); ++i) {
      total += p*idx[i];
      p *= dims[i];
    }
    return data[total];
  }

  int size() const {
    return product(dims);
  }

  ~tensor_t() {
    if(own)
      delete[] data;
  }

  vector<int> dims;
  float* data;
  bool own;
};

struct indexer_t {
  indexer_t(vector<int> const& szs_):
    szs(szs_),
    idx(szs_.size(), 0)
  {}

  int operator()() const {
    int p = 1;
    int total = 0;
    for(int i = 0; i != idx.size(); ++i) {
      total += p*idx[i];
      p *= szs[i];
    }
    return total;
  }

  bool increment() {
    for(int i = 0; i < idx.size(); ++i) {
      if(idx[i] + 1 == szs[i]) {
        idx[i] = 0;
      } else {
        idx[i] += 1;
        return true;
      }
    }
    return false;
  }

  vector<int> const szs;
  vector<int>       idx;
};

// out_permutation = [2,0,1] implies we have ijk->kij
template <typename T>
vector<T> permute(vector<int> const& permutation, vector<T> const& xs) {
  vector<T> ret;
  ret.reserve(xs.size());
  for(auto const& idx: permutation) {
    ret.push_back(xs[idx]);
  }
  return ret;
}

bool check(vector<int> out_permutation, tensor_t& inn, tensor_t& out) {
  indexer_t indexer(inn.dims);

  do {
    auto const& idx_inn = indexer.idx;
    vector<int> idx_out = permute(out_permutation, idx_inn);

    if(inn[idx_inn] != out[idx_out]) {
      return false;
    }

  } while(indexer.increment());

  return true;
}

using transpose_f = function<void(int,int,float*,float*)>;

void test_transpose(int ni, int nj, transpose_f f) {
  tensor_t inn({ni,nj});
  tensor_t out({nj,ni});

  for(int i = 0; i != ni; ++i) {
  for(int j = 0; j != nj; ++j) {
    inn[{i,j}] = 1 + i + 7*j;
  }}

  std::cout << "Test [num rows = " << ni << ", num cols = " << nj << "]" << std::endl;

  {
    raii_timer_t timer("Time(ms)");
    f(ni, nj, inn.data, out.data);
  }

  std::cout << "Was it correct? " << (check({1,0}, inn, out) ? "yes" : "no") << std::endl;
}

using permute_f = function<void(vector<int>, vector<int>, float*, float*)>;

void test_permutation(vector<int> dims, vector<int> perm, permute_f f) {
  tensor_t inn(dims);
  tensor_t out(permute(perm, dims));

  indexer_t indexer(dims);
  float cnt = 1.0;
  do {
    inn[indexer.idx] = cnt++;
  } while(indexer.increment());

  std::cout << "Test dims = " << dims << std::endl;

  f(dims, perm, inn.data, out.data);

  std::cout << "Was it correct? " << (check(perm, inn, out) ? "yes" : "no") << std::endl;
}

void performance_transpose(
  int repeat,
  int ni, int nj,
  vector<tuple<string, transpose_f>> tests)
{
  std::cout << "Initializing..." << std::endl;

  tensor_t inn({ni,nj});
  tensor_t out({nj,ni});

  for(int i = 0; i != ni; ++i) {
  for(int j = 0; j != nj; ++j) {
    inn[{i,j}] = 1 + i + 7*j;
  }}

  std::cout << "Running..." << std::endl;
  for(auto const& [msg, f]: tests) {
    for(int i = 0; i != repeat; ++i) {
      raii_timer_t timer(msg);
      f(ni, nj, inn.data, out.data);
    }
  }

}

void performance_permute(
  int repeat,
  vector<int> dims,
  vector<int> perm,
  vector<tuple<string, permute_f>> tests)
{
  std::cout << "Initializing..." << std::endl;

  tensor_t inn(dims);
  tensor_t out(permute(perm, dims));

  indexer_t indexer(dims);
  float cnt = 1.0;
  do {
    inn[indexer.idx] = cnt++;
  } while(indexer.increment());

  std::cout << "Running..." << std::endl;
  for(auto const& [msg, f]: tests) {
    for(int i = 0; i != repeat; ++i) {
      raii_timer_t timer(msg);
      f(dims, perm, inn.data, out.data);
    }
  }
}

void exp01() {
  vector<char> x{'i','j','k'};
  std::cout << x << "->" << permute({2,0,1}, x) << std::endl;
}

void exp02() {
  int nx = 5;
  int ny = 7;

  auto test = test_transpose;

  std::cout << "Naive hit inn -----------------------------------\n";
  test(nx, ny, naive_hit_inn);
  std::cout << std::endl;

  std::cout << "Naive hit out -----------------------------------\n";
  test(nx, ny, naive_hit_inn);
  std::cout << std::endl;

  std::cout << "With blocks 1 -----------------------------------\n";
  test(nx, ny, with_blocks_t(1));
  std::cout << std::endl;

  std::cout << "With blocks 2 -----------------------------------\n";
  test(nx, ny, with_blocks_t(2));
  std::cout << std::endl;

  std::cout << "With blocks 4 -----------------------------------\n";
  test(nx, ny, with_blocks_t(4));
  std::cout << std::endl;

  std::cout << "With blocks 5 -----------------------------------\n";
  test(nx, ny, with_blocks_t(5));
  std::cout << std::endl;

  std::cout << "With blocks 7 -----------------------------------\n";
  test(nx, ny, with_blocks_t(7));
  std::cout << std::endl;

  std::cout << "Recursive 3   -----------------------------------\n";
  test(nx, ny, recursive_t(3));
  std::cout << std::endl;

  std::cout << "Recursive 1   -----------------------------------\n";
  test(nx, ny, recursive_t(1));
  std::cout << std::endl;

  std::cout << "Recursive 7   -----------------------------------\n";
  test(nx, ny, recursive_t(7));
  std::cout << std::endl;
}

void exp03() {
  std::cout << "Permute 1024, rank 4" << std::endl;
  test_permutation({4,5,6,7,8}, {4,3,2,1,0}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, rank 2" << std::endl;
  test_permutation({4,5}, {1,0}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, rank 2" << std::endl;
  test_permutation({2,3}, {1,0}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, should copy" << std::endl;
  test_permutation({4,5}, {0,1}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, rank 3 batch" << std::endl;
  test_permutation({4,5,6}, {1,0,2}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, rank 3" << std::endl;
  test_permutation({4,5,6}, {2,0,1}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 1024, rank 4" << std::endl;
  test_permutation({4,5,6,7}, {2,3,1,0}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute 16, rank 4" << std::endl;
  test_permutation({4,5,6,7,8}, {4,3,2,1,0}, permute_t(16));
  std::cout << std::endl;

  std::cout << "Permute, ones 1" << std::endl;
  test_permutation({1,1,1,1,1}, {4,3,2,1,0}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute, ones 2" << std::endl;
  test_permutation({1,1,1,1,1}, {4,2,3,0,1}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute, fuse 1" << std::endl;
  test_permutation({2,2,2,2,2}, {4,0,1,2,3}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute, fuse 2" << std::endl;
  test_permutation({7,1,2,3,4}, {4,0,1,2,3}, permute_t(1024));
  std::cout << std::endl;

  std::cout << "Permute, fuse 3" << std::endl;
  test_permutation({2,2,2,2,2}, {0,1,4,2,3}, permute_t(1024));
  std::cout << std::endl;
}



int main() {
  exp03();

  int nx = 8000;
  int ny = 20000;

  using tuple_tr_t = tuple<string, transpose_f>;
  performance_transpose(1, nx, ny,
    {
      //tuple_tr_t("naive_hit_out", naive_hit_out),
      //tuple_tr_t("naive_hit_inn", naive_hit_inn),
      //tuple_tr_t("with blocks 8", with_blocks_t(8)),
      //tuple_tr_t("with blocks 12", with_blocks_t(12)),
      //tuple_tr_t("with blocks 16", with_blocks_t(16)),
      //tuple_tr_t("with blocks 64", with_blocks_t(64)),
      //tuple_tr_t("with blocks 128", with_blocks_t(128)),
      //tuple_tr_t("with blocks 1024", with_blocks_t(1024)),
      tuple_tr_t("recursive 8", recursive_t(8)),
      tuple_tr_t("recursive 32", recursive_t(32)),
      tuple_tr_t("recursive 64", recursive_t(64)),
    });

  using tuple_pm_t = tuple<string, permute_f>;
  performance_permute(1, {nx,ny}, {1,0},
    {
      tuple_pm_t("permute 32", permute_t(32)),
      tuple_pm_t("permute 64", permute_t(64)),
      tuple_pm_t("permute 128", permute_t(128)),
      tuple_pm_t("permute 256", permute_t(256)),
      tuple_pm_t("permute 1024", permute_t(1024)),
      tuple_pm_t("permute 2048", permute_t(2048)),
      tuple_pm_t("permute 4096", permute_t(4096)),
      tuple_pm_t("permute 8192", permute_t(8192)),
      tuple_pm_t("permute big", permute_t(10000000)),
    });

}
