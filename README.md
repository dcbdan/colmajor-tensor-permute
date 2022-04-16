# colmajor-tensor-permute

A tensor out-of-place permute implementation for column major ordered tensors.

1. Transpose was implemented using the naive algorithm, using blocks,
   and using the recursive [cache-oblivous](https://en.wikipedia.org/wiki/Cache-oblivious_algorithm)
   algorihtm. The cache oblivious algorithm was the best.
2. The tensor permute is based off of the generalization of the recursive algorithm.

The tensor permute algorihtm recursively splits the input tensor along the largest
dimension until the resulting tensor block falls below a threshold size. Then a
naive permute is applied to the small tensor.

On one particular laptop, the best block size to use is 1024. To compile,
```
g++ -o exp main.cc -std=c++17 -O3
```

The tensor permute has more overhead than the matrix transpose. Some care was taken so that
overhead is kept to a minimum. Transpose with tensor permute is on par with the
the matrix transpose implementation.

One tensor-permute specific transformation that is applied:
A permtuation of `[0,1,4,2,3]` permutation is equivalent to
a `[0,2,1]` permutation, with the dimensions multiplied accordingly.

The biggest performance problem with the tensor-permute is when
the tensor rank is larger than 5. In this case, the base case doesn't use
for loops and instead uses an indexer that keeps track of offsets. The
indexer is slow. An easy fix is to add for loops for ranks greater than 5.
But it would be satisfying to have the indexer match the for loops.
The issue is that compiler optimizations readily know what to do for the
for loops.
