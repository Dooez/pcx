

## Single load
`single_load` referes to the stages of the fft that process data that entirily fits in `NodeSize` simd vectors.
The first step of forward `single_load` is the same as normal butterfly of `NodeSize`.
Then, the length of contiguous data corresponding to a single fft group (`k`) becomes less then simd vector width. 
The data must be regrouped. Example for vectors of width 8:
```
input data:     [0 0 0 0 2 2 2 2] [1 1 1 1 3 3 3 3]

regrouped data: [0 0 0 0 1 1 1 1] [2 2 2 2 3 3 3 3]
twiddles:                         [0 0 0 0 1 1 1 1]
```
The data can be viewed as chunks, with each chunk corresponding to a single fft group.
The vectors are split into even end odd chunks, then the chunks are inteleaved.
The chunk size starts at half the vector width.

After each fft step the regrouping must be repeated with chunk size reduced by half.
```
input data:     [0 0 1 1 2 2 3 3] [4 4 5 5 6 6 7 7]
regrouped data: [0 0 4 4 2 2 6 6] [1 1 5 5 3 3 7 7]
twiddles:                         [0 0 2 2 1 1 3 3]
```
This regrouping arranges the data chunks in bit-reversed order. 
This regrouping is implemented by `split_regroup` member of `subtransform` class.

After the final step, the data must be returned to the natural order.
After performing the regrouping stages on the element indices, we get the following order:
```
normal indices:    [ 0  1  2  3  4  5  6  7] [ 8  9 10 11 12 13 14 15]
regrouped indices: [ 0  8  1  9  2 10  3 11] [ 4 12  5 13  6 14  7 15]
```
The reverse reordering is implemented by `regroup` member of `subtransform` class.
