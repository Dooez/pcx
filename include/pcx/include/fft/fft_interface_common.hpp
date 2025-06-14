#pragma once
#include "pcx/include/types.hpp"

namespace pcx {

enum class fft_permutation {
    bit_reversed,
    normal,
    shifted,
};
struct fft_options {
    fft_permutation pt = fft_permutation::normal;

    uZ coherent_size = 0;
    uZ lane_size     = 0;
    uZ node_size     = 8;
    uZ simd_width    = 0;
};
}    // namespace pcx
