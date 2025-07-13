#pragma once
#include "pcx/include/fft/util.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"

namespace pcx::detail_ {

template<typename T, uZ NodeSizeL>
static auto make_tw_node(uZ fft_size, uZ k) {
    constexpr auto n_tw = NodeSizeL / 2;

    auto tw_node = std::array<std::complex<T>, n_tw>{};
    uZ   i_tw    = 0;
    for (uZ l: stdv::iota(0U, log2i(NodeSizeL))) {
        for (uZ i: stdv::iota(0U, powi(2, l))) {
            if (i % 2 == 1)
                continue;
            auto tw          = pcx::detail_::wnk_br<T>(fft_size, k + i);
            tw_node.at(i_tw) = tw;
            ++i_tw;
        }
        k *= 2;
        fft_size *= 2;
    }
    return tw_node;
}

}    // namespace pcx::detail_
