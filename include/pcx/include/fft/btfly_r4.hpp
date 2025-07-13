#pragma once
#include "pcx/include/fft/btfly_common.hpp"
#include "pcx/include/fft/util.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"

namespace pcx::detail_ {
template<uZ NodeSize, typename T, uZ Width>
    requires(NodeSize >= 2)
struct btfly_r4_node_dit {
    template<simd::any_cx_vec... Ts>
    PCX_AINLINE static auto btfly4_impl(meta::ce_of<uZ> auto size,    //
                                        tupi::tuple<Ts...>   data,
                                        auto                 tws,
                                        auto                 conj_tw) {
        constexpr auto stride = NodeSize / size * 2;

        auto maybe_conj = [=](auto tw) {
            if constexpr (conj_tw) {
                return conj(tw);
            } else {
                return tw;
            }
        };
        auto mul_by_nj = [=](auto v) { return simd::mul(v, imag_unit<-1>); };

        auto quarts       = extract_quarts<stride>(data);    // tuple<tuple<>, tuple<>, tuple<>, tuple<>>
        auto tw_c         = tupi::group_invoke(tupi::group_invoke(maybe_conj), tws);
        auto [a, b, c, d] = tupi::group_invoke(tupi::group_invoke(simd::mul), quarts, tw_c);
        auto [xy, zw]     = tupi::group_invoke(tupi::group_invoke(simd::btfly),
                                           tupi::make_tuple(a, c),
                                           tupi::make_tuple(b, d));
        auto z            = tupi::group_invoke(tupi::get_copy<0>, zw);
        auto w            = tupi::group_invoke(tupi::get_copy<1>, zw);
        auto wnj          = tupi::group_invoke(mul_by_nj, w);
        auto res_tup      = tupi::group_invoke(tupi::group_invoke(simd::btfly), xy, tupi::make_tuple(z, wnj));
        auto [h, j]       = tupi::get<0>(res_tup);
        auto [k, l]       = tupi::get<1>(res_tup);
        return combine_quarts<stride>(h, j, k, l);
    };
};
}    // namespace pcx::detail_
