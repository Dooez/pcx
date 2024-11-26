#pragma once
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"
#include "pcx/include/tuple.hpp"
#include "pcx/include/types.hpp"

#include <array>
namespace pcx::simd {

constexpr struct btfly_t {
    PCX_AINLINE static auto operator()(any_cx_vec auto a, any_cx_vec auto b) {
        return std::make_tuple(add(a, b), sub(a, b));
        // return std::make_tuple(b, a);
    }
} btfly{};
}    // namespace pcx::simd

namespace pcx::detail_ {

template<typename T>
inline auto wnk(uZ n, uZ k) -> std::complex<T> {
    constexpr double pi = 3.14159265358979323846;
    if (n == k * 4)
        return {0, -1};
    if (n == k * 2)
        return {-1, 0};
    return exp(std::complex<T>(0, -2 * pi * static_cast<double>(k) / static_cast<double>(n)));
}
consteval auto log2i(u64 num) -> uZ {
    u64 order = 0;
    for (u8 shift = 32; shift > 0; shift /= 2) {
        if (num >> shift > 0) {
            order += num >> shift > 0 ? shift : 0;
            num >>= shift;
        }
    }
    return order;
}

consteval auto powi(u64 num, u64 pow) -> u64 {    // NOLINT(*recursion*)
    auto res = (pow % 2) == 1 ? num : 1UL;
    if (pow > 1) {
        auto half_pow = powi(num, pow / 2UL);
        res *= half_pow * half_pow;
    }
    return res;
}

constexpr auto reverse_bit_order(u64 num, u64 depth) -> u64 {
    if (depth == 0)
        return 0;
    //NOLINTBEGIN(*magic-numbers*)
    num = num >> 32U | num << 32U;
    num = (num & 0xFFFF0000FFFF0000U) >> 16U | (num & 0x0000FFFF0000FFFFU) << 16U;
    num = (num & 0xFF00FF00FF00FF00U) >> 8U | (num & 0x00FF00FF00FF00FFU) << 8U;
    num = (num & 0xF0F0F0F0F0F0F0F0U) >> 4U | (num & 0x0F0F0F0F0F0F0F0FU) << 4U;
    num = (num & 0xCCCCCCCCCCCCCCCCU) >> 2U | (num & 0x3333333333333333U) << 2U;
    num = (num & 0xAAAAAAAAAAAAAAAAU) >> 1U | (num & 0x5555555555555555U) << 1U;
    //NOLINTEND(*magic-numbers*)
    return num >> (64 - depth);
}

template<uZ Size, bool DecInTime>
struct order {
    static constexpr auto data = [] {
        std::array<uZ, Size> order;
        for (uZ i = 0; i < Size; ++i) {
            if constexpr (DecInTime) {
                order[i] = reverse_bit_order(i, log2i(Size));
            } else {
                order[i] = i;
            }
        }
        return order;
    }();

    static constexpr auto tw = []<uZ... N>(std::index_sequence<N...>) -> std::array<uZ, sizeof...(N)> {
        if constexpr (DecInTime) {
            return {(N > 0 ? (1U << log2i(N + 1)) - 1
                                 + reverse_bit_order(1 + N - (1U << log2i(N + 1)), log2i(N + 1))
                           : 0)...};
        } else {
            return {N...};
        }
    }
    (std::make_index_sequence<Size - 1>{});
};


template<uZ NodeSize, typename T, uZ Width>
struct btfly_node {
    using cx_vec = simd::cx_vec<T, false, false, Width>;

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool conj_tw;
        bool dit;
    };

    using pruned_tw_t = std::conditional_t<(NodeSize > 4),    //
                                           std::array<cx_vec, NodeSize / 4 - 1>,
                                           decltype([] {})>;
    using dest_t      = tupi::broadcast_tuple_t<T*, NodeSize>;
    using src_t       = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using full_tw_t   = tupi::broadcast_tuple_t<cx_vec, NodeSize - 1>;

    template<settings Settings>
    PCX_AINLINE static void perform(const dest_t& dest, const pruned_tw_t& tw) {
        auto data_tup = load<Settings>(dest);
    };

    // template<settings Settings>
    // PCX_AINLINE static void perform(const dest_t& dest, const src_t& src, const full_tw_t& tw) {
    //     auto data_tup = load<Settings>(src);
    //     // TODO(Timofey): inverse / conj
    //
    //     auto p1 = data_tup;
    //
    //     auto res = []<uZ L = 0>(this auto&& f, const auto& data, const auto& tw, uZ_constant<L> = {}) {
    //         if constexpr (powi(2, L + 1) == NodeSize) {
    //             return btfly<L>(data, tw);
    //         } else {
    //             auto tmp = btfly<L>(data, tw);
    //             return f(tmp, tw, uZ_constant<L + 1>{});
    //         }
    //     }(p1, tw);
    //
    //     // TODO(Timofey): inverse / conj
    //     store<Settings>(dest, res);
    // }

    template<settings Settings>
    PCX_AINLINE static void perform(const dest_t& dest, const full_tw_t& tw) {
        auto data     = tupi::group_invoke(simd::cxload<Settings.pack_src, Width>, dest);
        data          = reorder<Settings.dit>(data);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        // conj/inverse

        auto res = []<uZ Size = 1> PCX_LAINLINE(this auto f,    //
                                                auto      data,
                                                auto      tw,
                                                uZ_constant<Size> = {}) {
            if constexpr (Size == NodeSize / 2) {
                return btfly<Size>(data, tw);
            } else {
                auto tmp = btfly<Size>(data, tw);
                return f(tmp, tw, uZ_constant<Size * 2>{});
            }
        }(data_rep, tw);

        // conj/inverse
        res           = reorder<Settings.dit>(res);
        auto res_eval = tupi::group_invoke(simd::evaluate, res);
        auto res_rep  = tupi::group_invoke(simd::repack<Settings.pack_dest>, res_eval);
        tupi::group_invoke(simd::cxstore<Settings.pack_dest>, dest, res_rep);
    }

private:
    template<bool Dit>
    PCX_AINLINE static auto reorder(auto data) {
        return []<uZ... Is> PCX_LAINLINE(auto data, std::index_sequence<Is...>) {
            constexpr auto& data_idx = order<NodeSize, Dit>::data;
            return tupi::make_tuple(tupi::get<data_idx[Is]>(data)...);
        }(data, std::make_index_sequence<NodeSize>{});
    }

    /**
     * @brief Extracts two halves of the tuple.
     *
     * data = [0, 1, ..., N - 1]
     * lo   = [0,          1,              ..., Stride / 2 - 1, Stride        , Stride + 1,         ... ]
     * hi   = [Stride / 2, Stride / 2 + 1, ..., Stride - 1    , Stride * 3 / 2, Stride * 3 / 2 + 1, ... ]
     *
     * @return [lo, hi] - a tuple of tuples
     */
    template<uZ Stride, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto extract_halves(tupi::tuple<Ts...> data) {
        constexpr uZ   size     = sizeof...(Ts);
        constexpr auto get_half = []<uZ... Grp, uZ Start> PCX_LAINLINE(auto data,
                                                                       std::index_sequence<Grp...>,
                                                                       uZ_constant<Start>) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset> PCX_LAINLINE(auto data,
                                                                             std::index_sequence<Iters...>,
                                                                             uZ_constant<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(data)...);
            };
            return tupi::tuple_cat(iterate(data,
                                           std::make_index_sequence<Stride / 2>{},
                                           uZ_constant<Start + Grp * Stride>{})...);
        };
        return tupi::make_tuple(
            get_half(data, std::make_index_sequence<size / Stride>{}, uZ_constant<0>{}),
            get_half(data, std::make_index_sequence<size / Stride>{}, uZ_constant<Stride / 2>{}));
    }

    /**
     * @brief Combines two halves into a tuple
     *
     * lo     = [0,          1,              ..., Stride / 2 - 1, Stride        , Stride + 1,         ... ]
     * hi     = [Stride / 2, Stride / 2 + 1, ..., Stride - 1    , Stride * 3 / 2, Stride * 3 / 2 + 1, ... ]
     * return = [0, 1, ..., N - 1] 
     */
    template<uZ Stride, simd::any_cx_vec... Tsl, simd::any_cx_vec... Tsh>
    PCX_AINLINE static auto combine_halves(tupi::tuple<Tsl...> lo, tupi::tuple<Tsh...> hi) {
        constexpr uZ size = sizeof...(Tsl) * 2;
        return []<uZ... Grp> PCX_LAINLINE(auto lo, auto hi, std::index_sequence<Grp...>) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset> PCX_LAINLINE(auto lo,
                                                                             auto hi,
                                                                             std::index_sequence<Iters...>,
                                                                             uZ_constant<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(lo)..., tupi::get<Offset + Iters>(hi)...);
            };
            return tupi::tuple_cat(iterate(lo,    //
                                           hi,
                                           std::make_index_sequence<Stride / 2>{},
                                           uZ_constant<Grp * Stride / 2>{})...);
        }(lo, hi, std::make_index_sequence<size / Stride>{});
    }

    /**
     * @brief Performs a single level of butterfly operation.
     *
     * @tparam Level 
     */
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto btfly(tupi::tuple<Ts...> data, full_tw_t tw) {
        auto tws = []<uZ... Itw> PCX_LAINLINE(std::index_sequence<Itw...>, auto tw) {
            constexpr auto repeats = NodeSize / 2 / sizeof...(Itw);
            constexpr auto start   = Size - 1;
            return tupi::tuple_cat(tupi::make_broadcast_tuple<repeats>(tupi::get<start + Itw>(tw))...);
        }(std::make_index_sequence<Size>{}, tw);

        constexpr auto stride = NodeSize / Size;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto hi_tw     = tupi::group_invoke(simd::mul, hi, tws);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi_tw);
        auto new_lo    = tupi::group_invoke([](auto p) { return tupi::get<0>(p); }, btfly_res);
        auto new_hi    = tupi::group_invoke([](auto p) { return tupi::get<1>(p); }, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };

    static constexpr auto next_pow_2(u64 v) {
        v--;
        v |= v >> 1U;
        v |= v >> 2U;
        v |= v >> 4U;
        v |= v >> 8U;
        v |= v >> 16U;
        v |= v >> 32U;
        v++;
        return v;
    }

    template<uZ Size>
    struct const_btfly_t {
        /** Indexes are bit-reversed, and thus independent on the actual transform size e.g.
         * if twiddle is defined as tw = exp(-2 * pi * i * k / N)
         * `ITw == 0` => k == 0
         * `ITw == 1` => k == N/4
         * `ITw == 2` => k == N/8
         * `ITw == 3` => k == 3N/8
         * ...
         */
        static inline auto const_tw = []<uZ... Is>(std::index_sequence<Is...>) {
            constexpr auto get_tw = []<uZ I>(uZ_constant<I>) {
                if constexpr (I < 2) {
                    return imag_unit<I>;
                } else {
                    constexpr auto N = next_pow_2(I + 1) * 2;
                    constexpr auto K = (I - N / 4) * 2 + 1;
                    return wnk<T>(N, K);
                }
            };
            return tupi::make_broadcast_tuple<Size / 2>(imag_unit<0>);
        }(std::make_index_sequence<Size / 2>{});

        template<uZ I>
        PCX_AINLINE constexpr static auto tw() {
            if constexpr (I < 2) {
                return imag_unit<I>;
            } else {
                return simd::cxbroadcast<1, Width>(&tupi::get<I>(const_tw));
            }
        }

        template<simd::any_cx_vec... Ts>
        PCX_AINLINE auto operator()(tupi::tuple<Ts...> data) const {
            constexpr auto count  = sizeof...(Ts);
            constexpr auto stride = NodeSize / Size;

            auto tws = []<uZ... Is>(std::index_sequence<Is...>) {
                return tupi::tuple_cat(tupi::make_broadcast_tuple<count / Size>(tw<Is>())...);
            }(std::make_index_sequence<Size>{});

            auto [lo, hi]  = extract_halves<stride>(data);
            auto hi_tw     = tupi::group_invoke(simd::mul, hi, tws);
            auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi_tw);
            auto new_lo    = tupi::group_invoke([](auto p) { return tupi::get<0>(p); }, btfly_res);
            auto new_hi    = tupi::group_invoke([](auto p) { return tupi::get<1>(p); }, btfly_res);
            return combine_halves<stride>(new_lo, new_hi);
        };
    };
};

}    // namespace pcx::detail_
