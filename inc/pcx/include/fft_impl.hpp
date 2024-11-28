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
    }
} btfly{};
}    // namespace pcx::simd

namespace pcx::detail_ {

template<typename T>
inline auto wnk(uZ n, uZ k) -> std::complex<T> {
    constexpr auto pi = std::numbers::pi;
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
constexpr auto next_pow_2(u64 v) {
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

template<uZ NodeSize, typename T, uZ Width>
struct btfly_node_dit {
    using cx_vec = simd::cx_vec<T, false, false, Width>;

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool reverse;
    };

    using dest_t = tupi::broadcast_tuple_t<T*, NodeSize>;
    using src_t  = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using tw_t   = tupi::broadcast_tuple_t<cx_vec, NodeSize - 1>;

    template<settings Settings>
    PCX_AINLINE static void perform(const dest_t& dest, const src_t& src, const tw_t& tw) {
        if constexpr (Settings.reverse) {
            perform_reverse_impl<Settings.pack_dest, Settings.pack_src>(dest, src, make_tw_getter(tw));
        } else {
            perform_impl<Settings.pack_dest, Settings.pack_src>(dest, src, make_tw_getter(tw));
        }
    }
    template<settings Settings>
    PCX_AINLINE static void perform(const dest_t& dest, const tw_t& tw) {
        if constexpr (Settings.reverse) {
            perform_reverse_impl<Settings.pack_dest, Settings.pack_src>(dest, dest, make_tw_getter(tw));
        } else {
            perform_impl<Settings.pack_dest, Settings.pack_src>(dest, dest, make_tw_getter(tw));
        }
    }
    template<settings Settings>
    PCX_AINLINE static void perform_lo_k(const dest_t& dest, const src_t& src) {
        if constexpr (Settings.reverse) {
            perform_reverse_impl<Settings.pack_dest, Settings.pack_src>(dest, src, const_tw_getter);
        } else {
            perform_impl<Settings.pack_dest, Settings.pack_src>(dest, src, const_tw_getter);
        }
    }
    template<settings Settings>
    PCX_AINLINE static void perform_lo_k(const dest_t& dest) {
        if constexpr (Settings.reverse) {
            perform_reverse_impl<Settings.pack_dest, Settings.pack_src>(dest, dest, const_tw_getter);
        } else {
            perform_impl<Settings.pack_dest, Settings.pack_src>(dest, dest, const_tw_getter);
        }
    }

private:
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void perform_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize, Width>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = 2> PCX_LAINLINE(this auto f,    //
                                                auto      data,
                                                auto      get_tw,
                                                uZ_constant<Size> = {}) {
            if constexpr (Size == NodeSize) {
                return btfly_impl<Size>(data, get_tw(uZ_constant<Size>{}));
            } else {
                auto tmp = btfly_impl<Size>(data, get_tw(uZ_constant<Size>{}));
                return f(tmp, get_tw, uZ_constant<Size * 2>{});
            }
        }(data_rep, get_tw);
        auto res_eval = tupi::group_invoke(simd::evaluate, res);
        auto res_rep  = tupi::group_invoke(simd::repack<DestPackSize>, res_eval);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void perform_reverse_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = NodeSize> PCX_LAINLINE(this auto f,    //
                                                       auto      data,
                                                       auto      get_tw,
                                                       uZ_constant<Size> = {}) {
            if constexpr (Size == 2) {
                return rbtfly_impl<Size>(data, get_tw(uZ_constant<Size>{}));
            } else {
                auto tmp = rbtfly_impl<Size>(data, get_tw(uZ_constant<Size>{}));
                return f(tmp, get_tw, uZ_constant<Size / 2>{});
            }
        }(data_rep, get_tw);
        auto res_eval = tupi::group_invoke(simd::evaluate, res);
        auto res_rep  = tupi::group_invoke(simd::repack<DestPackSize>, res_eval);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto btfly_impl(tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto hi_tw     = tupi::group_invoke(simd::mul, hi, tws);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi_tw);
        auto new_lo    = tupi::group_invoke([](auto p) { return tupi::get<0>(p); }, btfly_res);
        auto new_hi    = tupi::group_invoke([](auto p) { return tupi::get<1>(p); }, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto rbtfly_impl(tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi);
        auto new_lo    = tupi::group_invoke([](auto p) { return tupi::get<0>(p); }, btfly_res);
        auto new_hi    = tupi::group_invoke([](auto p) { return tupi::get<1>(p); }, btfly_res);
        auto ctw       = conj(tws);
        auto new_hi_tw = tupi::group_invoke(simd::mul, new_hi, tws);
        return combine_halves<stride>(new_lo, new_hi_tw);
    };
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
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZ_constant<0>{}),
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZ_constant<Stride / 2>{}));
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
        }(lo, hi, std::make_index_sequence<NodeSize / Stride>{});
    }

    PCX_AINLINE static auto make_tw_getter(tw_t tw) {
        return [tw]<uZ Size> PCX_LAINLINE(uZ_constant<Size>) {
            return []<uZ... Itw> PCX_LAINLINE(auto tw, std::index_sequence<Itw...>) {
                constexpr auto repeats = NodeSize / Size;
                constexpr auto start   = Size / 2 - 1;
                return tupi::tuple_cat(tupi::make_broadcast_tuple<repeats>(tupi::get<start + Itw>(tw))...);
            }(tw, std::make_index_sequence<Size / 2>{});
            //
        };
    }
    static constexpr struct const_tw {
        /** Indexes are bit-reversed, and thus independent on the actual transform size e.g.
         * if twiddle is defined as tw = exp(-2 * pi * i * k / N)
         * `ITw == 0` => k == 0
         * `ITw == 1` => k == N/4
         * `ITw == 2` => k == N/8
         * `ITw == 3` => k == 3N/8
         * ...
         */
        static inline auto values = []<uZ... Is>(std::index_sequence<Is...>) {
            constexpr auto calc_tw = []<uZ I>(uZ_constant<I>) {
                if constexpr (I < 2) {
                    return imag_unit<I>;
                } else {
                    constexpr auto N = next_pow_2(I + 1) * 2;
                    constexpr auto K = (I - N / 4) * 2 + 1;
                    return wnk<T>(N, K);
                }
            };
            return tupi::make_tuple(calc_tw(uZ_constant<Is>{})...);
        }(std::make_index_sequence<NodeSize / 2>{});

        template<uZ I>
        PCX_AINLINE constexpr static auto get_tw_value() {
            if constexpr (I < 2) {
                return imag_unit<I>;
            } else {
                return simd::cxbroadcast<1, Width>(&tupi::get<I>(values));
            }
        }

        template<uZ Size>
        PCX_AINLINE auto operator()(uZ_constant<Size>) const {
            return []<uZ... Is>(std::index_sequence<Is...>) {
                constexpr auto repeats = NodeSize / Size;

                return tupi::tuple_cat(                     //
                    tupi::make_broadcast_tuple<repeats>(    //
                        get_tw_value<reverse_bit_order(Is, Size / 2)>())...);
            }(std::make_index_sequence<Size / 2>{});
        };
    } const_tw_getter;
};

}    // namespace pcx::detail_
