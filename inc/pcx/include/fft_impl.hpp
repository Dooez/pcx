#pragma once
#include "pcx/include/meta.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"

namespace pcx::simd {

constexpr struct btfly_t {
    PCX_AINLINE static auto operator()(any_cx_vec auto a, any_cx_vec auto b) {
        return std::make_tuple(add(a, b), sub(a, b));
    }
} btfly{};
}    // namespace pcx::simd

namespace pcx::detail_ {

constexpr auto log2i(u64 num) -> uZ {
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
template<typename T = f64>
inline auto wnk(uZ n, uZ k) -> std::complex<T> {
    while (k > 0 && k % 2 == 0) {
        k /= 2;
        n /= 2;
    }
    constexpr auto pi = std::numbers::pi;
    if (n == k * 2)
        return {-1, 0};
    if (n == k * 4)
        return {0, -1};
    if (k > n / 4)
        return static_cast<std::complex<T>>(std::complex<f64>(0, -1) * wnk(n, k - n / 4));
    return static_cast<std::complex<T>>(
        std::exp(std::complex<f64>(0, -2 * pi * static_cast<f64>(k) / static_cast<f64>(n))));
}
/**
 * @brief Returnes twiddle factor with k bit-reversed
 */
template<typename T = f64>
inline auto wnk_br(uZ n, uZ k) -> std::complex<T> {
    while (k % 2 == 0) {
        k /= 2;
        n /= 2;
    }
    k = reverse_bit_order(k, log2i(n));
    return wnk<T>(n, k);
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

    template<settings S>
    PCX_AINLINE static void perform(const dest_t& dest, const src_t& src, const tw_t& tw) {
        if constexpr (S.reverse) {
            rev_impl<S.pack_dest, S.pack_src>(dest, src, make_tw_getter(tw));
        } else {
            fwd_impl<S.pack_dest, S.pack_src>(dest, src, make_tw_getter(tw));
        }
    }
    template<settings S>
    PCX_AINLINE static void perform_lo_k(const dest_t& dest, const src_t& src) {
        if constexpr (S.reverse) {
            rev_impl<S.pack_dest, S.pack_src>(dest, src, const_tw_getter);
        } else {
            fwd_impl<S.pack_dest, S.pack_src>(dest, src, const_tw_getter);
        }
    }
    template<settings S>
    PCX_AINLINE static void perform(const dest_t& dest, const tw_t& tw) {
        perform<S>(dest, dest, tw);
    }
    template<settings S>
    PCX_AINLINE static void perform_lo_k(const dest_t& dest) {
        preform_lo_k<S>(dest, dest);
    }

    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void fwd_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize, Width>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == NodeSize) {
                    return btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size * 2>{});
                }
            }(data_rep, get_tw);
        auto res_eval = tupi::group_invoke(simd::evaluate, res);
        auto res_rep  = tupi::group_invoke(simd::repack<DestPackSize>, res_eval);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void rev_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = NodeSize> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == 2) {
                    return rbtfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = rbtfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size / 2>{});
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
        auto new_lo    = tupi::group_invoke(tupi::get<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get<1>, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto rbtfly_impl(tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi);
        auto new_lo    = tupi::group_invoke(tupi::get<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get<1>, btfly_res);
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
                                                                       uZc<Start>) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset> PCX_LAINLINE(auto data,
                                                                             std::index_sequence<Iters...>,
                                                                             uZc<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(data)...);
            };
            return tupi::tuple_cat(
                iterate(data, std::make_index_sequence<Stride / 2>{}, uZc<Start + Grp * Stride>{})...);
        };
        return tupi::make_tuple(
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZc<0>{}),
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZc<Stride / 2>{}));
    }
    /**
     * @brief Combines two halves into a tuple
     *
     * lo     = [0,          1,              ..., Stride / 2 - 1, Stride        , Stride + 1,         ... ]
     * hi     = [Stride / 2, Stride / 2 + 1, ..., Stride - 1    , Stride * 3 / 2, Stride * 3 / 2 + 1, ... ]
     * return = [0, 1, ..., N - 1] 
     */
    template<uZ Stride, typename... Tsl, typename... Tsh>
        requires(simd::any_cx_vec<std::remove_cvref_t<Tsl>> && ...)
                && (simd::any_cx_vec<std::remove_cvref_t<Tsh>> && ...)
    PCX_AINLINE static auto combine_halves(tupi::tuple<Tsl...> lo, tupi::tuple<Tsh...> hi) {
        return []<uZ... Grp> PCX_LAINLINE(auto lo, auto hi, std::index_sequence<Grp...>) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset> PCX_LAINLINE(auto lo,
                                                                             auto hi,
                                                                             std::index_sequence<Iters...>,
                                                                             uZc<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(lo)..., tupi::get<Offset + Iters>(hi)...);
            };
            return tupi::tuple_cat(iterate(lo,    //
                                           hi,
                                           std::make_index_sequence<Stride / 2>{},
                                           uZc<Grp * Stride / 2>{})...);
        }(lo, hi, std::make_index_sequence<NodeSize / Stride>{});
    }

    PCX_AINLINE static auto make_tw_getter(tw_t tw) {
        return [tw]<uZ Size> PCX_LAINLINE(uZc<Size>) {
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
            constexpr auto calc_tw = []<uZ I>(uZc<I>) {
                if constexpr (I < 2) {
                    return imag_unit<I>;
                } else {
                    constexpr auto N = next_pow_2(I + 1) * 2;
                    constexpr auto K = (I - N / 4) * 2 + 1;
                    return wnk<T>(N, K);
                }
            };
            return tupi::make_tuple(calc_tw(uZc<Is>{})...);
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
        PCX_AINLINE auto operator()(uZc<Size>) const {
            return []<uZ... Is>(std::index_sequence<Is...>) {
                constexpr auto repeats = NodeSize / Size;

                return tupi::tuple_cat(                     //
                    tupi::make_broadcast_tuple<repeats>(    //
                        get_tw_value<reverse_bit_order(Is, Size / 2)>())...);
            }(std::make_index_sequence<Size / 2>{});
        };
    } const_tw_getter{};
};

template<uZ NodeSize, typename T, uZ Width>
struct subtransform {
    using btfly_node = btfly_node_dit<NodeSize, T, Width>;
    using vec_traits = simd::detail_::vec_traits<T, Width>;


    template<uZ DestPackSize, uZ SrcPackSize>
    void perform_lo_k(uZ size, uZ max_size, T* dest_ptr, const T* tw_ptr) {
        static constexpr auto single_load_size = NodeSize * Width;

        while (max_size / size >= single_load_size * NodeSize) {
            iterate_lo_k<NodeSize>(max_size, size, dest_ptr, tw_ptr);
        }

        [&]<uZ... Is>(std::index_sequence<Is...>) {
            ((((max_size / size) == (single_load_size * NodeSize / (2 * Is)))
              && (iterate_lo_k<NodeSize / (2 * Is)>(max_size, size, dest_ptr, tw_ptr), true))
             || ...);
        }(std::make_index_sequence<log2i(NodeSize)>{});

        if (max_size / size == single_load_size) {
            single_load_lo_k<DestPackSize, DestPackSize>(dest_ptr, dest_ptr, tw_ptr);
            //do single load subtransform
            return;
        }
    };
    void perform(uZ size, uZ max_size, T* dest_ptr, const T* tw_ptr) {
        static constexpr auto single_load_size = NodeSize * Width;

        while (max_size / size >= single_load_size * NodeSize) {
            iterate<NodeSize>(max_size, size, dest_ptr, tw_ptr);
        }

        [&]<uZ... Is>(std::index_sequence<Is...>) {
            ((((max_size / size) >= (single_load_size * NodeSize / (2 * Is)))
              && (iterate<NodeSize / (2 * Is)>(max_size, size, dest_ptr, tw_ptr), true))
             || ...);
        }(std::make_index_sequence<log2i(NodeSize)>{});

        if (max_size / size == single_load_size) {
            //do single load subtransform
            return;
        }
    };

    // private:
    static constexpr auto half_node_idxs = std::make_index_sequence<NodeSize / 2>{};

    template<uZ NodeSizeL>
    PCX_AINLINE static auto iterate_lo_k(uZ max_size, uZ& size, auto data_ptr, auto tw_ptr) {
        static constexpr auto single_load_size = NodeSizeL * Width;

        using btfly_node = btfly_node_dit<NodeSizeL, T, Width>;

        constexpr auto settings = typename btfly_node::settings{
            .pack_dest = simd::max_width<T>,
            .pack_src  = simd::max_width<T>,
            .reverse   = false,
        };

        auto data_stride = max_size / size;
        for (auto i: stdv::iota(0U, data_stride / single_load_size)) {
            auto data =
                []<uZ... Is> PCX_LAINLINE(auto data_ptr, auto stride, std::index_sequence<Is...>) {
                    return tupi::make_tuple((data_ptr + stride * Is)...);
                }(data_ptr + i * data_stride * 2 * single_load_size,
                  data_stride,
                  std::make_index_sequence<NodeSizeL>{});
            btfly_node::template perform_lo_k<settings>(data);
        }
        for (auto k: stdv::iota(1U, size / 2)) {
            auto tw = []<uZ... Is> PCX_LAINLINE(auto tw_ptr, std::index_sequence<Is...>) {
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + Is * 2)...);
            }(tw_ptr, std::make_index_sequence<NodeSizeL - 1>{});
            tw_ptr += 2 * (NodeSizeL - 1);
            for (auto i: stdv::iota(0U, data_stride / single_load_size)) {
                auto data =
                    []<uZ... Is> PCX_LAINLINE(auto data_ptr, auto stride, std::index_sequence<Is...>) {
                        return tupi::make_tuple((data_ptr + stride * Is)...);
                    }(data_ptr + i * data_stride * 2 * single_load_size,
                      data_stride,
                      std::make_index_sequence<NodeSizeL>{});
                btfly_node::template perform<settings>(data, tw);
            }
        }
        size *= NodeSizeL;
    }
    template<uZ NodeSizeL>
    PCX_AINLINE static auto iterate(uZ max_size, uZ& size, auto data_ptr, auto& tw_ptr) {
        static constexpr auto single_load_size = NodeSizeL * Width;

        using btfly_node = btfly_node_dit<NodeSizeL, T, Width>;

        constexpr auto settings = typename btfly_node::settings{
            .pack_dest = simd::max_width<T>,
            .pack_src  = simd::max_width<T>,
            .reverse   = false,
        };

        auto data_stride = max_size / size;
        for (auto k: stdv::iota(1U, size / 2)) {
            auto tw = []<uZ... Is> PCX_LAINLINE(auto tw_ptr, std::index_sequence<Is...>) {
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + Is * 2)...);
            }(tw_ptr, std::make_index_sequence<NodeSizeL - 1>{});
            tw_ptr += 2 * (NodeSizeL - 1);
            for (auto i: stdv::iota(0U, data_stride / single_load_size)) {
                auto data =
                    []<uZ... Is> PCX_LAINLINE(auto data_ptr, auto stride, std::index_sequence<Is...>) {
                        return tupi::make_tuple((data_ptr + stride * Is)...);
                    }(data_ptr + i * data_stride * 2 * single_load_size,
                      data_stride,
                      std::make_index_sequence<NodeSizeL>{});
                btfly_node::template perform<settings>(data, tw);
            }
        }
        size *= NodeSizeL;
    }

    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static auto single_load_lo_k(T* data_ptr, const T* src_ptr, const T* tw_ptr) {
        auto data = []<uZ... Is>(auto data_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxload<SrcPackSize, Width>(data_ptr + Width * 2 * Is)...);
        }(src_ptr, std::make_index_sequence<NodeSize>{});
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);

        auto btfly_res_0 = []<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == NodeSize) {
                    return btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size * 2>{});
                }
            }(data_rep, btfly_node::const_tw_getter);

        auto data_lo = [btfly_res_0]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            return tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
        }(half_node_idxs);
        auto data_hi = [btfly_res_0]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            return tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
        }(half_node_idxs);
        auto [lo, hi] = [tw_ptr]<uZ NGroups = 2> PCX_LAINLINE(this auto f,
                                                              auto      data_lo,
                                                              auto      data_hi,
                                                              uZc<NGroups> = {}) {
            if constexpr (NGroups == NodeSize) {
                return regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
            } else {
                auto [lo, hi] = regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
                return f(lo, hi, uZc<NGroups * 2>{});
            }
        }(data_lo, data_hi);
        auto btfly_res_1 = tupi::group_invoke(regroup<1, Width>, lo, hi);
        auto res         = tupi::make_flat_tuple(btfly_res_1);
        auto res_eval    = tupi::group_invoke(simd::evaluate, res);
        auto res_rep     = tupi::group_invoke(simd::repack<DestPackSize>, res_eval);
        []<uZ... Is> PCX_LAINLINE(auto data_ptr, auto data, std::index_sequence<Is...>) {
            (simd::cxstore<DestPackSize>(data_ptr + Width * 2 * Is, get<Is>(data)), ...);
        }(data_ptr, res_rep, std::make_index_sequence<NodeSize>{});
    }

    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static auto single_load(T* data_ptr, const T* src_ptr, const T* tw_ptr) {
        auto data = []<uZ... Is>(auto data_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxload<SrcPackSize, Width>(data_ptr + Width * 2 * Is)...);
        }(src_ptr, std::make_index_sequence<NodeSize>{});
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);

        auto tw0 = []<uZ... Is>(auto tw_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + 2 * Is)...);
        }(tw_ptr, std::make_index_sequence<NodeSize - 1>{});
        tw_ptr += 2 * (NodeSize - 1);

        auto btfly_res_0 = []<uZ Size = 2> PCX_LAINLINE(this auto f, auto data, auto get_tw, uZc<Size> = {}) {
            if constexpr (Size == NodeSize) {
                return btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
            } else {
                auto tmp = btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                return f(tmp, get_tw, uZc<Size * 2>{});
            }
        }(data_rep, btfly_node::make_tw_getter(tw0));

        auto data_lo = [&btfly_res_0]<uZ... Is>(std::index_sequence<Is...>) {
            return tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
        }(half_node_idxs);
        auto data_hi = [&btfly_res_0]<uZ... Is>(std::index_sequence<Is...>) {
            return tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
        }(half_node_idxs);
        auto [lo, hi] = [&tw_ptr]<uZ NGroups = 2> PCX_LAINLINE(this auto f,
                                                               auto      data_lo,
                                                               auto      data_hi,
                                                               uZc<NGroups> = {}) {
            // if constexpr (NGroups == NodeSize ) {
            if constexpr (NGroups == 2) {
                return regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
            } else {
                auto [lo, hi] = regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
                tw_ptr += NGroups * 2 * NodeSize / 2;
                return f(lo, hi, uZc<NGroups * 2>{});
            }
        }(data_lo, data_hi);
        auto btfly_res_1 = tupi::group_invoke(regroup<1, Width>, lo, hi);
        auto res         = tupi::make_flat_tuple(btfly_res_1);
        // auto res      = tupi::make_flat_tuple(btfly_res_0);
        auto res_eval = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        // auto res_rep     = tupi::group_invoke(simd::repack<DestPackSize>, res_eval);
        []<uZ... Is>(auto data_ptr, auto data, std::index_sequence<Is...>) {
            (simd::cxstore<DestPackSize>(data_ptr + Width * 2 * Is, get<Is>(data)), ...);
        }(data_ptr, res_eval, std::make_index_sequence<NodeSize>{});
    }

    template<uZ NGroups>
        requires(NGroups > 1)
    struct regroup_btfly_t {
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...> lo, tupi::tuple<Thi...> hi, const T* tw_ptr) {
            auto tw_tup = tupi::make_broadcast_tuple<NodeSize / 2>(tw_ptr);

            auto regrouped = tupi::group_invoke(regroup<16, NodeSize / NGroups>, lo, hi);
            auto tw        = tupi::group_invoke(load_tw<NGroups>, tw_tup, half_node_tuple);
            // constexpr auto regr_ltw =
            //     tupi::make_tuple
            //     | tupi::pipeline(tupi::apply | tupi::group_invoke(regroup<16, NodeSize / NGroups>),
            //                      tupi::apply | tupi::group_invoke(load_tw<NGroups>));
            // auto [regrouped, tw] = regr_ltw(tupi::forward_as_tuple(lo, hi),    //
            //                                 tupi::forward_as_tuple(tw_tup, half_node_tuple));

            auto lo_re = tupi::group_invoke(tupi::get<0>, regrouped);
            auto hi_re = tupi::group_invoke(tupi::get<1>, regrouped);
            auto hi_tw = tupi::group_invoke(simd::mul, hi_re, tw);
            // auto btfly_res = tupi::group_invoke(simd::btfly, lo_re, hi_tw);
            auto btfly_res = tupi::group_invoke(tupi::make_tuple, lo_re, hi_tw);
            // auto btfly_res = tupi::group_invoke(simd::btfly, lo_re, hi_re);
            auto new_lo = tupi::group_invoke(tupi::get_copy<0>, btfly_res);
            auto new_hi = tupi::group_invoke(tupi::get_copy<1>, btfly_res);
            return tupi::make_tuple(new_lo, new_hi);
        }

    private:
        static constexpr auto half_node_tuple = []<uZ... Is>(std::index_sequence<Is...>) {
            return tupi::make_tuple(uZc<Is>{}...);
        }(half_node_idxs);
    };
    template<uZ Count>
    constexpr static auto regroup_btfly = regroup_btfly_t<Count>{};

    // clang-format off
    template<uZ Count>
    static constexpr auto load_tw =
        tupi::pass    
        | []<uZ IGroup>(const T* tw_ptr, uZc<IGroup>) {
            // return simd::cxload<Count, Count>(tw_ptr + Count * (2 * IGroup));
            auto tw = simd::cxload<1, Count>(tw_ptr + Count * (2 * IGroup));
            // auto twr = simd::repack<Count>(tw);
            return tw;
          }
        | tupi::group_invoke([](auto v){
                auto x = vec_traits::upsample(v.value);
                return x;
                })    
        | tupi::apply                                                  
        | [](auto re, auto im) { return simd::cx_vec<T, false, false, Width>{re, im}; };

    template<uZ GroupTo, uZ GroupFrom>
    static constexpr auto regroup = 
        tupi::pass 
        | []<simd::any_cx_vec V>(V a, V b) 
            requires(GroupTo <= V::width()) && (GroupFrom <= V::width()) 
          {
            auto re = tupi::make_tuple(a.real_v(), b.real_v());
            auto im = tupi::make_tuple(a.imag_v(), b.imag_v());
            return tupi::make_tuple(re, im, meta::types<V>{});
          }
        | tupi::pipeline(tupi::apply | vec_traits::template repack<GroupTo, GroupFrom>, 
                         tupi::apply | vec_traits::template repack<GroupTo, GroupFrom>,
                         tupi::pass)
        | tupi::apply
        | []<typename V>(auto re, auto im, meta::types<V>){
            return tupi::make_tuple(V{.m_real = get<0>(re), .m_imag = get<0>(im)},    
                                    V{.m_real = get<1>(re), .m_imag = get<1>(im)});
          };
    // clang-format on
};

}    // namespace pcx::detail_
