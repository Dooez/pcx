#pragma once
#include "pcx/include/meta.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"

#include <cassert>

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

constexpr auto powi(u64 num, u64 pow) -> u64 {    // NOLINT(*recursion*)
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
    while ((k > 0) && (k % 2 == 0)) {
        k /= 2;
        n /= 2;
    }
    if (k == 0)
        return {1, 0};
    if (n == k * 2)
        return {-1, 0};
    if (n == k * 4)
        return {0, -1};
    if (k > n / 4) {
        auto v = wnk<T>(n, k - n / 4);
        return {v.imag(), -v.real()};
    }
    constexpr auto pi = std::numbers::pi;
    return static_cast<std::complex<T>>(
        std::exp(std::complex<f64>(0, -2 * pi * static_cast<f64>(k) / static_cast<f64>(n))));
}
/**
 * @brief Returnes twiddle factor with k bit-reversed
 */
template<typename T = f64>
inline auto wnk_br(uZ n, uZ k) -> std::complex<T> {
    k = reverse_bit_order(k, log2i(n) - 1);
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
    requires(NodeSize >= 2)
struct btfly_node_dit {
    using cx_vec = simd::cx_vec<T, false, false, Width>;

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool reverse;
    };

    using dest_t = tupi::broadcast_tuple_t<T*, NodeSize>;
    using src_t  = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using ctw_t  = tupi::broadcast_tuple_t<cx_vec, NodeSize / 2>;

    template<settings S>
    PCX_AINLINE static void perform(cevalue<S>, const dest_t& dest, const src_t& src, const ctw_t& tw) {
        if constexpr (S.reverse) {
            rev_impl<S.pack_dest, S.pack_src>(dest, src, make_tw_getter(tw));
        } else {
            fwd_impl<S.pack_dest, S.pack_src>(dest, src, make_tw_getter(tw));
        }
    }
    template<settings S>
    PCX_AINLINE static void perform(const dest_t& dest, const src_t& src, const ctw_t& tw) {
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
    PCX_AINLINE static void perform(const dest_t& dest, const ctw_t& tw) {
        perform<S>(dest, dest, tw);
    }
    template<settings S>
    PCX_AINLINE static void perform_lo_k(const dest_t& dest) {
        perform_lo_k<S>(dest, dest);
    }

    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void fwd_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data = tupi::group_invoke(simd::cxload<SrcPackSize, Width> | simd::repack<Width>, src);
        auto res  = []<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == NodeSize) {
                    return btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size * 2>{});
                }
            }(data, get_tw);
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void rev_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data = tupi::group_invoke(simd::cxload<SrcPackSize> | simd::repack<Width>, src);
        auto res  = []<uZ Size = NodeSize> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == 2) {
                    return rbtfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = rbtfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size / 2>{});
                }
            }(data, get_tw);
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
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

    PCX_AINLINE static auto make_tw_getter(ctw_t tw) {
        return [tw]<uZ Size> PCX_LAINLINE(uZc<Size>) {
            return [&]<uZ... Itw> PCX_LAINLINE(std::index_sequence<Itw...>) {
                static_assert(Size <= NodeSize);
                constexpr auto repeats = NodeSize / Size;
                if constexpr (Size == 2) {
                    return tupi::make_broadcast_tuple<repeats>(tupi::get<0>(tw));
                } else {
                    constexpr auto start = Size / 4;
                    return tupi::tuple_cat(tupi::tuple_cat(
                        tupi::make_broadcast_tuple<repeats>(tupi::get<start + Itw>(tw)),
                        tupi::make_broadcast_tuple<repeats>(mul_by_j<-1>(tupi::get<start + Itw>(tw))))...);
                }
            }(std::make_index_sequence<Size / 4>{});
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
                if constexpr (I == 0) {
                    return imag_unit<0>;
                } else if constexpr (I == 1) {
                    return imag_unit<-1>;
                } else {
                    constexpr auto N = next_pow_2(I + 1) * 2;
                    return wnk_br<T>(N, I);
                }
            };
            return tupi::make_tuple(calc_tw(uZc<Is>{})...);
        }(std::make_index_sequence<NodeSize / 2>{});
        template<uZ I>
        PCX_AINLINE constexpr static auto get_tw_value() {
            if constexpr (I == 0) {
                return imag_unit<0>;
            } else if constexpr (I == 1) {
                return imag_unit<-1>;
            } else {
                return simd::cxbroadcast<1, Width>(reinterpret_cast<const T*>(&tupi::get<I>(values)));
            }
        }
        template<uZ Size>
        PCX_AINLINE auto operator()(uZc<Size>) const {
            return []<uZ... Is>(std::index_sequence<Is...>) {
                constexpr auto repeats = NodeSize / Size;
                return tupi::tuple_cat(                     //
                    tupi::make_broadcast_tuple<repeats>(    //
                        get_tw_value<Is>())...);
            }(std::make_index_sequence<Size / 2>{});
        };
    } const_tw_getter{};
};

template<uZ NodeSize, typename T, uZ Width>
struct subtransform {
    using btfly_node = btfly_node_dit<NodeSize, T, Width>;
    using vec_traits = simd::detail_::vec_traits<T, Width>;


    struct align_node_t {
        uZ node_size_pre  = 1;
        uZ node_size_post = 1;
    };

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK>
    static void perform(uZ data_size, T* dest_ptr, const T* tw_ptr) {
        constexpr auto single_load_size = NodeSize * Width;

        auto lsize          = data_size / single_load_size;
        auto slog           = log2i(lsize);
        auto a              = slog / log2i(NodeSize);
        auto b              = a * log2i(NodeSize);
        auto pre_align_node = powi(2, slog - b);

        [&]<uZ... Is>(std::index_sequence<Is...>) {
            auto check_align = [&]<uZ I>(uZc<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != pre_align_node)
                    return false;
                perform_impl<DestPackSize, SrcPackSize, LowK, {l_node_size, 1}>(data_size, dest_ptr, tw_ptr);
                return true;
            };
            (void)(check_align(uZc<Is>{}) || ...);
        }(std::make_index_sequence<log2i(NodeSize)>{});
    }

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK, align_node_t AlignNode>
    static void perform_impl(uZ data_size, T* dest_ptr, const T* tw_ptr) {
        constexpr auto single_load_size = NodeSize * Width;

        uZ size = 1;
        if constexpr (AlignNode.node_size_pre != 1) {
            constexpr auto align_node = AlignNode.node_size_pre;
            fft_iteration<align_node, Width, SrcPackSize, LowK>(data_size, size, dest_ptr, tw_ptr);
            if constexpr (LowK)
                tw_ptr += size;
        } else {
            fft_iteration<NodeSize, Width, SrcPackSize, LowK>(data_size, size, dest_ptr, tw_ptr);
        }

        while (data_size / (size * NodeSize) >= single_load_size)
            fft_iteration<NodeSize, Width, Width, LowK>(data_size, size, dest_ptr, tw_ptr);
        if constexpr (LowK) {
            if (size > AlignNode.node_size_pre)
                tw_ptr += size;
        }

        if constexpr (AlignNode.node_size_post != 1) {
            constexpr auto align_node = AlignNode.node_size_post;
            fft_iteration<align_node, Width, Width, LowK>(data_size, size, dest_ptr, tw_ptr);
            if constexpr (LowK)
                tw_ptr += size;
        }

        constexpr auto skip_single_load = false;
        if constexpr (skip_single_load) {
            for (auto i: stdv::iota(0U, data_size / Width)) {
                auto ptr = dest_ptr + i * Width * 2;
                auto rd  = (simd::cxload<Width, Width> | simd::repack<1>)(ptr);
                simd::cxstore<1>(ptr, rd);
            }
            return;
        }

        if constexpr (LowK) {
            single_load<DestPackSize, Width, LowK>(dest_ptr, dest_ptr, tw_ptr);
        }
        constexpr auto start = LowK ? 1UZ : 0UZ;
        for (auto i: stdv::iota(start, data_size / single_load_size)) {
            auto dest = dest_ptr + i * single_load_size * 2;
            single_load<DestPackSize, Width, false>(dest, dest, tw_ptr);
        }
    };

    // private:
    static constexpr auto half_node_idxs = std::make_index_sequence<NodeSize / 2>{};

    template<uZ NodeSizeL, uZ PackDest, uZ PackSrc, bool LowK>
    PCX_AINLINE static auto fft_iteration(uZ data_size, uZ& fft_size, auto data_ptr, auto& tw_ptr) {
        using btfly_node  = btfly_node_dit<NodeSizeL, T, Width>;
        auto  tw_ptr_copy = tw_ptr;
        auto& l_tw_ptr    = LowK ? tw_ptr_copy : tw_ptr;

        constexpr auto settings = typename btfly_node::settings{
            .pack_dest = PackDest,
            .pack_src  = PackSrc,
            .reverse   = false,
        };
        auto group_size    = data_size / fft_size / 2;
        auto make_data_tup = [=] PCX_LAINLINE(uZ k, uZ offset) {
            auto node_stride = group_size / NodeSizeL * 2 /*second group half*/ * 2 /*complex*/;
            auto k_stride    = group_size * 2 /*second group half*/ * 2 /*complex*/;
            return [node_stride]<uZ... Is> PCX_LAINLINE(auto data_ptr,    //
                                                        std::index_sequence<Is...>) {
                return tupi::make_tuple((data_ptr + node_stride * Is)...);
            }(data_ptr + k * k_stride + offset * 2, std::make_index_sequence<NodeSizeL>{});
        };

        constexpr auto n_tw = NodeSizeL / 2;
        if constexpr (LowK) {
            for (auto i: stdv::iota(0U, group_size / NodeSizeL * 2) | stdv::stride(Width)) {
                auto data = make_data_tup(0, i);
                btfly_node::template perform_lo_k<settings>(data);
            }
            l_tw_ptr += n_tw * 2;
        }
        constexpr auto start = LowK ? 1UZ : 0UZ;
        for (auto k: stdv::iota(start, fft_size)) {
            auto tw = [=]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(l_tw_ptr + Is * 2)...);
            }(std::make_index_sequence<n_tw>{});
            l_tw_ptr += n_tw * 2;
            for (auto i: stdv::iota(0U, group_size / NodeSizeL * 2) | stdv::stride(Width)) {
                auto data = make_data_tup(k, i);
                btfly_node::template perform<settings>(data, tw);
            }
        }
        fft_size *= NodeSizeL;
    }
    template<uZ DestPackSize, uZ SrcPackSize, bool LowK>
    PCX_AINLINE static auto single_load(T* data_ptr, const T* src_ptr, auto& tw_ptr) {
        auto data = []<uZ... Is>(auto data_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxload<SrcPackSize, Width>(data_ptr + Width * 2 * Is)...);
        }(src_ptr, std::make_index_sequence<NodeSize>{});
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);

        constexpr auto btfly0 =
            []<uZ Size = 2> PCX_LAINLINE(this auto f, auto data, auto get_tw, uZc<Size> = {}) {
                if constexpr (Size == NodeSize) {
                    return btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                } else {
                    auto tmp = btfly_node::template btfly_impl<Size>(data, get_tw(uZc<Size>{}));
                    return f(tmp, get_tw, uZc<Size * 2>{});
                }
            };
        auto btfly_res_0 = [&]() {
            if constexpr (LowK) {
                tw_ptr += NodeSize;
                return btfly0(data_rep, btfly_node::const_tw_getter);
            } else {
                auto tw0 = []<uZ... Is>(auto tw_ptr, std::index_sequence<Is...>) {
                    return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + 2 * Is)...);
                }(tw_ptr, std::make_index_sequence<NodeSize / 2>{});
                tw_ptr += NodeSize;
                return btfly0(data_rep, btfly_node::make_tw_getter(tw0));
            }
        }();

        auto [data_lo, data_hi] = [&]<uZ... Is>(std::index_sequence<Is...>) {
            auto lo = tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
            auto hi = tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
            return tupi::make_tuple(lo, hi);
        }(half_node_idxs);
        auto [lo, hi] = [&tw_ptr]<uZ NGroups = 2> PCX_LAINLINE    //
            (this auto f, auto data_lo, auto data_hi, uZc<NGroups> = {}) {
                if constexpr (NGroups == Width) {
                    auto tmp = regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
                    tw_ptr += NGroups * 2 * NodeSize / 2;
                    return tmp;
                } else {
                    auto [lo, hi] = regroup_btfly<NGroups>(data_lo, data_hi, tw_ptr);
                    tw_ptr += NGroups * 2 * NodeSize / 2;
                    return f(lo, hi, uZc<NGroups * 2>{});
                }
            }(data_lo, data_hi);
        auto btfly_res_1 = tupi::group_invoke(regroup<1, Width>, lo, hi);
        auto res         = tupi::make_flat_tuple(btfly_res_1);
        auto res_rep     = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        [data_ptr, res_rep]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            (simd::cxstore<DestPackSize>(data_ptr + Width * 2 * Is, get<Is>(res_rep)), ...);
        }(std::make_index_sequence<NodeSize>{});
    }

    template<uZ NGroups>
    struct regroup_btfly_t {
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...> lo, tupi::tuple<Thi...> hi, const T* tw_ptr) {
            auto tw_tup = tupi::make_broadcast_tuple<NodeSize / 2>(tw_ptr);

            constexpr auto regr_ltw =
                tupi::make_tuple
                | tupi::pipeline(tupi::apply | tupi::group_invoke(split_regroup<Width / NGroups>),
                                 tupi::apply | tupi::group_invoke(load_tw<NGroups>));
            auto [regrouped, tw] = regr_ltw(tupi::forward_as_tuple(lo, hi),    //
                                            tupi::forward_as_tuple(tw_tup, half_node_tuple));

            auto lo_re  = tupi::group_invoke(tupi::get<0>, regrouped);
            auto hi_re  = tupi::group_invoke(tupi::get<1>, regrouped);
            auto hi_tw  = tupi::group_invoke(simd::mul, hi_re, tw);
            auto res    = tupi::group_invoke(simd::btfly, lo_re, hi_tw);
            auto new_lo = tupi::group_invoke(tupi::get_copy<0>, res);
            auto new_hi = tupi::group_invoke(tupi::get_copy<1>, res);
            return tupi::make_tuple(new_lo, new_hi);
        }

    private:
        static constexpr auto half_node_tuple = []<uZ... Is>(std::index_sequence<Is...>) {
            return tupi::make_tuple(uZc<Is>{}...);
        }(half_node_idxs);
    };
    /**
     *  @brief Split-regroups input data, loads twiddles and performs a single butterfly operation.
     *  see `split_regroup<>`. 
     *  
     *  @tparam NGroups - number of fft groups (`k`) that fit in a single simd vector.
     */
    template<uZ NGroups>
    constexpr static auto regroup_btfly = regroup_btfly_t<NGroups>{};

    // clang-format off
    /**
     * @brief Loads and upsamples `Count` twiddles.
     */
    template<uZ Count>
    static constexpr auto load_tw =
        tupi::pass    
        | []<uZ IGroup>(const T* tw_ptr, uZc<IGroup>) {
            // return simd::cxload<Count, Count>(tw_ptr + Count * (2 * IGroup));
            auto tw = simd::cxload<1, Count>(tw_ptr + Count * (2 * IGroup));
            auto twr = simd::repack<Count>(tw);
            return twr;
          }
        | tupi::group_invoke([](auto v){
                return vec_traits::upsample(v.value);
          })    
        | tupi::apply                                                  
        | [](auto re, auto im) { return simd::cx_vec<T, false, false, Width>{re, im}; };


    /**
     * @brief Regroups input sismd vectors, similar to `simd::repack`, except
     * that the real and imaginary part are processed separately.
     */
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
    /**
     *  @brief Splits the input simd vectors into even/odd chunks of `ChunkSize`,
     *  interleaves the matching chunks of `a` and `b`.
     *
     *  @return `tupi::tuple<>` the interleaved even/odd chunks.
     *
     *  Example: 
     *  Width     == 8
     *  ChunkSize == 2
     *  a = [a0 a1 a2 a3 a4 a5 a6 a7]  
     *  b = [b0 b1 b2 b3 b4 b5 b6 b7]  
     * 
     *  result<0> = [a0 a1 b0 b1 a4 a5 b4 b5]
     *  result<1> = [a2 a3 b2 b3 a6 a7 b6 b7]
     */
    template<uZ ChunkSize>
    static constexpr auto split_regroup = 
        tupi::pass 
        | []<simd::eval_cx_vec V>(V a, V b) 
            requires(ChunkSize <= V::width())
          {
            auto re = tupi::make_tuple(a.real_v(), b.real_v());
            auto im = tupi::make_tuple(a.imag_v(), b.imag_v());
            return tupi::make_tuple(re, im, meta::types<V>{});
          }
        | tupi::pipeline(tupi::apply | vec_traits::template split_interleave<ChunkSize>, 
                         tupi::apply | vec_traits::template split_interleave<ChunkSize>,
                         tupi::pass)
        | tupi::apply
        | []<typename V>(auto re, auto im, meta::types<V>){
            return tupi::make_tuple(V{.m_real = get<0>(re), .m_imag = get<0>(im)},    
                                    V{.m_real = get<1>(re), .m_imag = get<1>(im)});
          };
    // clang-format on
};

template<uZ NodeSize, typename T, uZ Width>
struct transform {
    using subtf = subtransform<NodeSize, T, Width>;


    static constexpr uZ coherent_size = 2048;
    static constexpr uZ line_size     = 512 / Width / sizeof(T);

    template<uZ DestPackSize, uZ SrcPackSize>
    static void perform(uZ data_size, T* dest_ptr, const T* tw_ptr) {
        uZ max_subdivisions = coherent_size / Width;

        constexpr auto logKi = [](uZ k, uZ value) {
            auto pow = 1;
            while (k > value) {
                pow++;
                k /= value;
            };
            return pow;
        };
        uZ subdivision_depth = logKi(data_size / coherent_size, max_subdivisions);

        uZ current_depth  = 1;
        uZ n_subdivisions = std::min(max_subdivisions, data_size / coherent_size);

        uZ stride = 0;

        // constant stride
        //  [s0, s1,    ..., s0, ...    ]
        //  [0 , width, ..., stride, ...]
        //
        // uZ stride = Width * n_subdivisions;

        // bundled
        // [ b0[...], b1[...]    ,..., b0[...], b1[...], ... ]
        // [ 0,...  , bundle_size,..., stride , ...          ]
        //
        // uZ stride = data_size / n_subdivisions;
        // uZ bundle_size = coherent_size / n_subdivisions;
        //


        // uZ   depth    = 1;
        // auto l_stride = subsize;
        // while (true) {
        //     if (data_size <= l_stride * max_subdivisions)
        //         break;
        //     l_stride = l_stride * max_subdivisions;
        //     ++depth;
        // }

        //either [grp_size]
    };


    template<uZ DestPackSize, uZ SrcPackSize, uZ AlignSize = 1>
    void bundled_sparse_subtform(uZ bundles_size, uZ stride, uZ fft_size, T* dest_ptr, const T* tw_ptr) {
        constexpr auto single_load_size = NodeSize * Width;

        uZ size = 1;
        if constexpr (AlignSize != 1) {
            // fft_iteration<align_node, Width, SrcPackSize, LowK>(data_size, size, dest_ptr, tw_ptr);
        } else {
            // fft_iteration<NodeSize, Width, SrcPackSize, LowK>(data_size, size, dest_ptr, tw_ptr);
        }
        // while (data_size / (size * NodeSize) >= single_load_size)
        // fft_iteration<NodeSize, Width, Width, LowK>(data_size, size, dest_ptr, tw_ptr);
    };
    template<uZ NodeSizeL, uZ PackDest, uZ PackSrc>
    PCX_AINLINE static auto fft_iteration_bun(uZ    bundle_size,
                                              uZ    stride,
                                              uZ    bundle_stride,
                                              uZ    n_bundles_per_group,
                                              uZ&   fft_size,
                                              auto  data_ptr,
                                              auto& tw_ptr) {
        auto& l_tw_ptr = tw_ptr;

        auto make_data_tup = [=] PCX_LAINLINE(uZ i, uZ i_bun, uZ k) {
            auto base_ptr = data_ptr                       //
                            + i * Width * 2                //
                            + i_bun * bundle_stride * 2    //
                            + k * NodeSizeL * stride * 2;
            return [stride]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tupi::make_tuple((base_ptr + stride * Is)...);
            }(std::make_index_sequence<NodeSizeL>{});
        };

        constexpr auto n_tw = NodeSizeL / 2;
        for (auto k_group: stdv::iota(0U, fft_size)) {
            auto tw = [=]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(l_tw_ptr + Is * 2)...);
            }(std::make_index_sequence<n_tw>{});
            l_tw_ptr += n_tw * 2;

            for (auto i_bundle: stdv::iota(0U, n_bundles_per_group)) {
                for (auto i: stdv::iota(0U, bundle_size / Width)) {
                    auto data = make_data_tup(i, i_bundle, k_group);
                    //btfly::perfom(data, tw);
                }
            }
        }
    }

    template<uZ NodeSizeL, uZ PackDest, uZ PackSrc, bool LowK>
    PCX_AINLINE static auto fft_iteration_cs(uZ    stride,    //
                                             uZ    group_stride,
                                             uZ    group_size,
                                             uZ&   fft_size,
                                             auto  data_ptr,
                                             auto& tw_ptr) {
        using btfly_node        = btfly_node_dit<NodeSizeL, T, Width>;
        constexpr auto settings = cevalue<typename btfly_node::settings{
            .pack_dest = PackDest,
            .pack_src  = PackSrc,
            .reverse   = false,
        }>{};

        auto& l_tw_ptr = tw_ptr;

        auto make_data_tup = [=] PCX_LAINLINE(uZ i, uZ k) {
            auto base_ptr = data_ptr            //
                            + i * stride * 2    //
                            + k * NodeSizeL * group_stride * 2;
            return [group_stride]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tupi::make_tuple((base_ptr + group_stride * Is)...);
            }(std::make_index_sequence<NodeSizeL>{});
        };

        constexpr auto n_tw = NodeSizeL / 2;
        for (auto k_group: stdv::iota(0U, fft_size)) {
            auto tw = [=]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(l_tw_ptr + Is * 2)...);
            }(std::make_index_sequence<n_tw>{});
            l_tw_ptr += n_tw * 2;

            for (auto i: stdv::iota(0U, group_size / Width)) {
                auto data = make_data_tup(i, k_group);
                btfly_node::perform(settings, data);
            }
        }
    }
};

}    // namespace pcx::detail_
