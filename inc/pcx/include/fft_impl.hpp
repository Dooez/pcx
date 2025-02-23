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

template<uZ NodeSize, typename T, uZ Width>
    requires(NodeSize >= 2)
struct btfly_node_dit {
    using cx_vec = simd::cx_vec<T, false, false, Width>;

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool reverse;
    };

    using dest_t     = tupi::broadcast_tuple_t<T*, NodeSize>;
    using data_t     = tupi::broadcast_tuple_t<cx_vec, NodeSize>;
    using src_t      = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using tw_t       = tupi::broadcast_tuple_t<cx_vec, NodeSize / 2>;
    using low_k_tw_t = decltype([] {});

    PCX_LAINLINE static auto forward(data_t data, low_k_tw_t = {}) {
        return fwd_impl(data, const_tw_getter);
    }
    PCX_LAINLINE static auto forward(data_t data, const tw_t& tw) {
        return fwd_impl(data, make_tw_getter(tw));
    }
    PCX_LAINLINE static auto reverse(data_t data, low_k_tw_t = {}) {
        return rev_impl(data, const_tw_getter);
    }
    PCX_LAINLINE static auto reverse(data_t data, const tw_t& tw) {
        return rev_impl(data, make_tw_getter(tw));
    }

    template<settings S, typename Tw = low_k_tw_t>
        requires std::same_as<Tw, tw_t> || std::same_as<Tw, low_k_tw_t>
    PCX_AINLINE static void perform(val_ce<S>, const dest_t& dest, const src_t& src, const Tw& tw = {}) {
        auto data = tupi::group_invoke(simd::cxload<S.pack_src, Width> | simd::repack<Width>, src);
        auto res  = S.reverse ? reverse(data, tw) : forward(data, tw);
        // auto res = [=] {
        //     if constexpr (S.reverse) {
        //         return reverse(data, tw);
        //     } else {
        //         return forward(data, tw);
        //     }
        // }();
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<S.pack_dest>, res);
        tupi::group_invoke(simd::cxstore<S.pack_dest>, dest, res_rep);
    }
    template<settings S, typename Tw = low_k_tw_t>
        requires std::same_as<Tw, tw_t> || std::same_as<Tw, low_k_tw_t>
    PCX_AINLINE static void perform(val_ce<S> s, const dest_t& dest, const Tw& tw = {}) {
        perform(s, dest, dest, tw);
    }

    PCX_AINLINE static auto fwd_impl(data_t data, auto get_tw) {
        return []<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_ce<Size> size = {}) {
                if constexpr (size == NodeSize) {
                    return btfly_impl(size, data, get_tw(size));
                } else {
                    auto tmp = btfly_impl(size, data, get_tw(size));
                    return f(tmp, get_tw, uZ_ce<size * 2>{});
                }
            }(data, get_tw);
    }
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void fwd_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data    = tupi::group_invoke(simd::cxload<SrcPackSize, Width> | simd::repack<Width>, src);
        auto res     = fwd_impl(data, get_tw);
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }
    PCX_AINLINE static auto rev_impl(data_t data, auto get_tw) {
        return []<uZ Size = NodeSize> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_ce<Size> size = {}) {
                if constexpr (size == 2) {
                    return rbtfly_impl(size, data, get_tw(size));
                } else {
                    auto tmp = rbtfly_impl(size, data, get_tw(size));
                    return f(tmp, get_tw, uZ_ce<size / 2>{});
                }
            }(data, get_tw);
    }
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void rev_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data    = tupi::group_invoke(simd::cxload<SrcPackSize> | simd::repack<Width>, src);
        auto res     = rev_impl(data, get_tw);
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        tupi::group_invoke(simd::cxstore<DestPackSize>, dest, res_rep);
    }

    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto btfly_impl(uZ_ce<Size>, tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto hi_tw     = tupi::group_invoke(simd::mul, hi, tws);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi_tw);
        auto new_lo    = tupi::group_invoke(tupi::get<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get<1>, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto rbtfly_impl(uZ_ce<Size>, tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi);
        auto new_lo    = tupi::group_invoke(tupi::get<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get<1>, btfly_res);
        auto ctw       = tupi::group_invoke([](auto tw) { return conj(tw); }, tws);
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
                                                                       uZ_ce<Start>) {
            constexpr auto iterate = []<uZ... Iters, uZ Offset> PCX_LAINLINE(auto data,
                                                                             std::index_sequence<Iters...>,
                                                                             uZ_ce<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(data)...);
            };
            return tupi::tuple_cat(
                iterate(data, std::make_index_sequence<Stride / 2>{}, uZ_ce<Start + Grp * Stride>{})...);
        };
        return tupi::make_tuple(
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZ_ce<0>{}),
            get_half(data, std::make_index_sequence<NodeSize / Stride>{}, uZ_ce<Stride / 2>{}));
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
                                                                             uZ_ce<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(lo)..., tupi::get<Offset + Iters>(hi)...);
            };
            return tupi::tuple_cat(iterate(lo,    //
                                           hi,
                                           std::make_index_sequence<Stride / 2>{},
                                           uZ_ce<Grp * Stride / 2>{})...);
        }(lo, hi, std::make_index_sequence<NodeSize / Stride>{});
    }

    PCX_AINLINE static auto make_tw_getter(tw_t tw) {
        return [tw]<uZ Size> PCX_LAINLINE(uZ_ce<Size>) {
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
            constexpr auto calc_tw = []<uZ I>(uZ_ce<I>) {
                if constexpr (I == 0) {
                    return imag_unit<0>;
                } else if constexpr (I == 1) {
                    return imag_unit<-1>;
                } else {
                    constexpr auto N = next_pow_2(I + 1) * 2;
                    return wnk_br<T>(N, I);
                }
            };
            return tupi::make_tuple(calc_tw(uZ_ce<Is>{})...);
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
        PCX_AINLINE auto operator()(uZ_ce<Size>) const {
            return []<uZ... Is>(std::index_sequence<Is...>) {
                constexpr auto repeats = NodeSize / Size;
                return tupi::tuple_cat(                     //
                    tupi::make_broadcast_tuple<repeats>(    //
                        get_tw_value<Is>())...);
            }(std::make_index_sequence<Size / 2>{});
        };
    } const_tw_getter{};
};

template<typename T, bool LocalTw>
struct tw_data_t {
    static constexpr bool local = LocalTw;
};
template<typename T>
struct tw_data_t<T, false> {
    static constexpr bool local = false;

    const T* tw_ptr;
};
template<typename T>
struct tw_data_t<T, true> {
    static constexpr bool local = true;

    uZ start_fft_size;
    uZ start_k;
};


template<uZ NodeSize, typename T, uZ Width>
struct subtransform {
    using btfly_node = btfly_node_dit<NodeSize, T, Width>;
    using vec_traits = simd::detail_::vec_traits<T, Width>;


    struct align_node_t {
        uZ node_size_pre  = 1;
        uZ node_size_post = 1;
    };

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK, bool LocalTw>
    static void perform(uZ data_size, T* dest_ptr, tw_data_t<T, LocalTw> tw) {
        constexpr auto single_load_size = NodeSize * Width;

        auto lsize          = data_size / single_load_size;
        auto slog           = log2i(lsize);
        auto a              = slog / log2i(NodeSize);
        auto b              = a * log2i(NodeSize);
        auto pre_align_node = powi(2, slog - b);

        [&]<uZ... Is>(std::index_sequence<Is...>) {
            auto check_align = [&]<uZ I>(uZ_ce<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != pre_align_node)
                    return false;
                perform_impl<DestPackSize, SrcPackSize, LowK, {l_node_size, 1}>(data_size, dest_ptr, tw);
                return true;
            };
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(std::make_index_sequence<log2i(NodeSize)>{});
    }

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK, align_node_t AlignNode, bool LocalTw>
    static void perform_impl(uZ                    data_size,    // NOLINT (*complexity*)
                             T*                    dest_ptr,
                             tw_data_t<T, LocalTw> tw_data) {
        constexpr auto single_load_size = NodeSize * Width;

        uZ k_count = 1;
        if constexpr (AlignNode.node_size_pre != 1) {
            constexpr auto align_node = AlignNode.node_size_pre;
            fft_iteration<align_node, Width, SrcPackSize, LowK>(data_size, k_count, dest_ptr, tw_data);
            if constexpr (LowK && !LocalTw)
                tw_data.tw_ptr += k_count;
        } else {
            fft_iteration<NodeSize, Width, SrcPackSize, LowK>(data_size, k_count, dest_ptr, tw_data);
        }

        while (data_size / (k_count * NodeSize) >= single_load_size)
            fft_iteration<NodeSize, Width, Width, LowK>(data_size, k_count, dest_ptr, tw_data);

        if constexpr (LowK && !LocalTw) {
            if (k_count > AlignNode.node_size_pre)
                tw_data.tw_ptr += k_count;
        }

        if constexpr (AlignNode.node_size_post != 1) {
            constexpr auto align_node = AlignNode.node_size_post;
            fft_iteration<align_node, Width, Width, LowK>(data_size, k_count, dest_ptr, tw_data);
            if constexpr (LowK && !LocalTw)
                tw_data.tw_ptr += k_count;
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
            single_load<DestPackSize, Width, LowK>(dest_ptr, dest_ptr, tw_data);
        }
        constexpr auto start = LowK ? 1UZ : 0UZ;
        for (auto i: stdv::iota(start, data_size / single_load_size)) {
            auto dest = dest_ptr + i * single_load_size * 2;
            single_load<DestPackSize, Width, false>(dest, dest, tw_data);
        }
    };

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK, align_node_t AlignNode, bool LocalTw>
    void insert_impl(auto r, uZ data_size, auto tw_data) {
        constexpr auto single_load_size = NodeSize * Width;

        uZ k_count = 1;
        if constexpr (AlignNode.node_size_pre != 1) {
            constexpr auto align_node = AlignNode.node_size_pre;
            insert_iteration_tw<align_node, LowK>(r, k_count, tw_data);
        }

        while (data_size / (k_count * NodeSize) >= single_load_size) {
            if constexpr (!LowK) {
                k_count *= NodeSize;
            } else {
                insert_iteration_tw<NodeSize, LowK>(r, k_count, tw_data);
            }
        }
        if constexpr (!LowK)
            insert_iteration_tw<NodeSize, LowK>(r, k_count, tw_data);

        if constexpr (AlignNode.node_size_post != 1) {
            constexpr auto align_node = AlignNode.node_size_post;
            insert_iteration_tw<align_node, LowK>(r, k_count, tw_data);
        }

        insert_single_load_tw<LowK>(r, tw_data);
        for (auto i: stdv::iota(1U, data_size / single_load_size)) {
            insert_single_load_tw<false>(r, tw_data);
        }
    }

    // private:
    static constexpr auto half_node_idxs = std::make_index_sequence<NodeSize / 2>{};

    template<uZ NodeSizeL, uZ PackDest, uZ PackSrc, bool LowK, bool LocalTw>
    PCX_AINLINE static auto
    fft_iteration(uZ data_size, uZ& k_count, auto data_ptr, tw_data_t<T, LocalTw>& tw_data) {
        using btfly_node        = btfly_node_dit<NodeSizeL, T, Width>;
        constexpr auto settings = val_ce<typename btfly_node::settings{
            .pack_dest = PackDest,
            .pack_src  = PackSrc,
            .reverse   = false,
        }>{};

        using l_twd_t = std::conditional_t<LowK && !LocalTw, tw_data_t<T, LocalTw>, tw_data_t<T, LocalTw>&>;
        l_twd_t l_tw_data = tw_data;

        auto make_tw_tup = [&l_tw_data] {
            constexpr uZ n_tw = NodeSizeL / 2;
            if constexpr (LocalTw) {
                return [&l_tw_data] PCX_LAINLINE(uZ k) {
                    return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        auto tws = make_tw_node<T, NodeSizeL>(l_tw_data.start_fft_size * 2,    //
                                                              l_tw_data.start_k + k);

                        auto tw_ptr = reinterpret_cast<T*>(tws.data());
                        return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + 2 * Is)...);
                    }(make_uZ_seq<n_tw>{});
                };
            } else {
                return [&l_tw_data] PCX_LAINLINE(uZ) {
                    return [&l_tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        auto l_tw_ptr = l_tw_data.tw_ptr;
                        l_tw_data.tw_ptr += n_tw * 2;
                        return tupi::make_tuple(simd::cxbroadcast<1, Width>(l_tw_ptr + Is * 2)...);
                    }(make_uZ_seq<n_tw>{});
                };
            }
        }();

        auto group_size    = data_size / k_count / 2;
        auto make_data_tup = [=] PCX_LAINLINE(uZ k, uZ offset) {
            auto node_stride = group_size / NodeSizeL * 2 /*second group half*/ * 2 /*complex*/;
            auto k_stride    = group_size * 2 /*second group half*/ * 2 /*complex*/;
            auto base_ptr    = data_ptr + k * k_stride + offset * 2;
            return [base_ptr, node_stride]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                return tupi::make_tuple((base_ptr + node_stride * Is)...);
            }(make_uZ_seq<NodeSizeL>{});
        };

        if constexpr (LowK) {
            auto tw = make_tw_tup(0);
            for (auto i: stdv::iota(0U, group_size / NodeSizeL * 2) | stdv::stride(Width)) {
                auto data = make_data_tup(0, i);
                btfly_node::perform(settings, data);
            }
        }
        constexpr auto k_start = LowK ? 1UZ : 0UZ;
        for (auto k: stdv::iota(k_start, k_count)) {
            auto tw = make_tw_tup(k);
            for (auto i: stdv::iota(0U, group_size / NodeSizeL * 2) | stdv::stride(Width)) {
                auto data = make_data_tup(k, i);
                btfly_node::perform(settings, data, tw);
            }
        }
        k_count *= NodeSizeL;
        if constexpr (LocalTw) {
            l_tw_data.start_fft_size *= NodeSizeL;
            l_tw_data.start_k *= NodeSizeL;
        }
    }
    template<uZ NodeSizeL, bool LowK>
    void insert_iteration_tw(/* cx_range */ auto& r, uZ k_count, auto& tw_data) {
        // if constexpr(!LowK)
        {
            r.append_range(make_tw_node<T, NodeSizeL>(tw_data.start_fft_size * 2,    //
                                                      tw_data.start_k));
        }
        for (auto k: stdv::iota(1U, k_count)) {
            r.append_range(make_tw_node<T, NodeSizeL>(tw_data.start_fft_size * 2,    //
                                                      tw_data.start_k + k));
        }
    };

    template<uZ DestPackSize, uZ SrcPackSize, bool LowK, bool LocalTw>
    PCX_AINLINE static auto single_load(T* data_ptr, const T* src_ptr, tw_data_t<T, LocalTw>& tw_data) {
        auto data = []<uZ... Is> PCX_LAINLINE(auto data_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxload<SrcPackSize, Width>(data_ptr + Width * 2 * Is)...);
        }(src_ptr, std::make_index_sequence<NodeSize>{});
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);

        auto get_tw = [&tw_data] {
            if constexpr (LocalTw) {
                return [=]<uZ... Is>(uZ_seq<Is...>) {
                    auto tws    = make_tw_node<T, NodeSize>(tw_data.start_fft_size * 2, tw_data.start_k);
                    auto tw_ptr = reinterpret_cast<T*>(tws.data());
                    return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + 2 * Is)...);
                }(make_uZ_seq<NodeSize / 2>{});
            } else {
                auto tw0 = [tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_data.tw_ptr + 2 * Is)...);
                }(make_uZ_seq<NodeSize / 2>{});
                tw_data.tw_ptr += NodeSize;
                return tw0;
            }
        };
        auto btfly_res_0 = [&] PCX_LAINLINE {
            if constexpr (LowK) {
                get_tw();
                return btfly_node::forward(data_rep);
            } else {
                return btfly_node::forward(data_rep, get_tw());
            }
        }();

        auto regroup_tw_fact = [&]<uZ TwCount> PCX_LAINLINE(uZ_ce<TwCount>) {
            if constexpr (LocalTw) {
                return [=]<uZ KGroup>(uZ_ce<KGroup>) {
                    auto fft_size = tw_data.start_fft_size * NodeSize * TwCount;
                    auto k        = tw_data.start_k * NodeSize * TwCount / 2 + KGroup * TwCount;
                    auto tw_arr   = [=]<uZ... Is>(uZ_seq<Is...>) {
                        return std::array{wnk_br<T>(fft_size, k + Is)...};
                    }(make_uZ_seq<TwCount>{});
                    auto tw = simd::cxload<1, TwCount>(reinterpret_cast<T*>(tw_arr.data()));
                    return tw;
                };
            } else {
                auto l_tw_ptr = tw_data.tw_ptr;
                tw_data.tw_ptr += TwCount * 2 * NodeSize / 2;
                return [l_tw_ptr]<uZ KGroup> PCX_LAINLINE(uZ_ce<KGroup>) {
                    auto tw = simd::cxload<1, TwCount>(l_tw_ptr + TwCount * (2 * KGroup));
                    return tw;
                };
            }
        };

        auto [data_lo, data_hi] = [&]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            auto lo = tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
            auto hi = tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
            return tupi::make_tuple(lo, hi);
        }(half_node_idxs);

        auto [lo, hi] = [regroup_tw_fact]<uZ NGroups = 2> PCX_LAINLINE    //
            (this auto f, auto data_lo, auto data_hi, uZ_ce<NGroups> = {}) {
                if constexpr (NGroups == Width) {
                    return regroup_btfly<NGroups>(data_lo, data_hi, regroup_tw_fact(uZ_ce<NGroups>{}));
                } else {
                    auto [lo, hi] =
                        regroup_btfly<NGroups>(data_lo, data_hi, regroup_tw_fact(uZ_ce<NGroups>{}));
                    return f(lo, hi, uZ_ce<NGroups * 2>{});
                }
            }(data_lo, data_hi);
        auto btfly_res_1 = tupi::group_invoke(regroup<1, Width>, lo, hi);
        auto res         = tupi::make_flat_tuple(btfly_res_1);
        auto res_rep     = tupi::group_invoke(simd::evaluate | simd::repack<DestPackSize>, res);
        [data_ptr, res_rep]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            (simd::cxstore<DestPackSize>(data_ptr + Width * 2 * Is, get<Is>(res_rep)), ...);
        }(std::make_index_sequence<NodeSize>{});
        if constexpr (LocalTw) {
            ++tw_data.start_k;
        }
    }
    template<bool LowK>
    void insert_single_load_tw(/* cx_range */ auto& r, auto& tw_data) {
        // // if constexpr(!LowK)
        // {
        //     r.append_range(make_tw_node<T, NodeSize>(tw_data.start_fft_size * 2,    //
        //                                              tw_data.start_k));
        // }
        // auto insert_kg = [&]<uZ TwCount, uZ KGroup>(uZ_ce<TwCount>, uZ_ce<KGroup>) {
        //     auto fft_size = tw_data.start_fft_size * NodeSize * TwCount;
        //     auto k        = tw_data.start_k * NodeSize * TwCount / 2 + KGroup * TwCount;
        //     r.append_range([=]<uZ... Is>(uZ_seq<Is...>) {
        //         return std::array{wnk_br<T>(fft_size, k + Is)...};
        //     }(make_uZ_seq<TwCount>{}));
        // };
        // []<uZ NGroups = 2> PCX_LAINLINE    //
        //     (this auto f, uZ_ce<NGroups> = {}) {
        //         if constexpr (NGroups == Width) {
        //             insert_kg(uZ_ce<KGroup>)regroup_tw_fact(uZ_ce<NGroups>{}));
        //         } else {
        //             auto [lo, hi] =
        //                 regroup_btfly<NGroups>(data_lo, data_hi, regroup_tw_fact(uZ_ce<NGroups>{}));
        //             return f(lo, hi, uZ_ce<NGroups * 2>{});
        //         }
        //     }();


        ++tw_data.start_k;
    }
    // insert_single_load_tw(cx_range auto r, tw_data);

    template<uZ NGroups>
    struct regroup_btfly_t {
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...> lo, tupi::tuple<Thi...> hi, auto&& get_tw) {
            auto tw_tup = tupi::make_broadcast_tuple<NodeSize / 2>(get_tw);

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
            return tupi::make_tuple(uZ_ce<Is>{}...);
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
        | []<uZ KGroup>(auto&& get_tw, uZ_ce<KGroup> k) {
            auto tw = get_tw(k);
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

    /**
     * @brief Number of complex elements of type T that keep L1 cache coherency during subtransforms.
     */
    static constexpr uZ coherent_size  = 2048;
    static constexpr uZ lane_size      = std::max(64 / sizeof(T) / 2, Width);
    static constexpr uZ coherent_k_cnt = coherent_size / lane_size;


    /**
     * Subdivides the data into buckets of `coherent_size`.
     * A bucket represents contiguous chunks of `line_size` distributed with a constant stride.
     * A single subdivision can perform a maximum of `coherent_k_cnt` levels of fft.
     *
     */
    template<uZ DestPackSize, uZ SrcPackSize, uZ NSubdiv>
    static void perform(uZ data_size, T* dest_ptr, const T* tw_ptr, uZ n_subdiv) {
        // constant stride
        //  [s0, s1,    ..., s0, ...    ]
        //  [0 , width, ..., stride, ...]

        uZ n_buckets     = data_size / coherent_size;    // per bucket group
        uZ align_k_count = data_size / (coherent_size * powi(coherent_k_cnt, NSubdiv - 1));
        uZ stride        = Width * n_buckets;
        uZ k_count       = 1;
        {
            auto l_tw_ptr = tw_ptr;
            for (uZ i_b: stdv::iota(0U, n_buckets)) {
                auto bucket_ptr = dest_ptr + i_b * lane_size * 2;
                l_tw_ptr        = tw_ptr;
                sparse_subtform<SrcPackSize, 1, true>(stride,    //
                                                      k_count,
                                                      align_k_count,
                                                      bucket_ptr,
                                                      l_tw_ptr);
            }
            stride /= align_k_count;
            n_buckets /= align_k_count;
        }

        auto n_bucket_groups = align_k_count;
        for (auto d: stdv::iota(1U, NSubdiv)) {
            for (uZ i_b: stdv::iota(0U, n_buckets)) {
                auto bucket_ptr = dest_ptr + i_b * lane_size * 2;
                sparse_subtform<Width, 1, true>(stride,    //
                                                k_count,
                                                align_k_count,
                                                bucket_ptr,
                                                tw_ptr);
            }
            for (auto i_bucket_grp: stdv::iota(1U, n_bucket_groups)) {
                auto bucket_grp_ptr = dest_ptr + i_bucket_grp * data_size / n_bucket_groups;
                auto l_tw_ptr       = tw_ptr + 0 * i_bucket_grp;
                for (uZ i_b: stdv::iota(0U, n_buckets)) {
                    auto bucket_ptr = bucket_grp_ptr + i_b * lane_size * 2;
                    sparse_subtform<Width, 1, false>(stride,    //
                                                     k_count,
                                                     align_k_count,
                                                     bucket_ptr,
                                                     l_tw_ptr);
                }
            }
            n_bucket_groups *= coherent_k_cnt;
            stride /= coherent_k_cnt;
        }
    };


    template<uZ PackSrc, uZ PackAlign = 1, bool LowK = false>
    PCX_AINLINE static void sparse_subtform(uZ       stride,    //
                                            uZ       k_count,
                                            uZ       final_k_count,
                                            T*       dest_ptr,
                                            const T* tw_ptr) {
        if constexpr (PackAlign != 1) {
            fft_iteration_cs<PackAlign, PackSrc, LowK>(stride, dest_ptr, k_count, tw_ptr);
        } else if constexpr (PackSrc != Width) {
            fft_iteration_cs<NodeSize, PackSrc, LowK>(stride, dest_ptr, k_count, tw_ptr);
        }
        while (k_count < final_k_count)
            fft_iteration_cs<NodeSize, Width, LowK>(stride, dest_ptr, k_count, tw_ptr);
    };

    template<uZ NodeSizeL, uZ PackSrc, bool LowK, bool LocalTw>
    PCX_AINLINE static auto fft_iteration_cs(uZ                    stride,    //
                                             auto                  data_ptr,
                                             uZ&                   k_count,
                                             tw_data_t<T, LocalTw> tw_data
                                             // uZ&  fft_size,
                                             /* uZ&  starting_k */
    ) {
        using btfly_node        = btfly_node_dit<NodeSizeL, T, Width>;
        constexpr auto settings = val_ce<typename btfly_node::settings{
            .pack_dest = Width,
            .pack_src  = PackSrc,
            .reverse   = false,
        }>{};

        // auto data_size     = 0U;
        // auto group_size    = data_size / k_count / 2;
        // auto make_data_tup = [=] PCX_LAINLINE(uZ i, uZ k) {
        //     auto node_stride = data_size / k_count / 2 / NodeSizeL * 2 /*second group half*/ * 2 /*complex*/;
        //     auto i_stride    = data_size / k_count / 2 * 2 /*second group half*/ * 2 /*complex*/;
        //     auto base_ptr    = data_ptr
        //                        + i * i_stride
        //                        + k * Width * 2;
        //     return [base_ptr, node_stride]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
        //         return tupi::make_tuple((base_ptr + node_stride * Is)...);
        //     }(make_uZ_seq<NodeSizeL>{});
        // };
        //
        // stride == width

        auto k_group_size = coherent_size / k_count;
        auto k_stride     = stride * coherent_size / Width / k_count;

        auto make_data_tup = [=] PCX_LAINLINE(uZ i, uZ k) {
            auto base_ptr = data_ptr            //
                            + i * stride * 2    //
                            + k * NodeSizeL * k_stride * 2;
            return [k_stride]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                return tupi::make_tuple((base_ptr + k_stride * Is)...);
            }(make_uZ_seq<NodeSizeL>{});
        };

        constexpr auto n_tw = NodeSizeL / 2;

        auto make_tw_tup = [&tw_data] {
            constexpr uZ n_tw = NodeSizeL / 2;
            if constexpr (LocalTw) {
                return [&tw_data] PCX_LAINLINE(uZ k) {
                    return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        auto tws = make_tw_node<T, NodeSizeL>(tw_data.start_fft_size * 2,    //
                                                              tw_data.start_k + k);

                        auto tw_ptr = reinterpret_cast<T*>(tws.data());
                        return tupi::make_tuple(simd::cxbroadcast<1, Width>(tw_ptr + 2 * Is)...);
                    }(make_uZ_seq<n_tw>{});
                };
            } else {
                // return [&tw_data] PCX_LAINLINE(uZ) {
                //     return [&tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                //         auto l_tw_ptr = tw_data.tw_ptr;
                //         tw_data.tw_ptr += n_tw * 2;
                //         return tupi::make_tuple(simd::cxbroadcast<1, Width>(l_tw_ptr + Is * 2)...);
                //     }(make_uZ_seq<n_tw>{});
                // };
            }
        }();

        if constexpr (LowK) {
            // l_tw_ptr += n_tw * 2;
            for (auto i: stdv::iota(0U, k_group_size / lane_size)) {
                // for (auto r : stdv::iota(0U, lane_size/Width))
                auto data = make_data_tup(i, 0);
                btfly_node::perform(settings, data);
            }
        }
        constexpr auto k_start = LowK ? 1U : 0U;
        for (auto k_group: stdv::iota(k_start, k_count)) {
            auto tw = [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                auto tws = make_tw_node<T, NodeSizeL>(tw_data.start_fft_size, tw_data.start_k + k_group);
                return tupi::make_tuple(simd::cxbroadcast<1, Width>(&tws[Is])...);
            }(make_uZ_seq<n_tw>{});
            // l_tw_ptr += n_tw * 2;

            for (auto i: stdv::iota(0U, k_group_size / lane_size)) {
                // for (auto r : stdv::iota(0U, lane_size/Width))
                auto data = make_data_tup(i, k_group);
                btfly_node::perform(settings, data, tw);
            }
        }
        k_count *= NodeSizeL;
    }
};

}    // namespace pcx::detail_
