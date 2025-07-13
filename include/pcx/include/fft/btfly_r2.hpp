#pragma once
#include "pcx/include/fft/btfly_common.hpp"
#include "pcx/include/fft/util.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"

namespace pcx::detail_ {

template<uZ NodeSize, typename T, uZ Width>
    requires(NodeSize >= 2)
struct btfly_node_dit {
    static constexpr auto width     = uZ_ce<Width>{};
    static constexpr auto node_size = uZ_ce<NodeSize>{};

    using cx_vec = simd::cx_vec<T, false, false, Width>;

    struct settings {
        uZ   pack_dest;
        uZ   pack_src;
        bool reverse;
        bool conj_tw;
    };

    using dest_t = tupi::broadcast_tuple_t<T*, NodeSize>;
    using data_t = tupi::broadcast_tuple_t<cx_vec, NodeSize>;
    using src_t  = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using tw_t   = tupi::broadcast_tuple_t<cx_vec, NodeSize / 2>;

    PCX_LAINLINE static auto forward(data_t                 data,    //
                                     meta::ce_of<bool> auto lowk,
                                     tw_t                   tw,
                                     meta::ce_of<bool> auto conj_tw) {
        if constexpr (lowk)
            return fwd_impl(data, const_tw_getter, conj_tw);
        else
            return fwd_impl(data, make_tw_getter(tw), conj_tw);
    }
    PCX_LAINLINE static auto reverse(data_t                 data,    //
                                     meta::ce_of<bool> auto lowk,
                                     tw_t                   tw,
                                     meta::ce_of<bool> auto conj_tw) {
        if constexpr (lowk)
            return rev_impl(data, const_tw_getter, conj_tw);
        else
            return rev_impl(data, make_tw_getter(tw), conj_tw);
    }

    template<settings S>
    PCX_AINLINE static void
    perform_bf(val_ce<S>, meta::ce_of<bool> auto lowk, dest_t dest, src_t src, tw_t tw) {
        auto data    = tupi::group_invoke(simd::cxload<S.pack_src, Width> | simd::repack<Width>, src);
        auto res     = S.reverse ? reverse(data, lowk, tw, val_ce<S.conj_tw>{})
                                 : forward(data, lowk, tw, val_ce<S.conj_tw>{});
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<S.pack_dest>, res);
        tupi::group_invoke(simd::cxstore<S.pack_dest>, dest, res_rep);
    }
    template<settings S>
    PCX_AINLINE static void perform_bf(val_ce<S> s, meta::ce_of<bool> auto lowk, dest_t dest, tw_t tw) {
        perform_bf(s, lowk, dest, dest, tw);
    }

    PCX_AINLINE static auto fwd_impl(data_t data, auto get_tw, auto conj_tw) {
        return [=]<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_ce<Size> size = {}) {
                if constexpr (size == NodeSize) {
                    return btfly_impl(size, data, get_tw(size), conj_tw);
                } else {
                    auto tmp = btfly_impl(size, data, get_tw(size), conj_tw);
                    return f(tmp, get_tw, uZ_ce<size * 2>{});
                }
            }(data, get_tw);
    }
    PCX_AINLINE static auto rev_impl(data_t data, auto get_tw, auto conj_tw) {
        return [=]<uZ Size = NodeSize> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_ce<Size> size = {}) {
                if constexpr (size == 2) {
                    return rbtfly_impl(size, data, get_tw(size), conj_tw);
                } else {
                    auto tmp = rbtfly_impl(size, data, get_tw(size), conj_tw);
                    return f(tmp, get_tw, uZ_ce<size / 2>{});
                }
            }(data, get_tw);
    }
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto btfly_impl(uZ_ce<Size>, tupi::tuple<Ts...> data, auto tws, auto conj_tw) {
        constexpr auto stride = NodeSize / Size * 2;

        auto maybe_conj = [=](auto tw) {
            if constexpr (conj_tw) {
                return conj(tw);
            } else {
                return tw;
            }
        };

        auto [lo, hi]  = extract_halves<stride>(data);
        auto ctw       = tupi::group_invoke(maybe_conj, tws);
        auto hi_tw     = tupi::group_invoke(simd::mul, hi, ctw);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi_tw);
        auto new_lo    = tupi::group_invoke(tupi::get_copy<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get_copy<1>, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto rbtfly_impl(uZ_ce<Size>, tupi::tuple<Ts...> data, auto tws, auto conj_tw) {
        constexpr auto ns     = NodeSize;
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi);
        auto new_lo    = tupi::group_invoke(tupi::get_copy<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get_copy<1>, btfly_res);
        auto ctw       = tupi::group_invoke(simd::maybe_conj<conj_tw>, tws);
        auto new_hi_tw = tupi::group_invoke(simd::mul, new_hi, ctw);
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
        constexpr auto count = sizeof...(Ts);
        auto get_half        = [=]<uZ... Grp, uZ Start> PCX_LAINLINE(uZ_seq<Grp...>, uZ_ce<Start>) {
            auto iterate = [=]<uZ... Iters, uZ Offset> PCX_LAINLINE(uZ_seq<Iters...>, uZ_ce<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Iters>(data)...);
            };
            return tupi::tuple_cat(iterate(make_uZ_seq<Stride / 2>{}, uZ_ce<Start + Grp * Stride>{})...);
        };
        return tupi::make_tuple(get_half(make_uZ_seq<count / Stride>{}, uZ_ce<0>{}),
                                get_half(make_uZ_seq<count / Stride>{}, uZ_ce<Stride / 2>{}));
    }
    /**
     * @brief Combines two halves into a tuple
     *
     * lo     = [0,          1,              ..., Stride / 2 - 1, Stride        , Stride + 1,         ... ]
     * hi     = [Stride / 2, Stride / 2 + 1, ..., Stride - 1    , Stride * 3 / 2, Stride * 3 / 2 + 1, ... ]
     * return = [0, 1, ..., N - 1] 
     */
    template<uZ Stride, typename... Tsl, typename... Tsh>
        requires(simd::any_cx_vec<Tsl> && ...) && (simd::any_cx_vec<Tsh> && ...)
    PCX_AINLINE static auto combine_halves(tupi::tuple<Tsl...> lo, tupi::tuple<Tsh...> hi) {
        constexpr auto        count = sizeof...(Tsl) * 2;
        return [=]<uZ... Grp> PCX_LAINLINE(uZ_seq<Grp...>) {
            auto iterate = [=]<uZ... Is, uZ Offset> PCX_LAINLINE(uZ_seq<Is...>, uZ_ce<Offset>) {
                return tupi::make_tuple(tupi::get<Offset + Is>(lo)..., tupi::get<Offset + Is>(hi)...);
            };
            return tupi::tuple_cat(iterate(make_uZ_seq<Stride / 2>{}, uZ_ce<Grp * Stride / 2>{})...);
        }(make_uZ_seq<count / Stride>{});
    }

    PCX_AINLINE static auto make_tw_getter(tw_t tw) {
        return [tw]<uZ Size> PCX_LAINLINE(uZ_ce<Size>) {
            return [&]<uZ... Itw> PCX_LAINLINE(uZ_seq<Itw...>) {
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
            }(make_uZ_seq<Size / 4>{});
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
                return simd::cxbroadcast<1, Width>(&tupi::get<I>(values));
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

    static constexpr bool skip_lowk_tw = true;

    static auto make_tw_node(meta::ce_of<bool> auto lowk, uZ fft_size, uZ k) {
        if constexpr (lowk && skip_lowk_tw) {
            return std::array<std::complex<T>, 0>{};
        } else {
            constexpr auto n_tw = node_size / 2;

            auto tw_node = std::array<std::complex<T>, n_tw>{};
            uZ   i_tw    = 0;
            for (uZ l: stdv::iota(0U, log2i(node_size))) {
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
    }

    static auto next_tw(meta::ce_of<bool> auto lowk,
                        meta::ce_of<bool> auto reverse,
                        tw_data_for<T> auto&   tw_data) -> tw_t {
        const auto   local_tw = tw_data.is_local();
        constexpr uZ n_tw     = node_size / 2;
        if constexpr (local_tw) {
            if constexpr (reverse)
                recede_tw(lowk, tw_data, 1);
            auto tws = make_tw_node(lowk, tw_data.start_fft_size * 2, tw_data.k);
            if constexpr (!reverse)
                advance_tw(lowk, tw_data, 1);

            if constexpr (lowk && skip_lowk_tw) {
                return {};
            } else {
                return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(simd::cxbroadcast<1, width>(tws.data() + Is)...);
                }(make_uZ_seq<tws.size()>{});
            }
        } else {
            if constexpr (lowk && skip_lowk_tw) {
                return {};
            }
            return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                if constexpr (reverse)
                    tw_data.tw_ptr -= n_tw * 2;
                auto tws = tupi::make_tuple(simd::cxbroadcast<1, width>(tw_data.tw_ptr + Is * 2)...);
                if constexpr (!reverse)
                    tw_data.tw_ptr += n_tw * 2;
                return tws;
            }(make_uZ_seq<n_tw>{});
        }
    };
    static void advance_tw(meta::ce_of<bool> auto lowk,    //
                           tw_data_for<T> auto&   tw,
                           uZ                     k_count) {
        const auto local_tw = tw.is_local();
        if constexpr (!local_tw) {
            if (lowk && skip_lowk_tw)
                k_count -= node_size;
            tw.tw_ptr += k_count;
        } else {
            assert(k_count <= (tw.k_end - tw.k));
            tw.k += k_count;
            if (tw.k == tw.k_end) {
                tw.start_fft_size *= node_size;
                tw.k_begin *= node_size;
                tw.k_end *= node_size;
                tw.k = tw.k_begin;
            }
        }
    };
    static void recede_tw(meta::ce_of<bool> auto lowk,    //
                          tw_data_for<T> auto&   tw,
                          uZ                     k_count) {
        const auto local_tw = tw.is_local();
        if constexpr (!local_tw) {
            if (lowk && skip_lowk_tw)
                k_count -= node_size;
            tw.tw_ptr -= k_count;
        } else {
            if (tw.k == tw.k_begin) {
                tw.start_fft_size /= node_size;
                tw.k_end /= node_size;
                tw.k_begin /= node_size;
                tw.k = tw.k_end;
            }
            assert(k_count <= (tw.k - tw.k_begin));
            tw.k -= k_count;
        }
    };
};
}    // namespace pcx::detail_
