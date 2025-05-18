#pragma once
#include "pcx/include/fft/permuters.hpp"
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
    static constexpr auto width = uZ_ce<Width>{};

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
    struct low_k_tw_t {};

    template<bool ConjTw = false>
    PCX_LAINLINE static auto forward(data_t data, low_k_tw_t = {}, val_ce<ConjTw> conj_tw = {}) {
        return fwd_impl(data, const_tw_getter, conj_tw);
    }
    template<bool ConjTw = false>
    PCX_LAINLINE static auto forward(data_t data, const tw_t& tw, val_ce<ConjTw> conj_tw = {}) {
        return fwd_impl(data, make_tw_getter(tw), conj_tw);
    }
    template<bool ConjTw = false>
    PCX_LAINLINE static auto reverse(data_t data, low_k_tw_t = {}, val_ce<ConjTw> conj_tw = {}) {
        return rev_impl(data, const_tw_getter, conj_tw);
    }
    template<bool ConjTw = false>
    PCX_LAINLINE static auto reverse(data_t data, const tw_t& tw, val_ce<ConjTw> conj_tw = {}) {
        return rev_impl(data, make_tw_getter(tw), conj_tw);
    }

    template<settings S, typename Tw = low_k_tw_t>
        requires std::same_as<Tw, tw_t> || std::same_as<Tw, low_k_tw_t>
    PCX_AINLINE static void perform(val_ce<S>, const dest_t& dest, const src_t& src, const Tw& tw = {}) {
        auto data = tupi::group_invoke(simd::cxload<S.pack_src, Width> | simd::repack<Width>, src);
        auto res =
            S.reverse ? reverse(data, tw, val_ce<S.conj_tw>{}) : forward(data, tw, val_ce<S.conj_tw>{});
        auto res_rep = tupi::group_invoke(simd::evaluate | simd::repack<S.pack_dest>, res);
        tupi::group_invoke(simd::cxstore<S.pack_dest>, dest, res_rep);
    }
    template<settings S, typename Tw = low_k_tw_t>
        requires std::same_as<Tw, tw_t> || std::same_as<Tw, low_k_tw_t>
    PCX_AINLINE static void perform(val_ce<S> s, const dest_t& dest, const Tw& tw = {}) {
        perform(s, dest, dest, tw);
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
};

inline constexpr auto not_lowk = std::false_type{};

template<floating_point T, bool LocalTw>
struct tw_data_t;
template<floating_point T>
struct tw_data_t<T, false> {
    static constexpr auto is_local() -> std::false_type {
        return {};
    };

    const T* tw_ptr;
};

template<floating_point T>
struct tw_data_t<T, true> {
    static constexpr auto is_local() -> std::true_type {
        return {};
    };

    uZ start_fft_size = 1;
    uZ start_k        = 0;
};

template<typename T, floating_point fX>
struct is_tw_data_of : std::false_type {};
template<floating_point fX, bool LocalTw>
struct is_tw_data_of<tw_data_t<fX, LocalTw>, fX> : std::true_type {};
template<typename T, typename fX>
concept tw_data_for = is_tw_data_of<T, fX>::value;

template<uZ AlignNodeSize, bool PreAlign>
    requires power_of_two<AlignNodeSize>
struct align_param {
    static constexpr auto size_pre() -> uZ_ce<PreAlign ? AlignNodeSize : 1> {
        return {};
    };
    static constexpr auto size_post() -> uZ_ce<PreAlign ? 1 : AlignNodeSize> {
        return {};
    };
};

template<typename T>
struct is_align_param : public std::false_type {};
template<uZ AlignNodeSize, bool PreAlign>
struct is_align_param<align_param<AlignNodeSize, PreAlign>> : public std::true_type {};

template<typename T>
concept any_align = is_align_param<T>::value;

template<typename R, typename T>
concept twiddle_range_for = stdr::contiguous_range<R>                     //
                            && std::same_as<stdr::range_value_t<R>, T>    //
                            && requires(R r, T v) { r.push_back(v); };

template<uZ NodeSize, typename T, uZ Width>
struct subtransform {
    static constexpr auto node_size = uZ_ce<NodeSize>{};
    static constexpr auto width     = uZ_ce<Width>{};
    static constexpr auto w_pck     = cxpack<Width, T>{};

    static constexpr bool skip_lowk_tw = true;

    template<uZ NodeSizeL>
    PCX_AINLINE static void fft_iteration(cxpack_for<T> auto          dst_pck,
                                          cxpack_for<T> auto          src_pck,
                                          meta::ce_of<bool> auto      lowk,
                                          meta::ce_of<bool> auto      reverse,
                                          meta::ce_of<bool> auto      conj_tw,
                                          meta::maybe_ce_of<uZ> auto  batch_count,
                                          meta::maybe_ce_of<uZ> auto  batch_size,
                                          data_info_for<T> auto       dst_data,
                                          data_info_for<const T> auto src_data,
                                          uZ&                         k_count,
                                          tw_data_for<T> auto&        tw_data) {
        uZ w                    = width;
        uZ nsl                  = NodeSizeL;
        using btfly_node        = btfly_node_dit<NodeSizeL, T, width>;
        constexpr auto settings = val_ce<typename btfly_node::settings{
            .pack_dest = dst_pck,
            .pack_src  = src_pck,
            .reverse   = reverse,
            .conj_tw   = conj_tw,
        }>{};

        const auto inplace    = src_data.empty();
        const auto local      = tw_data.is_local();
        using l_tw_data_t     = std::conditional_t<lowk && !local, tw_data_t<T, local>, tw_data_t<T, local>&>;
        l_tw_data_t l_tw_data = tw_data;

        if constexpr (reverse) {
            k_count /= NodeSizeL;
            if constexpr (local) {
                l_tw_data.start_fft_size /= NodeSizeL;
                l_tw_data.start_k /= NodeSizeL;
            }
        }
        const auto k_batch_count = batch_count / k_count;

        // data division:
        // E - even, O - odd
        // [E0, E0,     ..., E0                  , O0, O0, ... , O<k_count - 1>, O<k_count - 1>, ...] subtform indexes
        // [i0, i1,     ..., i<batch_count/2 - 1>, i0, i1, ... , i0,             i1,             ...] batches
        // [0 , stride, ...                                                                         ]
        // iX is batch X

        // data for a butterfly:
        // k == 0, i == 0
        // [0,     ... , NodeSizeL/4 - 1,          ... , NodeSizeL/2 - 1, ... ] - tuple index
        // [E0:i0, ... , E0:i<k_group_size/2 - 1>, ... , O1:i0,           ... ]
        auto make_dst_tup = [=] PCX_LAINLINE(uZ i_b, uZ k, uZ offset) {
            return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                return tupi::make_tuple(
                    (dst_data.get_batch_base(i_b + k * k_batch_count + k_batch_count / NodeSizeL * Is)
                     + offset * 2)...);
            }(make_uZ_seq<NodeSizeL>{});
        };
        auto make_src_tup = [=] PCX_LAINLINE(uZ i_b, uZ k, uZ offset) {
            if constexpr (!inplace) {
                return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(
                        (src_data.get_batch_base(i_b + k * k_batch_count + k_batch_count / NodeSizeL * Is)
                         + offset * 2)...);
                }(make_uZ_seq<NodeSizeL>{});
            }
        };

        auto make_tw_tup = [&] {
            constexpr uZ n_tw = NodeSizeL / 2;
            if constexpr (local) {
                return [&l_tw_data] PCX_LAINLINE(uZ k) {
                    return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        auto tws = make_tw_node<T, NodeSizeL>(l_tw_data.start_fft_size * 2,    //
                                                              l_tw_data.start_k + k);
                        return tupi::make_tuple(simd::cxbroadcast<1, width>(tws.data() + Is)...);
                    }(make_uZ_seq<n_tw>{});
                };
            } else if constexpr (reverse && !lowk) {
                return [&] PCX_LAINLINE(uZ) {
                    return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        l_tw_data.tw_ptr -= n_tw * 2;
                        auto l_tw_ptr = l_tw_data.tw_ptr;
                        return tupi::make_tuple(simd::cxbroadcast<1, width>(l_tw_ptr + Is * 2)...);
                    }(make_uZ_seq<n_tw>{});
                };
            } else {
                return [&] PCX_LAINLINE(uZ) {
                    return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        auto l_tw_ptr = l_tw_data.tw_ptr;
                        l_tw_data.tw_ptr += n_tw * 2;
                        return tupi::make_tuple(simd::cxbroadcast<1, width>(l_tw_ptr + Is * 2)...);
                    }(make_uZ_seq<n_tw>{});
                };
            }
        }();

        if constexpr (lowk) {
            if constexpr (!skip_lowk_tw)
                make_tw_tup(0);
            for (auto i_batch: stdv::iota(0U, k_batch_count / NodeSizeL)) {
                for (auto r: stdv::iota(0U, batch_size / width)) {
                    auto dest = make_dst_tup(i_batch, 0, r * width);
                    if constexpr (inplace) {
                        btfly_node::perform(settings, dest);
                    } else {
                        auto src = make_src_tup(i_batch, 0, r * width);
                        btfly_node::perform(settings, dest, src);
                    }
                }
            }
        }
        auto k_range = [=] {
            if constexpr (reverse && !lowk) {
                return stdv::iota(0U, k_count) | stdv::reverse;
            } else {
                return stdv::iota(lowk ? 1U : 0U, k_count);
            }
        }();
        for (auto k_group: k_range) {
            auto tw = make_tw_tup(k_group);
            for (auto i_batch: stdv::iota(0U, k_batch_count / NodeSizeL)) {
                for (auto r: stdv::iota(0U, batch_size / width)) {
                    auto dest = make_dst_tup(i_batch, k_group, r * width);
                    if constexpr (inplace) {
                        btfly_node::perform(settings, dest, tw);
                    } else {
                        auto src = make_src_tup(i_batch, k_group, r * width);
                        btfly_node::perform(settings, dest, src, tw);
                    }
                }
            }
        }
        if constexpr (!reverse) {
            k_count *= NodeSizeL;
            if constexpr (local) {
                l_tw_data.start_fft_size *= NodeSizeL;
                l_tw_data.start_k *= NodeSizeL;
            }
        }
    }
    template<uZ NodeSizeL>
    static void insert_iteration_tw(twiddle_range_for<T> auto& r, auto& tw_data, uZ& k_count, bool lowk) {
        auto insert = [&](uZ k) {
            constexpr uZ n_tw = NodeSizeL / 2;
            auto         tws  = make_tw_node<T, NodeSizeL>(tw_data.start_fft_size * 2, tw_data.start_k + k);
            for (auto tw: tws) {
                r.push_back(tw.real());
                r.push_back(tw.imag());
            }
        };

        if (lowk && !skip_lowk_tw)
            insert(0);
        auto k_start = lowk ? 1U : 0U;
        for (auto k_group: stdv::iota(k_start, k_count)) {
            insert(k_group);
        }
        k_count *= NodeSizeL;
        tw_data.start_fft_size *= NodeSizeL;
        tw_data.start_k *= NodeSizeL;
    };

    PCX_AINLINE static void perform(cxpack_for<T> auto          dst_pck,
                                    cxpack_for<T> auto          src_pck,
                                    any_align auto              align,
                                    meta::ce_of<bool> auto      lowk,
                                    meta::ce_of<bool> auto      reverse,
                                    meta::ce_of<bool> auto      conj_tw,
                                    meta::maybe_ce_of<uZ> auto  batch_count,
                                    meta::maybe_ce_of<uZ> auto  batch_size,
                                    data_info_for<T> auto       dst_data,
                                    data_info_for<const T> auto src_data,
                                    uZ                          final_k_count,
                                    tw_data_for<T> auto&        tw_data) {
        assert(dst_pck == w_pck       //
               || src_pck == w_pck    //
               || final_k_count >= node_size);

        uZ         k_count{};
        const auto inplace    = src_data.empty();
        const auto local_tw   = tw_data.is_local();
        using l_tw_data_t     = std::conditional_t<local_tw, tw_data_t<T, local_tw>, tw_data_t<T, local_tw>&>;
        l_tw_data_t l_tw_data = tw_data;

        auto fft_iter = [&]<uZ NodeSizeL> PCX_LAINLINE(uZ_ce<NodeSizeL>,    //
                                                       auto dst_pck,
                                                       auto src_pck,
                                                       auto src) {
            fft_iteration<NodeSizeL>(dst_pck,
                                     src_pck,
                                     lowk,
                                     reverse,
                                     conj_tw,
                                     batch_count,
                                     batch_size,
                                     dst_data,
                                     src,
                                     k_count,
                                     l_tw_data);
        };

        if constexpr (!reverse) {
            k_count = 1;
            if constexpr (align.size_pre() != 1) {
                fft_iter(align.size_pre(), w_pck, src_pck, src_data);
                if constexpr (lowk && !local_tw)
                    l_tw_data.tw_ptr += k_count - (skip_lowk_tw ? align.size_pre() : 0);
            } else if constexpr (src_pck != w_pck || !inplace) {
                fft_iter(node_size, w_pck, src_pck, src_data);
            }

            auto fk = [&] {
                if constexpr (align.size_post() > 1) {
                    return final_k_count / align.size_post();
                } else if constexpr (dst_pck != w_pck) {
                    return final_k_count / node_size;
                } else {
                    return final_k_count;
                }
            }();

            while (k_count <= fk)
                fft_iter(node_size, w_pck, w_pck, inplace_src);

            if constexpr (align.size_post() != 1) {
                fft_iter(align.size_post(), w_pck, src_pck, src_data);
                if constexpr (lowk && !local_tw)
                    l_tw_data.tw_ptr += k_count - (skip_lowk_tw ? align.size_post() : 0);
            } else {
                if constexpr (dst_pck != w_pck)
                    fft_iter(node_size, dst_pck, w_pck, inplace_src);
                if constexpr (lowk && !local_tw) {
                    if (k_count > align.size_pre())
                        l_tw_data.tw_ptr += k_count - (skip_lowk_tw ? node_size : 0);
                }
            }
        } else {
            k_count = final_k_count * 2;
            if constexpr (align.size_post() != 1) {
                if constexpr (lowk && !local_tw)
                    l_tw_data.tw_ptr -= k_count - (skip_lowk_tw ? align.size_post() : 0);
                fft_iter(align.size_post(), w_pck, src_pck, src_data);
            } else {
                if (k_count > align.size_pre()) {
                    if constexpr (lowk && !local_tw)
                        l_tw_data.tw_ptr -= k_count - (skip_lowk_tw ? node_size : 0);
                }
                if constexpr (src_pck != w_pck || !inplace)
                    fft_iter(node_size, w_pck, src_pck, src_data);
            }
            constexpr auto fk = [&] {
                if constexpr (align.size_pre() > 1) {
                    return align.size_pre();
                } else if constexpr (dst_pck != w_pck) {
                    return node_size;
                } else {
                    return uZ_ce<1>{};
                }
            }();

            while (k_count > fk)
                fft_iter(node_size, w_pck, w_pck, inplace_src);

            if constexpr (align.size_pre() != 1) {
                if constexpr (lowk && !local_tw)
                    l_tw_data.tw_ptr -= k_count - (skip_lowk_tw ? align.size_pre() : 0);
                fft_iter(align.size_pre(), dst_pck, w_pck, inplace_src);
            } else if constexpr (dst_pck != w_pck) {
                fft_iter(node_size, dst_pck, w_pck, inplace_src);
            }
        }
    }

    static void insert_tw(twiddle_range_for<T> auto& r,
                          any_align auto             align,
                          bool                       lowk,
                          uZ                         final_k_count,
                          tw_data_t<T, true>&        tw_data) {
        uZ k_count = 1;
        if constexpr (align.size_pre() != 1)
            insert_iteration_tw<align.size_pre()>(r, tw_data, k_count, lowk);

        if (lowk) {
            k_count  = 2 * final_k_count / align.size_post() / node_size;
            auto d_k = 2 * k_count / align.size_pre();
            if (k_count > 0) {
                tw_data.start_fft_size *= d_k;
                tw_data.start_k *= d_k;
                insert_iteration_tw<node_size>(r, tw_data, k_count, lowk);
            }
        } else {
            while (k_count * align.size_post() <= final_k_count)
                insert_iteration_tw<node_size>(r, tw_data, k_count, lowk);
        }
        if constexpr (align.size_post() != 1)
            insert_iteration_tw<align.size_post()>(r, tw_data, k_count, lowk);
    }
};

template<uZ NodeSize, typename T, uZ Width>
struct sequential_subtransform {
    using btfly_node = btfly_node_dit<NodeSize, T, Width>;
    using vec_traits = simd::detail_::vec_traits<T, Width>;

    static constexpr auto width     = uZ_ce<Width>{};
    static constexpr auto node_size = uZ_ce<NodeSize>{};
    static constexpr auto w_pck     = cxpack<width, T>{};

    static constexpr bool skip_lowk_tw          = true;
    static constexpr bool skip_lowk_single_load = false;
    // for debugging
    static constexpr auto skip_single_load = false;
    static constexpr auto skip_sub_width   = false;

    static constexpr auto get_align_node(uZ data_size) {
        constexpr auto single_load_size = node_size * width;

        auto lsize      = data_size / single_load_size;
        auto slog       = log2i(lsize);
        auto a          = slog / log2i(node_size);
        auto b          = a * log2i(node_size);
        auto align_node = powi(2, slog - b);
        return align_node;
    }

    PCX_AINLINE static void perform(cxpack_for<T> auto          dst_pck,
                                    cxpack_for<T> auto          src_pck,
                                    meta::ce_of<bool> auto      lowk,
                                    meta::ce_of<bool> auto      half_tw,
                                    meta::ce_of<bool> auto      reverse,
                                    meta::ce_of<bool> auto      conj_tw,
                                    uZ                          data_size,
                                    data_info_for<T> auto       dst_data,
                                    data_info_for<const T> auto src_data,
                                    meta::maybe_ce_of<uZ> auto  align_node,
                                    tw_data_for<T> auto&        tw) {
        [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            auto check_align = [&]<uZ I> PCX_LAINLINE(uZ_ce<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != align_node)
                    return false;

                constexpr auto align = align_param<l_node_size, true>{};
                perform_impl(dst_pck,
                             src_pck,
                             align,
                             lowk,
                             half_tw,
                             reverse,
                             conj_tw,
                             data_size,
                             dst_data,
                             src_data,
                             tw);
                return true;
            };
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<log2i(NodeSize)>{});
    }

    PCX_AINLINE static void perform_impl(cxpack_for<T> auto          dst_pck,
                                         cxpack_for<T> auto          src_pck,
                                         any_align auto              align,
                                         meta::ce_of<bool> auto      lowk,
                                         meta::ce_of<bool> auto      half_tw,
                                         meta::ce_of<bool> auto      reverse,
                                         meta::ce_of<bool> auto      conj_tw,
                                         meta::maybe_ce_of<uZ> auto  data_size,
                                         data_info_for<T> auto       dst_data,
                                         data_info_for<const T> auto src_data,
                                         tw_data_for<T> auto&        tw_data) {
        static_assert(dst_data.contiguous());
        static_assert(src_data.empty() || src_data.contiguous());

        constexpr auto single_load_size = node_size * width;
        const auto     local_tw         = tw_data.is_local();
        using tw_t = std::conditional_t<local_tw, tw_data_t<T, local_tw>, tw_data_t<T, local_tw>&>;
        tw_t tw    = tw_data;

        using fnode        = subtransform<node_size, T, width>;
        auto final_k_count = data_size / single_load_size / 2;

        auto multi_load = [&] PCX_LAINLINE(auto dst_pck, auto src_pck, auto src) {
            constexpr auto batch_size = width;
            const auto     batch_cnt  = [=] {
                if constexpr (meta::ce_of<decltype(data_size), uZ>) {
                    return uZ_ce<data_size / batch_size>{};
                } else {
                    return data_size / batch_size;
                }
            }();
            fnode::perform(dst_pck,
                           src_pck,
                           align,
                           lowk,
                           reverse,
                           conj_tw,
                           batch_cnt,
                           batch_size,
                           dst_data,
                           src,
                           final_k_count,
                           tw);
        };
        auto l_single_load = [&] PCX_LAINLINE(auto dst_pck, auto src_pck, auto lowk, auto dst, auto src) {
            single_load(dst_pck, src_pck, lowk, half_tw, conj_tw, reverse, dst, src, tw);
        };

        if constexpr (!reverse) {
            multi_load(w_pck, src_pck, src_data);

            if constexpr (skip_single_load) {
                for (auto i: stdv::iota(0U, data_size / width)) {
                    auto ptr = dst_data.get_batch_base(i);
                    auto rd  = (simd::cxload<width, width> | simd::repack<1>)(ptr);
                    simd::cxstore<1>(ptr, rd);
                }
                return;
            }
            if constexpr (local_tw) {
                tw.start_fft_size *= final_k_count * 2;
                tw.start_k *= final_k_count * 2;
            }
            auto data_ptr = dst_data.get_batch_base(0);
            if constexpr (lowk && !skip_lowk_single_load)
                l_single_load(dst_pck, w_pck, lowk, data_ptr, data_ptr);
            constexpr auto start = lowk ? 1UZ : 0UZ;
            for (auto k: stdv::iota(start, final_k_count * 2)) {
                auto dest = data_ptr + k * single_load_size * 2;
                l_single_load(dst_pck, w_pck, not_lowk, dest, dest);
            }
        } else {
            if constexpr (skip_single_load) {
                for (auto i: stdv::iota(0U, data_size / width)) {
                    auto ptr = dst_data.get_batch_base(i * width);
                    auto rd  = (simd::cxload<1, width> | simd::repack<width>)(ptr);
                    simd::cxstore<width>(ptr, rd);
                }
            } else {
                if constexpr (local_tw) {
                    tw.start_fft_size /= single_load_size;
                    tw.start_k /= single_load_size;
                }
                auto dst_ptr = dst_data.get_batch_base(0);
                auto src_ptr = [=] PCX_LAINLINE {
                    if constexpr (src_data.empty())
                        return dst_ptr;
                    else
                        return src_data.get_batch_base(0);
                }();

                auto k_start = lowk && !skip_lowk_single_load ? 1UZ : 0UZ;
                auto k_range = stdv::iota(k_start, final_k_count * 2) | stdv::reverse;
                for (auto k: k_range) {
                    auto src = src_ptr + k * single_load_size * 2;
                    auto dst = dst_ptr + k * single_load_size * 2;
                    l_single_load(w_pck, src_pck, not_lowk, dst, src);
                }
                if constexpr (lowk && !skip_lowk_single_load)
                    l_single_load(w_pck, src_pck, lowk, dst_ptr, src_ptr);
            }
            multi_load(dst_pck, w_pck, inplace_src);
        }
    };

    static void insert_tw(twiddle_range_for<T> auto& r,
                          any_align auto             align,
                          bool                       lowk,
                          uZ                         data_size,
                          tw_data_t<T, true>&        tw_data,
                          meta::ce_of<bool> auto     half_tw) {
        constexpr auto single_load_size = node_size * width;
        auto           final_k_count    = data_size / single_load_size / 2;
        using fnode                     = subtransform<node_size, T, width>;
        fnode::insert_tw(r, align, lowk, final_k_count, tw_data);
        if constexpr (skip_single_load)
            return;
        if (lowk && !skip_lowk_single_load)
            insert_single_load_tw(r, tw_data, lowk, half_tw);
        auto k_start = lowk && !skip_lowk_single_load ? 1UZ : 0UZ;
        for (auto k: stdv::iota(k_start, final_k_count * 2))
            insert_single_load_tw(r, tw_data, false, half_tw);
    }

    template<bool LocalTw>
    PCX_AINLINE static auto single_load(cxpack_for<T> auto     dst_pck,
                                        cxpack_for<T> auto     src_pck,
                                        meta::ce_of<bool> auto lowk,
                                        meta::ce_of<bool> auto half_tw,
                                        meta::ce_of<bool> auto conj_tw,
                                        meta::ce_of<bool> auto reverse,
                                        T* const               data_ptr,
                                        const T* const         src_ptr,
                                        tw_data_t<T, LocalTw>& tw_data) {
        if constexpr (LocalTw && reverse)
            --tw_data.start_k;
        auto regroup_tw_fact = [&]<uZ TwCount> PCX_LAINLINE(uZ_ce<TwCount>) {
            if constexpr (LocalTw) {
                return [=]<uZ KGroup> PCX_LAINLINE(uZ_ce<KGroup>) {
                    auto fft_size = tw_data.start_fft_size * node_size * TwCount;
                    auto k        = tw_data.start_k * node_size * TwCount / 2 + KGroup * TwCount;

                    constexpr auto adj_tw_count = half_tw && node_size == 2 ? TwCount / 2 : TwCount;

                    auto tw = [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                        if constexpr (half_tw) {
                            return std::array{wnk_br<T>(fft_size, k + Is * 2)...};
                        } else {
                            return std::array{wnk_br<T>(fft_size, k + Is)...};
                        }
                    }(make_uZ_seq<adj_tw_count>{});
                    auto twv = simd::cxload<1, adj_tw_count>(tw.data());
                    return simd::repack<adj_tw_count>(twv);
                };
            } else {
                constexpr auto adj_tw_count = half_tw && node_size == 2 ? TwCount / 2 : TwCount;
                if constexpr (reverse)
                    tw_data.tw_ptr -= TwCount * 2 * node_size / 2 / (half_tw ? 2 : 1);
                auto l_tw_ptr = tw_data.tw_ptr;
                if constexpr (!reverse)
                    tw_data.tw_ptr += TwCount * 2 * node_size / 2 / (half_tw ? 2 : 1);

                return [=]<uZ KGroup> PCX_LAINLINE(uZ_ce<KGroup>) {
                    constexpr uZ offset = (half_tw ? KGroup / 2 : KGroup) * adj_tw_count * 2;
                    return simd::cxload<adj_tw_count, adj_tw_count>(l_tw_ptr + offset);
                };
            }
        };
        auto get_tw = [&] PCX_LAINLINE {
            if constexpr (lowk) {
                if constexpr (!LocalTw && !skip_lowk_tw)
                    tw_data.tw_ptr += reverse ? -node_size : node_size;
                return typename btfly_node::low_k_tw_t{};
            } else if constexpr (LocalTw) {
                return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    auto tw = make_tw_node<T, node_size>(tw_data.start_fft_size * 2, tw_data.start_k);
                    return tupi::make_tuple(simd::cxbroadcast<1, width>(tw.data() + Is)...);
                }(make_uZ_seq<node_size / 2>{});
            } else {
                if constexpr (reverse)
                    tw_data.tw_ptr -= node_size;
                auto tw0 = [tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(simd::cxbroadcast<1, width>(tw_data.tw_ptr + 2 * Is)...);
                }(make_uZ_seq<node_size / 2>{});
                if constexpr (!reverse)
                    tw_data.tw_ptr += node_size;
                return tw0;
            }
        };

        auto data0 = [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            return tupi::make_tuple(simd::cxload<src_pck, width>(src_ptr + width * 2 * Is)...);
        }(make_uZ_seq<node_size>{});
        auto data = tupi::group_invoke(simd::repack<width>, data0);
        if constexpr (!reverse) {
            auto btfly_res_0 = btfly_node::forward(data, get_tw(), conj_tw);

            auto [data_lo, data_hi] = [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                auto lo = tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
                auto hi = tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
                return tupi::make_tuple(lo, hi);
            }(make_uZ_seq<node_size / 2>{});

            auto [lo, hi] = [=]<uZ NGroups = 2> PCX_LAINLINE    //
                (this auto f, auto data_lo, auto data_hi, uZ_ce<NGroups> = {}) {
                    if constexpr (width == 1) {
                        return tupi::make_tuple(data_lo, data_hi);
                    } else {
                        auto x = regroup_btfly<NGroups>(data_lo,
                                                        data_hi,
                                                        regroup_tw_fact(uZ_ce<NGroups>{}),
                                                        half_tw,
                                                        reverse,
                                                        conj_tw);
                        if constexpr (NGroups == width) {
                            return x;
                        } else {
                            auto [lo, hi] = x;
                            return f(lo, hi, uZ_ce<NGroups * 2>{});
                        }
                    }
                }(data_lo, data_hi);
            if constexpr (half_tw && node_size > 2) {
                // [ 0  4  8 12] [ 2  6 10 14] before
                // [ 0  2  4  6] [ 8 10 12 14]
                lo = regroup_half_tw(lo, reverse);
                hi = regroup_half_tw(hi, reverse);
            }
            auto btfly_res_1 = tupi::group_invoke(regroup<1, width>, lo, hi);
            auto res         = tupi::make_flat_tuple(btfly_res_1);
            auto res_rep     = tupi::group_invoke(simd::evaluate | simd::repack<dst_pck>, res);

            [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                (simd::cxstore<dst_pck>(data_ptr + Width * 2 * Is, get<Is>(res_rep)), ...);
            }(make_uZ_seq<node_size>{});
            if constexpr (LocalTw)
                ++tw_data.start_k;
        } else {
            auto [data_lo, data_hi] = [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                auto lo = tupi::make_tuple(get<Is * 2>(data)...);
                auto hi = tupi::make_tuple(get<Is * 2 + 1>(data)...);
                return tupi::make_tuple(lo, hi);
            }(make_uZ_seq<node_size / 2>{});
            auto data_rg = tupi::group_invoke(regroup<width, 1>, data_lo, data_hi);
            auto lo      = tupi::group_invoke(tupi::get_copy<0>, data_rg);
            auto hi      = tupi::group_invoke(tupi::get_copy<1>, data_rg);

            if constexpr (half_tw && node_size > 2) {
                // [ 0  4  8 12] [ 2  6 10 14] before
                // [ 0  2  4  6] [ 8 10 12 14]
                lo = regroup_half_tw(lo, reverse);
                hi = regroup_half_tw(hi, reverse);
            }
            auto [lo_1, hi_1] = [=]<uZ NGroups = width> PCX_LAINLINE    //
                (this auto f, auto data_lo, auto data_hi, uZ_ce<NGroups> = {}) {
                    if constexpr (width == 1) {
                        return tupi::make_tuple(data_lo, data_hi);
                    } else {
                        auto x = regroup_btfly<NGroups>(data_lo,
                                                        data_hi,
                                                        regroup_tw_fact(uZ_ce<NGroups>{}),
                                                        half_tw,
                                                        reverse,
                                                        conj_tw);
                        if constexpr (NGroups == 2) {
                            return x;
                        } else {
                            auto [lo, hi] = x;
                            return f(lo, hi, uZ_ce<NGroups / 2>{});
                        }
                    }
                }(lo, hi);

            auto data1 = tupi::make_flat_tuple(tupi::group_invoke(tupi::make_tuple, lo_1, hi_1));

            auto data0    = btfly_node::reverse(data1, get_tw(), conj_tw);
            auto data_rep = tupi::group_invoke(simd::repack<dst_pck>, data0);

            [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                (simd::cxstore<dst_pck>(data_ptr + Width * 2 * Is, get<Is>(data_rep)), ...);
            }(make_uZ_seq<node_size>{});
        }
    }
    static void insert_single_load_tw(twiddle_range_for<T> auto& r,
                                      tw_data_t<T, true>&        tw_data,
                                      bool                       lowk,
                                      meta::ce_of<bool> auto     half_tw) {
        auto insert = [&r](auto tws) {
            for (auto tw: tws) {
                r.push_back(tw.real());
                r.push_back(tw.imag());
            }
        };
        if (!lowk || !skip_lowk_tw) {
            auto tws = make_tw_node<T, NodeSize>(tw_data.start_fft_size * 2,    //
                                                 tw_data.start_k);
            insert(tws);
        }
        if constexpr (skip_sub_width) {
            ++tw_data.start_k;
            return;
        }

        auto regroup_tw_fact = [&]<uZ TwCount>(uZ_ce<TwCount>) {
            return [=]<uZ KGroup>(uZ_ce<KGroup>) {
                auto fft_size = tw_data.start_fft_size * NodeSize * TwCount;
                auto k        = tw_data.start_k * NodeSize * TwCount / 2 + KGroup * TwCount;

                constexpr auto adj_tw_count = half_tw && node_size == 2 ? TwCount / 2 : TwCount;

                auto tw_arr = [=]<uZ... Is>(uZ_seq<Is...>) {
                    if constexpr (half_tw) {
                        return std::array{wnk_br<T>(fft_size, k + Is * 2)...};
                    } else {
                        return std::array{wnk_br<T>(fft_size, k + Is)...};
                    }
                }(make_uZ_seq<adj_tw_count>{});
                if constexpr (adj_tw_count > 1) {
                    auto data = simd::repack<adj_tw_count>(simd::cxload<1, adj_tw_count>(tw_arr.data()));
                    simd::cxstore<adj_tw_count>(reinterpret_cast<T*>(tw_arr.data()), data);
                }
                insert(tw_arr);
            };
        };

        auto insert_regroup_tw = [](auto insert_tw, auto half_tw) {
            constexpr uZ raw_tw_count = half_tw && node_size > 2 ? node_size / 4 : node_size / 2;
            [=]<uZ... Is>(uZ_seq<Is...>) {
                (insert_tw(uZ_ce<half_tw ? Is * 2 : Is>{}), ...);
            }(make_uZ_seq<raw_tw_count>{});
        };

        [=]<uZ NGroups = 2> PCX_LAINLINE    //
            (this auto f, uZ_ce<NGroups> = {}) {
                if constexpr (NGroups >= Width) {
                    insert_regroup_tw(regroup_tw_fact(uZ_ce<NGroups>{}), half_tw);
                } else {
                    insert_regroup_tw(regroup_tw_fact(uZ_ce<NGroups>{}), half_tw);
                    f(uZ_ce<NGroups * 2>{});
                }
            }();
        ++tw_data.start_k;
    }

    static constexpr struct {
        // lo: [0 0 1 1] [4 4 5 5] hi: [2 2 3 3] [6 6 7 7]
        // std::tie(lo, hi) = switch_1_2(lo, hi);
        // lo: [0 0 1 1] [2 2 3 3] hi: [4 4 5 5] [6 6 7 7]

        // after split-ileave:
        // lo: [0 0 4 4] [2 2 6 6] hi: [1 1 5 5] [3 3 7 7]
        // corresponding twiddles:
        // [  1   1   5   5] [        3         3         7         7]
        // [tw0 tw0 tw1 tw1] [conj(tw0) conj(tw0) conj(tw1) conj(tw1)]
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...> lo, tupi::tuple<Thi...> hi) {
            return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                auto new_lo = tupi::tuple_cat(tupi::make_tuple(tupi::get<Is * 2>(lo),    //
                                                               tupi::get<Is * 2>(hi))...);
                auto new_hi = tupi::tuple_cat(tupi::make_tuple(tupi::get<Is * 2 + 1>(lo),    //
                                                               tupi::get<Is * 2 + 1>(hi))...);
                return tupi::make_tuple(new_lo, new_hi);
            }(make_uZ_seq<sizeof...(Tlo) / 2>{});
        }
    } switch_1_2{};

    static constexpr struct {
        template<simd::any_cx_vec... Ts>
            requires(sizeof...(Ts) > 1)
        PCX_AINLINE static auto operator()(tupi::tuple<Ts...> vecs) {
            return [&]<uZ... IPairs> PCX_LAINLINE(uZ_seq<IPairs...>) {
                return tupi::tuple_cat(regroup<1, width>(tupi::get<IPairs * 2>(vecs),    //
                                                         tupi::get<IPairs * 2 + 1>(vecs))...);
            }(make_uZ_seq<sizeof...(Ts) / 2>{});
        }
        template<simd::any_cx_vec... Ts>
            requires(sizeof...(Ts) > 1)
        PCX_AINLINE static auto operator()(tupi::tuple<Ts...> vecs, meta::ce_of<bool> auto reverse) {
            return [&]<uZ... IPairs> PCX_LAINLINE(uZ_seq<IPairs...>) {
                if constexpr (!reverse) {
                    return tupi::tuple_cat(regroup<1, width>(tupi::get<IPairs * 2>(vecs),    //
                                                             tupi::get<IPairs * 2 + 1>(vecs))...);
                } else {
                    return tupi::tuple_cat(regroup<width, 1>(tupi::get<IPairs * 2>(vecs),    //
                                                             tupi::get<IPairs * 2 + 1>(vecs))...);
                }
            }(make_uZ_seq<sizeof...(Ts) / 2>{});
        }
    } regroup_half_tw{};


    // clang-format off
    /**
     * @brief Loads and upsamples `Count` twiddles.
     */
    template<uZ Count>
    static constexpr auto load_tw =
        tupi::pass    
        | []<uZ KGroup>(auto&& get_tw, uZ_ce<KGroup> k) {
            return get_tw(k);
          }
        | tupi::group_invoke([](auto v){
                return vec_traits::upsample(v.value);
          })    
        | tupi::apply                                                  
        | [](auto re, auto im) { return simd::cx_vec<T, false, false, Width>{re, im}; };


    /**
     * @brief Regroups input simd vectors, similar to `simd::repack`, except
     * that the real and imaginary part are processed independently.
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
     *  @return `tupi::tuple<>{even, odd}` interleaved even/odd chunks.
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

    /**
     *  @brief Split-regroups input data, loads twiddles and performs a single butterfly operation.
     *  see `split_regroup<>`. 
     *  
     *  @tparam NGroups - number of fft groups (`k`) that fit in a single simd vector.
     */
    template<uZ NGroups>
    struct regroup_btfly_t {
        static constexpr auto make_j_tuple = [] {
            if constexpr (node_size > 2) {
                return tupi::pass |
                       [](simd::any_cx_vec auto tw) { return tupi::make_tuple(tw, mul_by_j<-1>(tw)); };
            } else {
                return tupi::pass    //
                       | [](simd::any_cx_vec auto tw) { return tupi::make_tuple(tw, mul_by_j<-1>(tw)); }
                       | tupi::pipeline(tupi::pass, simd::evaluate)    //
                       | tupi::apply                                   //
                       | split_regroup<width / NGroups>                //
                       | tupi::get_copy<0>;
            }
        }();

        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...>    lo,
                                           tupi::tuple<Thi...>    hi,
                                           auto&&                 get_tw,
                                           meta::ce_of<bool> auto half_tw) {
            constexpr uZ   raw_tw_count = half_tw && node_size > 2 ? node_size / 4 : node_size / 2;
            constexpr auto tw_idx_tup   = [=]<uZ... Is>(uZ_seq<Is...>) {
                return tupi::make_tuple(uZ_ce<half_tw ? Is * 2 : Is>{}...);
            }(make_uZ_seq<raw_tw_count>{});

            auto get_tw_tup = tupi::make_broadcast_tuple<raw_tw_count>(get_tw);

            constexpr auto ltw = [=] {
                if constexpr (half_tw) {
                    return tupi::apply                                              //
                           | tupi::group_invoke(load_tw<NGroups> | make_j_tuple)    //
                           | tupi::make_flat_tuple;
                } else {
                    return tupi::apply | tupi::group_invoke(load_tw<NGroups>);
                }
            }();

            constexpr auto regr_ltw =
                tupi::make_tuple
                | tupi::pipeline(tupi::apply | tupi::group_invoke(split_regroup<width / NGroups>),    //
                                 ltw);

            if constexpr (half_tw && node_size > 2)
                std::tie(lo, hi) = switch_1_2(lo, hi);
            auto [regrouped, tw] = regr_ltw(tupi::forward_as_tuple(lo, hi),    //
                                            tupi::forward_as_tuple(get_tw_tup, tw_idx_tup));

            auto lo_re  = tupi::group_invoke(tupi::get<0>, regrouped);
            auto hi_re  = tupi::group_invoke(tupi::get<1>, regrouped);
            auto hi_tw  = tupi::group_invoke(simd::mul, hi_re, tw);
            auto res    = tupi::group_invoke(simd::btfly, lo_re, hi_tw);
            auto new_lo = tupi::group_invoke(tupi::get_copy<0>, res);
            auto new_hi = tupi::group_invoke(tupi::get_copy<1>, res);
            return tupi::make_tuple(new_lo, new_hi);
        }
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...>    lo,
                                           tupi::tuple<Thi...>    hi,
                                           auto&&                 get_tw,
                                           meta::ce_of<bool> auto half_tw,
                                           meta::ce_of<bool> auto reverse,
                                           meta::ce_of<bool> auto conj_tw) {
            constexpr uZ   raw_tw_count = half_tw && node_size > 2 ? node_size / 4 : node_size / 2;
            constexpr auto tw_idx_tup   = [=]<uZ... Is>(uZ_seq<Is...>) {
                return tupi::make_tuple(uZ_ce<half_tw ? Is * 2 : Is>{}...);
            }(make_uZ_seq<raw_tw_count>{});

            auto get_tw_tup = tupi::make_broadcast_tuple<raw_tw_count>(get_tw);

            if constexpr (!reverse) {
                constexpr auto ltw = [=] {
                    if constexpr (half_tw) {
                        return tupi::apply                                              //
                               | tupi::group_invoke(load_tw<NGroups> | make_j_tuple)    //
                               | tupi::make_flat_tuple;
                    } else {
                        return tupi::apply | tupi::group_invoke(load_tw<NGroups>);
                    }
                }();

                constexpr auto regr_ltw =
                    tupi::make_tuple
                    | tupi::pipeline(tupi::apply | tupi::group_invoke(split_regroup<width / NGroups>),    //
                                     ltw);

                if constexpr (half_tw && node_size > 2)
                    std::tie(lo, hi) = switch_1_2(lo, hi);
                auto [regrouped, tw] = regr_ltw(tupi::forward_as_tuple(lo, hi),    //
                                                tupi::forward_as_tuple(get_tw_tup, tw_idx_tup));

                auto lo_re  = tupi::group_invoke(tupi::get<0>, regrouped);
                auto hi_re  = tupi::group_invoke(tupi::get<1>, regrouped);
                auto hi_tw  = tupi::group_invoke(simd::mul, hi_re, tw);
                auto res    = tupi::group_invoke(simd::btfly, lo_re, hi_tw);
                auto new_lo = tupi::group_invoke(tupi::get_copy<0>, res);
                auto new_hi = tupi::group_invoke(tupi::get_copy<1>, res);
                return tupi::make_tuple(new_lo, new_hi);
            } else {
                constexpr auto ltw = [=] {
                    if constexpr (half_tw) {
                        return tupi::group_invoke(load_tw<NGroups> | make_j_tuple)    //
                               | tupi::make_flat_tuple;
                    } else {
                        return tupi::group_invoke(load_tw<NGroups>);
                    }
                }();

                auto tw        = ltw(get_tw_tup, tw_idx_tup);
                auto ctw       = tupi::group_invoke(simd::maybe_conj<conj_tw>, tw);
                auto res       = tupi::group_invoke(simd::btfly, lo, hi);
                auto new_lo    = tupi::group_invoke(tupi::get_copy<0>, res);
                auto new_hi    = tupi::group_invoke(tupi::get_copy<1>, res);
                auto hi_tw     = tupi::group_invoke(simd::mul, new_hi, ctw);
                auto regrouped = tupi::group_invoke(split_regroup<width / NGroups>, new_lo, hi_tw);
                auto lo_re     = tupi::group_invoke(tupi::get_copy<0>, regrouped);
                auto hi_re     = tupi::group_invoke(tupi::get_copy<1>, regrouped);
                if constexpr (half_tw && node_size > 2)
                    std::tie(lo_re, hi_re) = switch_1_2(lo_re, hi_re);
                return tupi::make_tuple(lo_re, hi_re);
            }
        }
    };
    template<uZ NGroups>
    constexpr static auto regroup_btfly = regroup_btfly_t<NGroups>{};
};

template<uZ NodeSize, typename T, uZ Width, uZ CohSize = 0, uZ LaneSize = 0>
struct transform {
    using seq_subtf_t = sequential_subtransform<NodeSize, T, Width>;
    // using subtf_t     = subtransform<NodeSize, T, Width>;

    /**
     * @brief Number of complex elements of type T that keep L1 cache coherency during subtransforms.
     */
    static constexpr auto coherent_size = uZ_ce<CohSize != 0 ? CohSize : 8194 / sizeof(T)>{};
    static constexpr auto lane_size = uZ_ce<std::max(LaneSize != 0 ? LaneSize : 64 / sizeof(T) / 2, Width)>{};

    static constexpr auto width     = uZ_ce<Width>{};
    static constexpr auto w_pck     = cxpack<width, T>{};
    static constexpr auto node_size = uZ_ce<NodeSize>{};

    static constexpr auto skip_seq_subtf = false;

    static constexpr auto logKi(meta::maybe_ce_of<uZ> auto k, u64 value) {
        uZ p = 0;
        while (value > k) {
            value /= k;
            ++p;
        }
        return p;
    }
    static constexpr auto get_align_node(uZ size) {
        auto slog       = log2i(size);
        auto a          = slog / log2i(node_size);
        auto b          = a * log2i(node_size);
        auto align_node = powi(2, slog - b);
        return align_node;
    };

    /**
     * Subdivides the data into buckets. Default bucket size is `coherent_size`.
     * A bucket is a set of `batches`, distributed with a constant stride. `Batch` is contiguous data with default size of `lane_size`.
     */
    PCX_AINLINE static void perform(cxpack_for<T> auto          dst_pck,
                                    cxpack_for<T> auto          src_pck,
                                    meta::ce_of<bool> auto      half_tw,
                                    meta::ce_of<bool> auto      lowk,
                                    data_info_for<T> auto       dst_data,
                                    data_info_for<const T> auto src_data,
                                    uZ                          fft_size,
                                    tw_data_for<T> auto         tw_data,
                                    auto                        permuter,
                                    uZ                          data_size = 0) {
        const auto bucket_size = coherent_size;
        const auto batch_size  = lane_size;
        const auto batch_cnt   = bucket_size / batch_size;
        const auto reverse     = std::false_type{};
        const auto conj_tw     = std::false_type{};
        const auto local_tw    = tw_data.is_local();
        const auto sequential  = dst_data.sequential();
        static_assert(src_data.sequential() == sequential || src_data.empty());

        const auto batch_align_seq = [=] {
            constexpr auto min_align = std::min(dst_pck.value, src_pck.value);
            constexpr auto pbegin    = log2i(min_align);
            constexpr auto pend      = log2i(batch_size);
            return []<uZ... Ps>(uZ_seq<Ps...>) {
                return std::index_sequence<powi(2, pend - 1 - Ps)...>{};
            }(std::make_index_sequence<pend - pbegin>{});
        }();
        constexpr auto bucket_tfsize = uZ_ce<bucket_size / (sequential ? 1 : batch_size)>{};
        constexpr auto batch_tfsize  = uZ_ce<sequential ? batch_size : 1>{};

        if (fft_size <= bucket_tfsize) {
            if constexpr (sequential) {
                dst_data        = dst_data.mul_stride(width);
                src_data        = src_data.mul_stride(width);
                auto align_node = seq_subtf_t::get_align_node(fft_size);
                [&]<uZ... Is>(uZ_seq<Is...>) {
                    auto check_align = [&]<uZ I>(uZ_ce<I>) {
                        constexpr auto l_node_size = powi(2, I);
                        if (l_node_size != align_node)
                            return false;
                        seq_subtf_t::perform(dst_pck,
                                             src_pck,
                                             lowk,
                                             half_tw,
                                             reverse,
                                             conj_tw,
                                             fft_size,
                                             dst_data,
                                             src_data,
                                             uZ_ce<l_node_size>{},
                                             tw_data);
                        return true;
                    };
                    (void)(check_align(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<log2i(node_size)>{});
                permuter.sequential_permute(dst_pck, dst_pck, dst_data, inplace_src);
            } else {
                auto batch_size = bucket_tfsize / fft_size * lane_size;
                auto batch_cnt  = fft_size;
                auto tform      = [=](auto width, auto align, auto batch_size, auto dst, auto src, auto tw) {
                    using subtf_t = subtransform<node_size, T, width>;
                    subtf_t::perform(dst_pck,
                                     src_pck,
                                     align,
                                     lowk,
                                     reverse,
                                     conj_tw,
                                     batch_cnt,
                                     batch_size,
                                     dst,
                                     src,
                                     fft_size / 2,
                                     tw);
                    auto(permuter)
                        .small_permute(width, batch_size, reverse, dst_pck, dst_pck, dst, inplace_src);
                };
                auto align_node = get_align_node(fft_size);
                [&]<uZ... Is>(uZ_seq<Is...>) {
                    auto check_align = [&]<uZ I>(uZ_ce<I>) {
                        constexpr auto l_node_size = powi(2, I);
                        if (l_node_size != align_node)
                            return false;
                        constexpr auto align = align_param<l_node_size, true>{};
                        while (true) {
                            if (data_size < batch_size) {
                                if (batch_size <= lane_size)
                                    break;
                                batch_size /= 2;
                                continue;
                            }
                            tform(width, align, batch_size, dst_data, src_data, tw_data);
                            data_size -= batch_size;
                            dst_data = dst_data.offset_contents(batch_size);
                            src_data = src_data.offset_contents(batch_size);
                        }
                        [&]<uZ... Batch> PCX_LAINLINE(uZ_seq<Batch...>) {
                            auto small_tform = [&](auto small_batch) {
                                if (data_size >= small_batch) {
                                    constexpr auto lwidth = uZ_ce<std::min(width.value, small_batch.value)>{};
                                    uZ             lw     = lwidth;
                                    tform(lwidth, align, small_batch, dst_data, src_data, tw_data);
                                    data_size -= small_batch;
                                    dst_data = dst_data.offset_contents(small_batch);
                                    src_data = src_data.offset_contents(small_batch);
                                }
                                return data_size != 0;
                            };
                            (void)(small_tform(uZ_ce<Batch>{}) && ...);
                        }(batch_align_seq);
                        return true;
                    };
                    (void)(check_align(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<log2i(node_size)>{});
            }
            return;
        }

        const auto pass_k_cnt            = bucket_size / batch_size / 2;
        const auto [pass_cnt, rem_k_cnt] = [=] {
            if constexpr (sequential) {
                uZ pass_cnt  = logKi(pass_k_cnt * 2, fft_size / bucket_size);
                uZ rem_k_cnt = fft_size / bucket_size / powi(pass_k_cnt * 2, pass_cnt) / 2;
                return tupi::make_tuple(pass_cnt, rem_k_cnt);
            } else {
                uZ pass_cnt  = logKi(pass_k_cnt * 2, fft_size) - 1;
                uZ rem_k_cnt = fft_size / powi(pass_k_cnt * 2, pass_cnt + 1) / 2;
                return tupi::make_tuple(pass_cnt, rem_k_cnt);
            }
        }();
        auto stride = batch_tfsize * fft_size / bucket_tfsize;

        dst_data = dst_data.mul_stride(stride);
        src_data = src_data.mul_stride(stride);

        auto tform = [=](auto width, auto batch_size, auto dst, auto src, auto tw_data) {
            constexpr auto w_pck = cxpack<width, T>{};

            uZ bucket_cnt     = fft_size / bucket_tfsize;
            uZ bucket_grp_cnt = 1;
            using subtf_t     = subtransform<node_size, T, width>;
            auto subtf        = [&](auto  dst_pck,
                             auto  src_pck,
                             auto  align,
                             auto  lowk,
                             auto  dst,
                             auto  src,
                             auto  k_cnt,
                             auto& tw) {
                subtf_t::perform(dst_pck,
                                 src_pck,
                                 align,
                                 lowk,
                                 reverse,
                                 conj_tw,
                                 batch_cnt,
                                 batch_size,
                                 dst,
                                 src,
                                 k_cnt,
                                 tw);
            };
            auto l_permuter    = permuter;
            auto id_perm       = identity_permuter;
            auto coherent_perm = [=](auto& permuter, auto pck, auto dst, auto src) {
                permuter.coherent_permute(width, batch_size, reverse, pck, pck, dst, src);
            };

            auto iterate_buckets = [&](auto  dst_pck,
                                       auto  src_pck,
                                       auto  align,
                                       auto  k_cnt,
                                       auto  src,
                                       auto& permuter) {
                auto l_tw_data  = tw_data;
                auto l_permuter = permuter;
                for (uZ i_b: stdv::iota(0U, bucket_cnt)) {
                    l_tw_data       = tw_data;
                    l_permuter      = permuter;
                    auto bucket_dst = dst.offset_k(i_b * batch_tfsize);
                    auto bucket_src = src.offset_k(i_b * batch_tfsize);
                    subtf(dst_pck, src_pck, align, lowk, bucket_dst, bucket_src, k_cnt, l_tw_data);
                    coherent_perm(l_permuter, dst_pck, bucket_dst, bucket_src);
                }
                tw_data  = l_tw_data;
                permuter = l_permuter;
                for (uZ i_bg: stdv::iota(1U, bucket_grp_cnt)) {
                    if constexpr (local_tw)
                        tw_data.start_k = i_bg;
                    auto bg_offset    = i_bg * bucket_cnt * bucket_tfsize;
                    auto bg_dst_start = dst.offset_k(bg_offset);
                    auto bg_src_start = src.offset_k(bg_offset);
                    for (uZ i_b: stdv::iota(0U, bucket_cnt)) {
                        l_tw_data       = tw_data;
                        l_permuter      = permuter;
                        auto bucket_dst = bg_dst_start.offset_k(i_b * batch_tfsize);
                        auto bucket_src = bg_src_start.offset_k(i_b * batch_tfsize);
                        subtf(dst_pck, src_pck, align, not_lowk, bucket_dst, bucket_src, k_cnt, l_tw_data);
                        coherent_perm(l_permuter, dst_pck, bucket_dst, bucket_src);
                    }
                    tw_data  = l_tw_data;
                    permuter = l_permuter;
                }
                bucket_cnt /= k_cnt * 2;
                bucket_grp_cnt *= k_cnt * 2;
                dst = dst.div_stride(k_cnt * 2);
                if constexpr (local_tw) {
                    tw_data.start_k = 0;
                    tw_data.start_fft_size *= k_cnt * 2;
                }
            };

            auto pre_pass_align_node = get_align_node(rem_k_cnt * 2);
            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = powi(2, I);
                    if (l_node_size != pre_pass_align_node)
                        return false;
                    constexpr auto align = align_param<l_node_size, true>{};
                    iterate_buckets(w_pck, src_pck, align, rem_k_cnt, src, id_perm);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<log2i(node_size)>{});

            auto           tw_data_bak     = tw_data;
            constexpr auto pass_align_node = align_param<get_align_node(pass_k_cnt * 2), true>{};
            for (uZ pass: stdv::iota(0U, pass_cnt)) {
                if constexpr (!local_tw)
                    tw_data = tw_data_bak;
                iterate_buckets(w_pck, w_pck, pass_align_node, pass_k_cnt, inplace_src, id_perm);
            }

            if constexpr (!sequential) {
                if constexpr (!local_tw)
                    tw_data = tw_data_bak;
                iterate_buckets(dst_pck, w_pck, pass_align_node, pass_k_cnt, inplace_src, l_permuter);
                dst = dst.reset_stride();
                l_permuter.permute(width, batch_size, reverse, dst_pck, dst_pck, dst, inplace_src);
            } else {
                if constexpr (width < batch_size)
                    dst = dst.div_stride(dst.stride / width);
                if constexpr (skip_seq_subtf) {
                    for (auto i: stdv::iota(0U, fft_size / width)) {
                        auto ptr = dst.get_batch_base(i);
                        auto rd  = (simd::cxload<width, width> | simd::repack<1>)(ptr);
                        simd::cxstore<1>(ptr, rd);
                    }
                    return;
                }
                auto seq = [&] PCX_LAINLINE(auto lowk, auto offset) {
                    constexpr auto sequential_align_node =
                        uZ_ce<seq_subtf_t::get_align_node(bucket_tfsize)>{};
                    uZ x = sequential_align_node;
                    seq_subtf_t::perform(dst_pck,
                                         w_pck,
                                         lowk,
                                         half_tw,
                                         reverse,
                                         conj_tw,
                                         bucket_size,
                                         dst.offset_k(offset),
                                         inplace_src,
                                         sequential_align_node,
                                         tw_data);
                };
                if constexpr (lowk)
                    seq(lowk, 0);
                constexpr uZ sequential_start = lowk ? 1U : 0U;
                for (uZ i_bg: stdv::iota(sequential_start, bucket_grp_cnt)) {
                    if constexpr (local_tw)
                        tw_data.start_k = i_bg;
                    auto bucket_offset = i_bg * bucket_tfsize;
                    seq(not_lowk, bucket_offset);
                }
                auto(permuter).sequential_permute(dst_pck, dst_pck, dst, inplace_src);
            }
        };

        if constexpr (sequential) {
            tform(width, batch_size, dst_data, src_data, tw_data);
        } else {
            while (data_size >= batch_size) {
                tform(width, batch_size, dst_data, src_data, tw_data);
                data_size -= batch_size;
                dst_data = dst_data.offset_contents(batch_size);
                src_data = src_data.offset_contents(batch_size);
            }
            [&]<uZ... Batch> PCX_LAINLINE(uZ_seq<Batch...>) {
                auto small_tform = [&](auto small_batch) {
                    if (data_size >= small_batch) {
                        constexpr auto lwidth = uZ_ce<std::min(width.value, small_batch.value)>{};
                        tform(lwidth, small_batch, dst_data, src_data, tw_data);
                        data_size -= small_batch;
                        dst_data = dst_data.offset_contents(small_batch);
                        src_data = src_data.offset_contents(small_batch);
                    }
                    return data_size != 0;
                };
                (void)(small_tform(uZ_ce<Batch>{}) && ...);
            }(batch_align_seq);
        }
    };

    PCX_AINLINE static void perform_rev(cxpack_for<T> auto          dst_pck,
                                        cxpack_for<T> auto          src_pck,
                                        meta::ce_of<bool> auto      half_tw,
                                        meta::ce_of<bool> auto      lowk,
                                        data_info_for<T> auto       dst_data,
                                        data_info_for<const T> auto src_data,
                                        uZ                          fft_size,
                                        tw_data_for<T> auto         tw_data,
                                        auto                        permuter,
                                        uZ                          data_size = 1) {
        const auto bucket_size = coherent_size;
        const auto batch_size  = lane_size;
        const auto batch_cnt   = bucket_size / batch_size;
        const auto reverse     = std::true_type{};
        const auto conj_tw     = std::true_type{};
        const auto local_tw    = tw_data.is_local();
        const auto sequential  = dst_data.sequential();
        static_assert(src_data.sequential() == sequential || src_data.empty());

        const auto batch_align_seq = [=] {
            constexpr auto min_align = std::min(dst_pck.value, src_pck.value);
            constexpr auto pbegin    = log2i(min_align);
            constexpr auto pend      = log2i(batch_size);
            return []<uZ... Ps>(uZ_seq<Ps...>) {
                return std::index_sequence<powi(2, pend - 1 - Ps)...>{};
            }(std::make_index_sequence<pend - pbegin>{});
        }();
        constexpr auto bucket_tfsize = uZ_ce<bucket_size / (sequential ? 1 : batch_size)>{};
        constexpr auto batch_tfsize  = uZ_ce<sequential ? batch_size : 1>{};

        if (fft_size <= bucket_tfsize) {
            if constexpr (sequential) {
                auto o_src = permuter.sequential_permute(src_pck, src_pck, dst_data, src_data);
                dst_data   = dst_data.mul_stride(width);
                o_src      = o_src.mul_stride(width);

                auto sequential_align_node = seq_subtf_t::get_align_node(fft_size);
                [&]<uZ... Is>(uZ_seq<Is...>) {
                    auto check_align = [&]<uZ I>(uZ_ce<I>) {
                        constexpr auto l_node_size = powi(2, I);
                        if (l_node_size != sequential_align_node)
                            return false;
                        seq_subtf_t::perform(dst_pck,
                                             src_pck,
                                             lowk,
                                             half_tw,
                                             reverse,
                                             conj_tw,
                                             fft_size,
                                             dst_data,
                                             o_src,
                                             uZ_ce<l_node_size>{},
                                             tw_data);
                        return true;
                    };
                    (void)(check_align(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<log2i(node_size)>{});
            } else {
                auto batch_size = bucket_tfsize / fft_size * lane_size;
                auto batch_cnt  = fft_size;
                auto tform      = [=](auto width, auto align, auto batch_size, auto dst, auto src, auto tw) {
                    auto o_src =
                        auto(permuter).small_permute(width, batch_size, reverse, src_pck, src_pck, dst, src);
                    using subtf_t = subtransform<node_size, T, width>;
                    subtf_t::perform(dst_pck,
                                     src_pck,
                                     align,
                                     lowk,
                                     reverse,
                                     conj_tw,
                                     batch_cnt,
                                     batch_size,
                                     dst,
                                     o_src,
                                     fft_size / 2,
                                     tw);
                };
                auto align_node = get_align_node(fft_size);
                [&]<uZ... Is>(uZ_seq<Is...>) {
                    auto check_align = [&]<uZ I>(uZ_ce<I>) {
                        constexpr auto l_node_size = powi(2, I);
                        if (l_node_size != align_node)
                            return false;
                        constexpr auto align = align_param<l_node_size, true>{};
                        while (true) {
                            if (data_size < batch_size) {
                                if (batch_size <= lane_size)
                                    break;
                                batch_size /= 2;
                                continue;
                            }
                            tform(width, align, batch_size, dst_data, src_data, tw_data);
                            data_size -= batch_size;
                            dst_data = dst_data.offset_contents(batch_size);
                            src_data = src_data.offset_contents(batch_size);
                        }
                        [&]<uZ... Batch> PCX_LAINLINE(uZ_seq<Batch...>) {
                            auto small_tform = [&](auto small_batch) {
                                if (data_size >= small_batch) {
                                    constexpr auto lwidth = uZ_ce<std::min(width.value, small_batch.value)>{};
                                    uZ             lw     = lwidth;
                                    tform(lwidth, align, small_batch, dst_data, src_data, tw_data);
                                    data_size -= small_batch;
                                    dst_data = dst_data.offset_contents(small_batch);
                                    src_data = src_data.offset_contents(small_batch);
                                }
                                return data_size != 0;
                            };
                            (void)(small_tform(uZ_ce<Batch>{}) && ...);
                        }(batch_align_seq);
                        return true;
                    };
                    (void)(check_align(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<log2i(node_size)>{});
            }
            return;
        }
        const auto pass_k_cnt            = bucket_size / batch_size / 2;
        const auto [pass_cnt, rem_k_cnt] = [=] {
            if constexpr (sequential) {
                uZ pass_cnt  = logKi(pass_k_cnt * 2, fft_size / bucket_size);
                uZ rem_k_cnt = fft_size / bucket_size / powi(pass_k_cnt * 2, pass_cnt) / 2;
                return tupi::make_tuple(pass_cnt, rem_k_cnt);
            } else {
                uZ pass_cnt  = logKi(pass_k_cnt * 2, fft_size) - 1;
                uZ rem_k_cnt = fft_size / powi(pass_k_cnt * 2, pass_cnt + 1) / 2;
                return tupi::make_tuple(pass_cnt, rem_k_cnt);
            }
        }();

        auto tform = [=](auto width, auto batch_size, auto dst, auto src, auto tw_data, auto permuter) {
            constexpr auto w_pck = cxpack<width, T>{};

            uZ bucket_cnt       = 1;    // per bucket group
            uZ bucket_group_cnt = fft_size / bucket_tfsize;

            auto subtf = [&](auto  dst_pck,
                             auto  src_pck,
                             auto  align,
                             auto  lowk,
                             auto  dst,
                             auto  src,
                             auto  k_cnt,
                             auto& tw) {
                using subtf_t = subtransform<node_size, T, width>;
                subtf_t::perform(dst_pck,
                                 src_pck,
                                 align,
                                 lowk,
                                 reverse,
                                 conj_tw,
                                 batch_cnt,
                                 batch_size,
                                 dst,
                                 src,
                                 k_cnt,
                                 tw);
            };
            auto tw_data_bak   = tw_data;
            auto coherent_perm = [=](auto& permuter, auto pck, auto dst, auto src) {
                return permuter.coherent_permute(width, batch_size, reverse, pck, pck, dst, src);
            };
            auto iterate_buckets = [&](auto dst_pck,
                                       auto src_pck,
                                       auto align,
                                       auto k_cnt,
                                       auto src,
                                       auto permuter,
                                       bool first = false) {
                if (!first) {
                    dst = dst.mul_stride(k_cnt * 2);
                    src = src.mul_stride(k_cnt * 2);
                    bucket_cnt *= k_cnt * 2;
                    bucket_group_cnt /= k_cnt * 2;
                }
                auto l_tw_data  = tw_data;
                auto l_permuter = permuter;
                for (uZ i_bg: stdv::iota(0U, bucket_group_cnt) | stdv::drop(1) | stdv::reverse) {
                    if constexpr (local_tw)
                        tw_data.start_k = i_bg * k_cnt * 2;
                    if (i_bg == bucket_group_cnt / (k_cnt * 2) - 1)
                        tw_data_bak = tw_data;
                    auto bg_offset    = i_bg * bucket_cnt * bucket_tfsize;
                    auto bg_dst_start = dst.offset_k(bg_offset);
                    auto bg_src_start = src.offset_k(bg_offset);
                    for (uZ i_b: stdv::iota(0U, bucket_cnt)) {
                        l_tw_data       = tw_data;
                        l_permuter      = permuter;
                        auto bucket_dst = bg_dst_start.offset_k(i_b * batch_tfsize);
                        auto bucket_src = bg_src_start.offset_k(i_b * batch_tfsize);
                        auto s_src      = coherent_perm(l_permuter, dst_pck, bucket_dst, bucket_src);
                        subtf(dst_pck, src_pck, align, not_lowk, bucket_dst, s_src, k_cnt, l_tw_data);
                    }
                    tw_data  = l_tw_data;
                    permuter = l_permuter;
                }
                if constexpr (local_tw)
                    tw_data.start_k = 0;
                for (uZ i_b: stdv::iota(0U, bucket_cnt)) {
                    l_tw_data       = tw_data;
                    l_permuter      = permuter;
                    auto bucket_dst = dst.offset_k(i_b * batch_tfsize);
                    auto bucket_src = src.offset_k(i_b * batch_tfsize);
                    auto s_src      = coherent_perm(l_permuter, dst_pck, bucket_dst, bucket_src);
                    subtf(dst_pck, src_pck, align, lowk, bucket_dst, s_src, k_cnt, l_tw_data);
                }
                if constexpr (local_tw)
                    tw_data.start_fft_size /= k_cnt * 2;
                else
                    tw_data = l_tw_data;
                permuter = l_permuter;
            };

            constexpr auto pass_align_node = align_param<get_align_node(pass_k_cnt * 2), true>{};
            if constexpr (!sequential)
                iterate_buckets(w_pck, src_pck, pass_align_node, pass_k_cnt, src, permuter, true);
            for (uZ pass: stdv::iota(0U, pass_cnt)) {
                if constexpr (!local_tw)
                    tw_data = tw_data_bak;
                iterate_buckets(w_pck, w_pck, pass_align_node, pass_k_cnt, inplace_src, identity_permuter);
            }

            auto pre_pass_align_node = get_align_node(rem_k_cnt * 2);
            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = powi(2, I);
                    if (l_node_size != pre_pass_align_node)
                        return false;
                    constexpr auto align = align_param<l_node_size, true>{};
                    iterate_buckets(dst_pck, w_pck, align, rem_k_cnt, inplace_src, identity_permuter);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<log2i(node_size)>{});
        };
        if constexpr (sequential) {
            auto o_src = permuter.sequential_permute(src_pck, src_pck, dst_data, src_data);
            dst_data   = dst_data.mul_stride(width);
            o_src      = o_src.mul_stride(width);
            auto seq   = [&] PCX_LAINLINE(auto lowk, auto offset) {
                constexpr auto sequential_align_node = uZ_ce<seq_subtf_t::get_align_node(bucket_tfsize)>{};
                seq_subtf_t::perform(w_pck,
                                     src_pck,
                                     lowk,
                                     half_tw,
                                     reverse,
                                     conj_tw,
                                     bucket_size,
                                     dst_data.offset_k(offset),
                                     o_src.offset_k(offset),
                                     sequential_align_node,
                                     tw_data);
            };
            auto tw_data_bak = tw_data;
            if constexpr (skip_seq_subtf) {
                for (auto i: stdv::iota(0U, fft_size / width)) {
                    // auto src_ptr = src_data.get_batch_base(i);
                    auto dst_ptr = dst_data.get_batch_base(i);
                    auto rd      = (simd::cxload<1, width> | simd::repack<width>)(dst_ptr);
                    simd::cxstore<width>(dst_ptr, rd);
                }
                if constexpr (local_tw) {
                    tw_data.start_fft_size /= bucket_tfsize;
                    tw_data.start_k = 0;
                }
            } else {
                uZ   bucket_group_cnt = fft_size / bucket_tfsize;
                auto bg_range         = stdv::iota(lowk ? 1U : 0U, bucket_group_cnt) | stdv::reverse;
                for (uZ i_bg: bg_range) {
                    if constexpr (local_tw) {
                        tw_data         = tw_data_bak;
                        tw_data.start_k = bucket_tfsize + i_bg * bucket_tfsize;
                    }
                    auto bucket_offset = i_bg * bucket_tfsize;
                    seq(not_lowk, bucket_offset);
                }
                if constexpr (local_tw) {
                    tw_data         = tw_data_bak;
                    tw_data.start_k = bucket_tfsize;
                }
                if constexpr (lowk)
                    seq(lowk, 0);
            }
            dst_data = dst_data.mul_stride(batch_size / width);
            o_src    = o_src.mul_stride(batch_size / width);
            tform(width, batch_size, dst_data, o_src, tw_data, identity_permuter);
        } else {
            auto permute = [&](auto width, auto batch_size) {
                auto l_permuter = permuter;
                auto o_src =
                    l_permuter.permute(width, batch_size, reverse, src_pck, src_pck, dst_data, src_data);
                return tupi::make_tuple(o_src, l_permuter);
            };
            while (data_size >= batch_size) {
                auto [o_src, permuter] = permute(width, batch_size);
                tform(width, batch_size, dst_data, o_src, tw_data, permuter);
                data_size -= batch_size;
                dst_data = dst_data.offset_contents(batch_size);
                src_data = src_data.offset_contents(batch_size);
            }
            [&]<uZ... Batch> PCX_LAINLINE(uZ_seq<Batch...>) {
                auto small_tform = [&](auto small_batch) {
                    if (data_size >= small_batch) {
                        constexpr auto lwidth  = uZ_ce<std::min(width.value, small_batch.value)>{};
                        auto [o_src, permuter] = permute(lwidth, small_batch);
                        tform(lwidth, small_batch, dst_data, o_src, tw_data, permuter);
                        data_size -= small_batch;
                        dst_data = dst_data.offset_contents(small_batch);
                        src_data = src_data.offset_contents(small_batch);
                    }
                    return data_size != 0;
                };
                (void)(small_tform(uZ_ce<Batch>{}) && ...);
            }(batch_align_seq);
        }
    }

    static void insert_tw(twiddle_range_for<T> auto& r,    //
                          uZ                         fft_size,
                          bool                       lowk,
                          meta::ce_of<bool> auto     half_tw,
                          meta::ce_of<bool> auto     sequential) {
        const auto bucket_size  = coherent_size;
        const auto batch_size   = lane_size;
        const auto pass_k_count = bucket_size / batch_size / 2;

        constexpr auto bucket_tfsize = uZ_ce<bucket_size / (sequential ? 1 : batch_size)>{};
        constexpr auto batch_tfsize  = uZ_ce<sequential ? batch_size : 1>{};
        using seq_subtf_t            = sequential_subtransform<node_size, T, width>;
        using subtf_t                = subtransform<node_size, T, width>;

        if (fft_size <= bucket_tfsize) {
            auto l_tw_data = tw_data_t<T, true>{1, 0};
            uZ   align_node{};
            if constexpr (sequential)
                align_node = seq_subtf_t::get_align_node(fft_size);
            else
                align_node = get_align_node(fft_size);

            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = powi(2, I);
                    if (l_node_size != align_node)
                        return false;
                    constexpr auto align = align_param<l_node_size, true>{};
                    if constexpr (sequential)
                        seq_subtf_t::insert_tw(r, align, lowk, fft_size, l_tw_data, half_tw);
                    else
                        subtf_t::insert_tw(r, align, lowk, fft_size / 2, l_tw_data);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<log2i(node_size)>{});
            return;
        }

        auto pass_count       = logKi(pass_k_count * 2, fft_size / bucket_tfsize);
        uZ   pre_pass_k_count = fft_size / bucket_tfsize / powi(pass_k_count * 2, pass_count) / 2;
        auto iterate_buckets  = [&](meta::ce_of<uZ> auto align_node,    //
                                   uZ                   k_count,
                                   uZ                   bucket_group_count) {
            constexpr auto align = align_param<align_node, true>{};
            for (uZ i_bg: stdv::iota(0U, bucket_group_count)) {
                auto l_tw_data = tw_data_t<T, true>{bucket_group_count, i_bg};
                subtf_t::insert_tw(r, align, lowk && i_bg == 0, k_count, l_tw_data);
            }
        };

        auto pre_pass_align_node = get_align_node(pre_pass_k_count * 2);
        [&]<uZ... Is>(uZ_seq<Is...>) {
            auto check_align = [&]<uZ I>(uZ_ce<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != pre_pass_align_node)
                    return false;
                iterate_buckets(uZ_ce<l_node_size>{}, pre_pass_k_count, 1);
                return true;
            };
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<log2i(node_size)>{});

        constexpr auto pass_align_node = get_align_node(pass_k_count * 2);

        pass_count = sequential ? pass_count : pass_count + 1;
        if (pass_count > 0) {
            const auto start_k = pre_pass_k_count == 0 ? 1 : pre_pass_k_count;

            const auto final_bucket_group_count = start_k * 2 * powi(pass_k_count * 2, pass_count - 1);
            iterate_buckets(uZ_ce<pass_align_node>{}, pass_k_count, final_bucket_group_count);
        }
        if (sequential) {
            if constexpr (skip_seq_subtf)
                return;
            auto bucket_count = fft_size / bucket_size;

            constexpr auto seq_align = align_param<seq_subtf_t::get_align_node(bucket_size), true>{};
            for (uZ i_bg: stdv::iota(0U, bucket_count)) {
                auto l_tw_data = tw_data_t<T, true>{bucket_count, i_bg};
                seq_subtf_t::insert_tw(r, seq_align, lowk && i_bg == 0, bucket_size, l_tw_data, half_tw);
            }
            return;
        }
    };
};
}    // namespace pcx::detail_
