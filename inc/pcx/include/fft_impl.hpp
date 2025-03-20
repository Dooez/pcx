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
struct br_permute_t {
    template<eval_cx_vec... Vs>
        requires meta::equal_values<sizeof...(Vs), Vs::width()...> && meta::equal_values<Vs::pack_size()...>
    PCX_AINLINE static auto operator()(tupi::tuple<Vs...> data) {
        constexpr auto width = sizeof...(Vs);
        using data_t         = tupi::tuple<Vs...>;
        using cx_vec_t       = tupi::tuple_element_t<0, data_t>;
        using T              = cx_vec_t::real_type;
        using traits         = detail_::vec_traits<T, width>;
        constexpr auto pack  = cx_vec_t::pack_size();
        constexpr auto pass  = [=]<uZ Stride, uZ Chunk> PCX_LAINLINE(uZ_ce<Stride>, uZ_ce<Chunk>, auto data) {
            static_assert(Chunk != pack);
            constexpr auto splinter = [=] {
                if constexpr (Chunk < width) {
                    return tupi::pass |
                           [](auto v0, auto v1) {
                               return tupi::make_tuple(tupi::make_tuple(v0.real_v(), v1.real_v()),
                                                       tupi::make_tuple(v0.imag_v(), v1.imag_v()));
                           }
                           | tupi::group_invoke(tupi::apply | traits::template split_interleave<Chunk>)
                           | tupi::apply    //
                           | [](auto re, auto im) {
                                 return tupi::make_tuple(cx_vec_t(get<0>(re), get<0>(im)),
                                                         cx_vec_t(get<1>(re), get<1>(im)));
                             };
                } else {
                    return tupi::pass    //
                           | [](auto v0, auto v1) {
                                 return tupi::make_tuple(cx_vec_t(v0.real(), v1.real()),
                                                         cx_vec_t(v0.imag(), v1.imag()));
                             };
                }
            }();
            auto [lo, hi] = extract_halves<Stride>(data);
            auto res      = tupi::group_invoke(splinter, lo, hi);
            auto nlo      = tupi::group_invoke(tupi::get_copy<0>, res);
            auto nhi      = tupi::group_invoke(tupi::get_copy<1>, res);
            return combine_halves<Stride>(nlo, nhi);
        };
        return [pass]<uZ Stride = width, uZ Chunk = 1> PCX_LAINLINE(this auto     f,
                                                                    auto          l_data,
                                                                    uZ_ce<Stride> stride = {},
                                                                    uZ_ce<Chunk>  chunk  = {}) {
            if constexpr (chunk == pack) {
                return f(l_data, stride, uZ_ce<chunk * 2>{});
            } else if constexpr (stride == 2) {
                return pass(stride, chunk, l_data);
            } else {
                auto tmp = pass(stride, chunk, l_data);
                return f(tmp, uZ_ce<stride / 2>{}, uZ_ce<chunk * 2>{});
            }
        }(data);
    }
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
};    // namespace pcx::simd
inline constexpr auto br_permute = br_permute_t{};
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
    if (n == 0)
        throw std::runtime_error("N should be non-zero\n");

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

    using dest_t = tupi::broadcast_tuple_t<T*, NodeSize>;
    using data_t = tupi::broadcast_tuple_t<cx_vec, NodeSize>;
    using src_t  = tupi::broadcast_tuple_t<const T*, NodeSize>;
    using tw_t   = tupi::broadcast_tuple_t<cx_vec, NodeSize / 2>;
    struct low_k_tw_t {};

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
        auto data    = tupi::group_invoke(simd::cxload<S.pack_src, Width> | simd::repack<Width>, src);
        auto res     = S.reverse ? reverse(data, tw) : forward(data, tw);
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
        auto new_lo    = tupi::group_invoke(tupi::get_copy<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get_copy<1>, btfly_res);
        return combine_halves<stride>(new_lo, new_hi);
    };
    template<uZ Size, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto rbtfly_impl(uZ_ce<Size>, tupi::tuple<Ts...> data, auto tws) {
        constexpr auto stride = NodeSize / Size * 2;

        auto [lo, hi]  = extract_halves<stride>(data);
        auto btfly_res = tupi::group_invoke(simd::btfly, lo, hi);
        auto new_lo    = tupi::group_invoke(tupi::get_copy<0>, btfly_res);
        auto new_hi    = tupi::group_invoke(tupi::get_copy<1>, btfly_res);
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

    uZ start_fft_size;
    uZ start_k;
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
    PCX_AINLINE static void fft_iteration(cxpack_for<T> auto         dst_pck,
                                          cxpack_for<T> auto         src_pck,
                                          meta::any_ce_of<bool> auto lowk,
                                          meta::maybe_ce_of<uZ> auto bucket_size,
                                          meta::maybe_ce_of<uZ> auto stride,
                                          meta::maybe_ce_of<uZ> auto batch_size,
                                          auto                       data_ptr,
                                          uZ&                        k_count,
                                          tw_data_for<T> auto&       tw_data) {
        using btfly_node        = btfly_node_dit<NodeSizeL, T, width>;
        constexpr auto settings = val_ce<typename btfly_node::settings{
            .pack_dest = dst_pck,
            .pack_src  = src_pck,
            .reverse   = false,
        }>{};

        const auto local      = tw_data.is_local();
        using l_tw_data_t     = std::conditional_t<lowk && !local, tw_data_t<T, local>, tw_data_t<T, local>&>;
        l_tw_data_t l_tw_data = tw_data;

        const auto batch_count = bucket_size / k_count / batch_size;
        const auto k_stride    = stride * batch_count;

        // data division:
        // E - even, O - odd
        // [E0, E0,     ..., E0                  , O0, O0, ... , O<k_count - 1>, O<k_count - 1>, ...] subtform indexes
        // [i0, i1,     ..., i<batch_count/2 - 1>, i0, i1, ... , i0,             i1,             ...] batches
        // [0 , stride, ...                                                                         ]
        // iX is batch X
        //
        //
        // k is an index of O elements

        // data for a butterfly:
        // k == 0, i == 0
        // [0,     ... , NodeSizeL/4 - 1,          ... , NodeSizeL/2 - 1, ... ] - tuple index
        // [E0:i0, ... , E0:i<k_group_size/2 - 1>, ... , O1:i0,           ... ]
        //
        //
        auto make_data_tup = [=] PCX_LAINLINE(uZ i, uZ k, uZ offset) {
            auto base_ptr = data_ptr              //
                            + i * stride * 2      //
                            + k * k_stride * 2    //
                            + offset * 2;
            return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                return tupi::make_tuple((base_ptr + k_stride * 2 / NodeSizeL * Is)...);
            }(make_uZ_seq<NodeSizeL>{});
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
            } else {
                return [&l_tw_data] PCX_LAINLINE(uZ) {
                    return [&l_tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
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

            for (auto i_batch: stdv::iota(0U, batch_count / NodeSizeL)) {
                for (auto r: stdv::iota(0U, batch_size / width)) {
                    auto data = make_data_tup(i_batch, 0, r * width);
                    btfly_node::perform(settings, data);
                }
            }
        }
        constexpr auto k_start = lowk ? 1U : 0U;
        for (auto k_group: stdv::iota(k_start, k_count)) {
            auto tw = make_tw_tup(k_group);
            auto x  = 0;
            for (auto i_batch: stdv::iota(0U, batch_count / NodeSizeL)) {
                for (auto r: stdv::iota(0U, batch_size / width)) {
                    auto data = make_data_tup(i_batch, k_group, r * width);
                    btfly_node::perform(settings, data, tw);
                }
            }
        }
        k_count *= NodeSizeL;
        if constexpr (local) {
            l_tw_data.start_fft_size *= NodeSizeL;
            l_tw_data.start_k *= NodeSizeL;
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

        if (lowk && !skip_lowk_tw) {
            insert(0);
        }
        auto k_start = lowk ? 1U : 0U;
        for (auto k_group: stdv::iota(k_start, k_count)) {
            insert(k_group);
        }
        k_count *= NodeSizeL;
        tw_data.start_fft_size *= NodeSizeL;
        tw_data.start_k *= NodeSizeL;
    };

    PCX_AINLINE static void perform(cxpack_for<T> auto         src_pck,
                                    any_align auto             align,
                                    meta::any_ce_of<bool> auto lowk,
                                    meta::maybe_ce_of<uZ> auto bucket_size,
                                    meta::maybe_ce_of<uZ> auto stride,
                                    meta::maybe_ce_of<uZ> auto batch_size,
                                    uZ                         final_k_count,
                                    T*                         data_ptr,
                                    tw_data_for<T> auto&       tw_data) {
        uZ         k_count    = 1;
        const auto local      = tw_data.is_local();
        using l_tw_data_t     = std::conditional_t<local, tw_data_t<T, local>, tw_data_t<T, local>&>;
        l_tw_data_t l_tw_data = tw_data;

        auto fft_iter = [&]<uZ NodeSizeL> PCX_LAINLINE(uZ_ce<NodeSizeL>, auto dst_pck, auto src_pck) {
            fft_iteration<NodeSizeL>(dst_pck,
                                     src_pck,
                                     lowk,
                                     bucket_size,
                                     stride,
                                     batch_size,
                                     data_ptr,
                                     k_count,
                                     l_tw_data);
        };
        if constexpr (align.size_pre() != 1) {
            fft_iter(align.size_pre(), w_pck, src_pck);
            if constexpr (lowk && !local) {
                if constexpr (skip_lowk_tw) {
                    l_tw_data.tw_ptr += k_count - align.size_pre();
                } else {
                    l_tw_data.tw_ptr += k_count;
                }
            }
        } else {
            fft_iter(node_size, w_pck, src_pck);
        }

        while (k_count * align.size_post() <= final_k_count)
            fft_iter(node_size, w_pck, w_pck);

        if constexpr (lowk && !local) {
            if (k_count > align.size_pre()) {
                if constexpr (skip_lowk_tw) {
                    l_tw_data.tw_ptr += k_count - node_size;
                } else {
                    l_tw_data.tw_ptr += k_count;
                }
            }
        }
        if constexpr (align.size_post() != 1) {
            if (k_count > 1)
                fft_iter(align.size_post(), w_pck, w_pck);
            else {
                fft_iter(align.size_post(), w_pck, src_pck);
            }
            if constexpr (lowk && !local) {
                if constexpr (skip_lowk_tw) {
                    l_tw_data.tw_ptr += k_count - align.size_post();
                } else {
                    l_tw_data.tw_ptr += k_count;
                }
            }
        }
    }

    static void insert_tw(twiddle_range_for<T> auto& r,
                          any_align auto             align,
                          bool                       lowk,
                          uZ                         final_k_count,
                          tw_data_t<T, true>&        tw_data) {
        uZ k_count = 1;
        if constexpr (align.size_pre() != 1) {
            insert_iteration_tw<align.size_pre()>(r, tw_data, k_count, lowk);
        }
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
        if constexpr (align.size_post() != 1) {
            insert_iteration_tw<align.size_post()>(r, tw_data, k_count, lowk);
        }
    }
};

template<uZ NodeSize, typename T, uZ Width>
struct coherent_subtransform {
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

    PCX_AINLINE static void perform(cxpack_for<T> auto         dst_pck,
                                    cxpack_for<T> auto         src_pck,
                                    meta::any_ce_of<bool> auto lowk,
                                    meta::any_ce_of<bool> auto half_tw,
                                    uZ                         data_size,
                                    T*                         dest_ptr,
                                    meta::maybe_ce_of<uZ> auto align_node,
                                    tw_data_for<T> auto&       tw) {
        [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            auto check_align = [&]<uZ I> PCX_LAINLINE(uZ_ce<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != align_node)
                    return false;

                constexpr auto align = align_param<l_node_size, true>{};
                perform_impl(dst_pck, src_pck, align, lowk, half_tw, data_size, dest_ptr, tw);
                return true;
            };
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<log2i(NodeSize)>{});
    }

    PCX_AINLINE static void perform_impl(cxpack_for<T> auto         dst_pck,
                                         cxpack_for<T> auto         src_pck,
                                         any_align auto             align,
                                         meta::any_ce_of<bool> auto lowk,
                                         meta::any_ce_of<bool> auto half_tw,
                                         meta::maybe_ce_of<uZ> auto data_size,
                                         T*                         data_ptr,
                                         tw_data_for<T> auto&       tw_data) {
        constexpr auto single_load_size = node_size * width;
        const auto     local            = tw_data.is_local();

        using fnode        = subtransform<node_size, T, width>;
        auto final_k_count = data_size / single_load_size / 2;
        fnode::perform(src_pck, align, lowk, data_size, width, width, final_k_count, data_ptr, tw_data);

        if constexpr (skip_single_load) {
            for (auto i: stdv::iota(0U, data_size / width)) {
                auto ptr = data_ptr + i * width * 2;
                auto rd  = (simd::cxload<width, width> | simd::repack<1>)(ptr);
                simd::cxstore<1>(ptr, rd);
            }
            return;
        }
        if constexpr (local) {
            tw_data.start_fft_size *= final_k_count * 2;
            tw_data.start_k *= final_k_count * 2;
        }
        if constexpr (lowk && !skip_lowk_single_load) {
            single_load(dst_pck, w_pck, lowk, half_tw, data_ptr, data_ptr, tw_data);
        }
        constexpr auto start = lowk ? 1UZ : 0UZ;
        for (auto k: stdv::iota(start, final_k_count * 2)) {
            auto dest = data_ptr + k * single_load_size * 2;
            single_load(dst_pck, w_pck, std::false_type{}, half_tw, dest, dest, tw_data);
        }
    };

    static void insert_tw(twiddle_range_for<T> auto& r,
                          any_align auto             align,
                          bool                       lowk,
                          uZ                         data_size,
                          tw_data_t<T, true>&        tw_data,
                          meta::any_ce_of<bool> auto half_tw) {
        constexpr auto single_load_size = node_size * width;
        auto           final_k_count    = data_size / single_load_size / 2;
        using fnode                     = subtransform<node_size, T, width>;
        fnode::insert_tw(r, align, lowk, final_k_count, tw_data);
        if constexpr (skip_single_load) {
            return;
        }
        if (lowk && !skip_lowk_single_load) {
            insert_single_load_tw(r, tw_data, lowk, half_tw);
        }
        auto k_start = lowk && !skip_lowk_single_load ? 1UZ : 0UZ;
        for (auto k: stdv::iota(k_start, final_k_count * 2)) {
            insert_single_load_tw(r, tw_data, false, half_tw);
        }
    }

    template<bool LocalTw>
    PCX_AINLINE static auto single_load(cxpack_for<T> auto         dst_pck,
                                        cxpack_for<T> auto         src_pck,
                                        meta::any_ce_of<bool> auto lowk,
                                        meta::any_ce_of<bool> auto half_tw,
                                        T*                         data_ptr,
                                        const T*                   src_ptr,
                                        tw_data_t<T, LocalTw>&     tw_data) {
        auto data = [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            return tupi::make_tuple(simd::cxload<src_pck, width>(src_ptr + width * 2 * Is)...);
        }(make_uZ_seq<node_size>{});

        auto get_tw = [&] PCX_LAINLINE {
            if constexpr (LocalTw) {
                return [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    auto tw = make_tw_node<T, node_size>(tw_data.start_fft_size * 2, tw_data.start_k);
                    return tupi::make_tuple(simd::cxbroadcast<1, width>(tw.data() + Is)...);
                }(make_uZ_seq<node_size / 2>{});
            } else {
                auto tw0 = [tw_data]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(simd::cxbroadcast<1, width>(tw_data.tw_ptr + 2 * Is)...);
                }(make_uZ_seq<node_size / 2>{});
                tw_data.tw_ptr += node_size;
                return tw0;
            }
        };
        auto btfly_res_0 = [&] PCX_LAINLINE {
            if constexpr (lowk) {
                if constexpr (!skip_lowk_tw)
                    get_tw();
                return btfly_node::forward(data);
            } else {
                return btfly_node::forward(data, get_tw());
            }
        }();
        if constexpr (skip_sub_width) {
            auto          res     = tupi::make_flat_tuple(btfly_res_0);
            auto          res_rep = tupi::group_invoke(simd::evaluate | simd::repack<dst_pck>, res);
            [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                (simd::cxstore<dst_pck>(data_ptr + Width * 2 * Is, get<Is>(res_rep)), ...);
            }(make_uZ_seq<node_size>{});
            if constexpr (LocalTw)
                ++tw_data.start_k;
            return;
        }
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

                    if constexpr (adj_tw_count < 2) {
                        return simd::cxbroadcast<1, 2>(tw.data());
                    } else {
                        auto twv = simd::cxload<1, adj_tw_count>(tw.data());
                        return simd::repack<adj_tw_count>(twv);
                    }
                };
            } else {
                constexpr auto adj_tw_count = half_tw && node_size == 2 ? TwCount / 2 : TwCount;

                auto l_tw_ptr = tw_data.tw_ptr;
                tw_data.tw_ptr += TwCount * 2 * node_size / 2 / (half_tw ? 2 : 1);

                return [=]<uZ KGroup> PCX_LAINLINE(uZ_ce<KGroup>) {
                    constexpr uZ offset = (half_tw ? KGroup / 2 : KGroup) * adj_tw_count * 2;
                    if constexpr (adj_tw_count < 2) {
                        return simd::cxbroadcast<1, 2>(l_tw_ptr + offset);
                    } else {
                        return simd::cxload<adj_tw_count, adj_tw_count>(l_tw_ptr + offset);
                    }
                };
            }
        };


        auto [data_lo, data_hi] = [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            auto lo = tupi::make_tuple(get<Is * 2>(btfly_res_0)...);
            auto hi = tupi::make_tuple(get<Is * 2 + 1>(btfly_res_0)...);
            return tupi::make_tuple(lo, hi);
        }(make_uZ_seq<node_size / 2>{});

        auto [lo, hi] = [=]<uZ NGroups = 2> PCX_LAINLINE    //
            (this auto f, auto data_lo, auto data_hi, uZ_ce<NGroups> = {}) {
                if constexpr (NGroups == width) {
                    return regroup_btfly<NGroups>(data_lo,
                                                  data_hi,
                                                  regroup_tw_fact(uZ_ce<NGroups>{}),
                                                  half_tw);
                } else {
                    auto [lo, hi] =
                        regroup_btfly<NGroups>(data_lo, data_hi, regroup_tw_fact(uZ_ce<NGroups>{}), half_tw);
                    return f(lo, hi, uZ_ce<NGroups * 2>{});
                }
            }(data_lo, data_hi);
        if constexpr (half_tw && node_size > 2) {
            // [ 0  4  8 12] [ 2  6 10 14] before
            // [ 0  2  4  6] [ 8 10 12 14]
            lo = regroup_half_tw(lo);
            hi = regroup_half_tw(hi);
        }
        auto btfly_res_1 = tupi::group_invoke(regroup<1, width>, lo, hi);
        auto res         = tupi::make_flat_tuple(btfly_res_1);
        auto res_rep     = tupi::group_invoke(simd::evaluate | simd::repack<dst_pck>, res);

        [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            (simd::cxstore<dst_pck>(data_ptr + Width * 2 * Is, get<Is>(res_rep)), ...);
        }(make_uZ_seq<node_size>{});
        if constexpr (LocalTw)
            ++tw_data.start_k;
    }
    static void insert_single_load_tw(twiddle_range_for<T> auto& r,
                                      tw_data_t<T, true>&        tw_data,
                                      bool                       lowk,
                                      meta::any_ce_of<bool> auto half_tw) {
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
                if constexpr (NGroups == Width) {
                    insert_regroup_tw(regroup_tw_fact(uZ_ce<NGroups>{}), half_tw);
                } else {
                    insert_regroup_tw(regroup_tw_fact(uZ_ce<NGroups>{}), half_tw);
                    f(uZ_ce<NGroups * 2>{});
                }
            }();
        ++tw_data.start_k;
    }

    /**
     *  @brief Split-regroups input data, loads twiddles and performs a single butterfly operation.
     *  see `split_regroup<>`. 
     *  
     *  @tparam NGroups - number of fft groups (`k`) that fit in a single simd vector.
     */
    template<uZ NGroups>
    struct regroup_btfly_t {
        template<simd::any_cx_vec... Tlo, simd::any_cx_vec... Thi>
        PCX_AINLINE static auto operator()(tupi::tuple<Tlo...>        lo,
                                           tupi::tuple<Thi...>        hi,
                                           auto&&                     get_tw,
                                           meta::any_ce_of<bool> auto half_tw) {
            constexpr uZ   raw_tw_count = half_tw && node_size > 2 ? node_size / 4 : node_size / 2;
            constexpr auto tw_idx_tup   = [=]<uZ... Is>(uZ_seq<Is...>) {
                return tupi::make_tuple(uZ_ce<half_tw ? Is * 2 : Is>{}...);
            }(make_uZ_seq<raw_tw_count>{});

            auto get_tw_tup = tupi::make_broadcast_tuple<raw_tw_count>(get_tw);

            constexpr auto make_j_tuple = [] {
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

            if constexpr (half_tw && node_size > 2) {
                std::tie(lo, hi) = switch_1_2(lo, hi);
            }
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
    };
    template<uZ NGroups>
    constexpr static auto regroup_btfly = regroup_btfly_t<NGroups>{};

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
};

template<uZ NodeSize, typename T, uZ Width>
struct transform {
    using coh_subtf = coherent_subtransform<NodeSize, T, Width>;
    using subtf     = subtransform<NodeSize, T, Width>;

    /**
     * @brief Number of complex elements of type T that keep L1 cache coherency during subtransforms.
     */
    static constexpr auto coherent_size  = uZ_ce<2048>{};
    static constexpr auto width          = uZ_ce<Width>{};
    static constexpr auto w_pck          = cxpack<width, T>{};
    static constexpr auto lane_size      = uZ_ce<std::max(64 / sizeof(T) / 2, Width)>{};
    static constexpr auto coherent_k_cnt = uZ_ce<coherent_size / lane_size>{};
    static constexpr auto node_size      = uZ_ce<NodeSize>{};

    static constexpr auto skip_coherent_subtf = false;

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
        auto a          = slog / log2i(NodeSize);
        auto b          = a * log2i(NodeSize);
        auto align_node = powi(2, slog - b);
        return align_node;
    };

    /**
     * Subdivides the data into buckets. Default bucket size is `coherent_size`.
     * A bucket represents contiguous batches of data, distributed with a constant stride. Default batch size is `lane_size`.
     * A single subdivision can perform a maximum of `coherent_k_cnt` levels of fft.
     */
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void perform(uZ                         data_size,
                                    T* const                   dest_ptr,
                                    tw_data_for<T> auto        tw_data,
                                    meta::any_ce_of<bool> auto half_tw,
                                    meta::any_ce_of<bool> auto lowk) {
        constexpr auto src_pck = cxpack<SrcPackSize, T>{};
        constexpr auto dst_pck = cxpack<DestPackSize, T>{};

        const auto bucket_size  = coherent_size;
        const auto batch_size   = lane_size;
        const auto pass_k_count = bucket_size / batch_size / 2;

        if (data_size <= bucket_size) {
            auto coherent_align_node = coh_subtf::get_align_node(data_size);
            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = powi(2, I);
                    if (l_node_size != coherent_align_node)
                        return false;
                    coh_subtf::perform(dst_pck,
                                       src_pck,
                                       lowk,
                                       half_tw,
                                       data_size,
                                       dest_ptr,
                                       uZ_ce<l_node_size>{},
                                       tw_data);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<log2i(NodeSize)>{});
            return;
        }

        uZ bucket_count       = data_size / bucket_size;    // per bucket group
        uZ bucket_group_count = 1;
        uZ stride             = batch_size * bucket_count;

        auto pass_count       = logKi(pass_k_count * 2, data_size / bucket_size);
        uZ   pre_pass_k_count = data_size / bucket_size / powi(pass_k_count * 2, pass_count) / 2;

        auto iterate_buckets =
            [&](meta::any_ce_of<uZ> auto align_node, cxpack_for<T> auto src_pck, auto k_count) {
                constexpr auto align     = align_param<align_node, true>{};
                auto           l_tw_data = tw_data;
                for (uZ i_b: stdv::iota(0U, bucket_count)) {
                    l_tw_data       = tw_data;
                    auto bucket_ptr = dest_ptr + i_b * batch_size * 2;
                    subtf::perform(src_pck,
                                   align,
                                   lowk,
                                   bucket_size,
                                   stride,
                                   batch_size,
                                   k_count,
                                   bucket_ptr,
                                   l_tw_data);
                }
                tw_data = l_tw_data;
                for (uZ i_bg: stdv::iota(1U, bucket_group_count)) {
                    auto bucket_group_start = dest_ptr + i_bg * bucket_count * bucket_size * 2;
                    for (uZ i_b: stdv::iota(0U, bucket_count)) {
                        if constexpr (l_tw_data.is_local()) {
                            l_tw_data.start_k = i_bg;
                        } else {
                            l_tw_data = tw_data;
                        }
                        auto bucket_ptr = bucket_group_start + i_b * batch_size * 2;
                        subtf::perform(src_pck,
                                       align,
                                       std::false_type{},
                                       bucket_size,
                                       stride,
                                       batch_size,
                                       k_count,
                                       bucket_ptr,
                                       l_tw_data);
                    }
                    tw_data = l_tw_data;
                }
                bucket_count /= k_count * 2;
                stride /= k_count * 2;
                bucket_group_count *= k_count * 2;
                if constexpr (tw_data.is_local()) {
                    tw_data.start_fft_size *= k_count * 2;
                    tw_data.start_k = 0;
                }
            };


        auto pre_pass_align_node = get_align_node(pre_pass_k_count * 2);
        [&]<uZ... Is>(uZ_seq<Is...>) {
            auto check_align = [&]<uZ I>(uZ_ce<I>) {
                constexpr auto l_node_size = powi(2, I);
                if (l_node_size != pre_pass_align_node)
                    return false;
                iterate_buckets(uZ_ce<l_node_size>{}, src_pck, pre_pass_k_count);
                return true;
            };
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<log2i(NodeSize)>{});

        auto           tw_data_bak     = tw_data;
        constexpr auto pass_align_node = get_align_node(pass_k_count * 2);
        for (uZ pass: stdv::iota(0U, pass_count)) {
            tw_data = tw_data_bak;
            iterate_buckets(uZ_ce<pass_align_node>{}, w_pck, pass_k_count);
        }
        if constexpr (skip_coherent_subtf) {
            for (auto i: stdv::iota(0U, data_size / width)) {
                auto ptr = dest_ptr + i * width * 2;
                auto rd  = (simd::cxload<width, width> | simd::repack<1>)(ptr);
                simd::cxstore<1>(ptr, rd);
            }
            return;
        }
        bucket_group_count = data_size / bucket_size;
        if constexpr (tw_data.is_local()) {
            tw_data.start_fft_size = bucket_group_count;
            tw_data.start_k        = 0;
        }
        tw_data_bak = tw_data;

        constexpr auto coherent_align_node = uZ_ce<coh_subtf::get_align_node(bucket_size)>{};
        constexpr auto coh_pck             = dst_pck;
        if constexpr (lowk) {
            coh_subtf::perform(coh_pck,
                               w_pck,
                               lowk,
                               half_tw,
                               bucket_size,
                               dest_ptr,
                               coherent_align_node,
                               tw_data);
        }
        constexpr uZ coh_start = lowk ? 1U : 0U;
        for (uZ i_bg: stdv::iota(coh_start, bucket_group_count)) {
            if constexpr (tw_data.is_local()) {
                tw_data         = tw_data_bak;
                tw_data.start_k = i_bg;
            }
            auto bucket_ptr = dest_ptr + i_bg * bucket_size * 2;
            coh_subtf::perform(coh_pck,
                               w_pck,
                               std::false_type{},
                               half_tw,
                               bucket_size,
                               bucket_ptr,
                               coherent_align_node,
                               tw_data);
        }
    };
    static void
    insert_tw(twiddle_range_for<T> auto& r, uZ data_size, bool lowk, meta::any_ce_of<bool> auto half_tw) {
        const auto bucket_size  = coherent_size;
        const auto batch_size   = lane_size;
        const auto pass_k_count = bucket_size / batch_size / 2;

        if (data_size <= bucket_size) {
            auto coherent_align_node = coh_subtf::get_align_node(data_size);
            [&]<uZ... Is>(uZ_seq<Is...>) {
                auto check_align = [&]<uZ I>(uZ_ce<I>) {
                    constexpr auto l_node_size = powi(2, I);
                    if (l_node_size != coherent_align_node)
                        return false;
                    auto           l_tw_data = tw_data_t<T, true>{1, 0};
                    constexpr auto align     = align_param<l_node_size, true>{};
                    coh_subtf::insert_tw(r, align, lowk, data_size, l_tw_data, half_tw);
                    return true;
                };
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<log2i(NodeSize)>{});
            return;
        }

        auto pass_count       = logKi(pass_k_count * 2, data_size / bucket_size);
        uZ   pre_pass_k_count = data_size / bucket_size / powi(pass_k_count * 2, pass_count) / 2;

        auto iterate_buckets = [&](meta::any_ce_of<uZ> auto align_node,    //
                                   uZ                       k_count,
                                   uZ                       bucket_group_count) {
            constexpr auto align = align_param<align_node, true>{};
            for (uZ i_bg: stdv::iota(0U, bucket_group_count)) {
                auto l_tw_data = tw_data_t<T, true>{bucket_group_count, i_bg};
                subtf::insert_tw(r, align, lowk && i_bg == 0, k_count, l_tw_data);
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
        }(make_uZ_seq<log2i(NodeSize)>{});

        constexpr auto pass_align_node = get_align_node(pass_k_count * 2);
        if (pass_count > 0) {
            const auto final_bucket_group_count =
                pre_pass_k_count * 2 * powi(pass_k_count * 2, pass_count - 1);
            iterate_buckets(uZ_ce<pass_align_node>{}, pass_k_count, final_bucket_group_count);
        }

        if constexpr (skip_coherent_subtf) {
            return;
        };

        auto bucket_count = data_size / bucket_size;

        constexpr auto coherent_align = align_param<coh_subtf::get_align_node(bucket_size), true>{};
        for (uZ i_bg: stdv::iota(0U, bucket_count)) {
            auto l_tw_data = tw_data_t<T, true>{bucket_count, i_bg};
            coh_subtf::insert_tw(r, coherent_align, lowk && i_bg == 0, bucket_size, l_tw_data, half_tw);
        }
    };
};

template<bool DoSort, uZ SortWidth>
struct sort_param {};

template<typename T, uZ Width, bool DoSort>
struct br_sort_inplace {
    static constexpr auto width   = uZ_ce<Width>{};
    static constexpr auto do_sort = std::bool_constant<DoSort>{};

    const uZ* sort_idxs{};
    void      perform(cxpack_for<T> auto pck, uZ size, T* dest_ptr, uZ n_swaps, const uZ* idxs) {
        uZ n_sort_lanes = n_swaps;

        auto next_ptr_tup = [&] PCX_LAINLINE {
            return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                auto idx = *idxs;
                ++idxs;
                return tupi::make_tuple((dest_ptr + idx * width + Is * size / width)...);
            }(make_uZ_seq<width>{});
        };
        bool swap = true;
        while (true) {
            for ([[maybe_unused]] auto i: stdv::iota(0U, n_sort_lanes)) {
                auto ptr_tup   = next_ptr_tup();
                auto data      = tupi::group_invoke(simd::cxload<pck, width>, ptr_tup);
                auto data_perm = simd::br_permute(data);
                if (!swap) {
                    tupi::group_invoke(simd::cxstore<pck>, ptr_tup, data);
                } else {
                    auto ptr_tup2   = next_ptr_tup();
                    auto data2      = tupi::group_invoke(simd::cxload<pck, width>, ptr_tup2);
                    auto data_perm2 = simd::br_permute(data);
                    tupi::group_invoke(simd::cxstore<pck>, ptr_tup, data_perm2);
                    tupi::group_invoke(simd::cxstore<pck>, ptr_tup2, data_perm);
                }
            }
            if (swap) {
                swap         = false;
                n_sort_lanes = size / Width - n_swaps;
                continue;
            }
            break;
        }
    }
    auto insert_idxs(auto& r, uZ size) {
        auto n = size / width / width;
        for (auto i: stdv::iota(0U, n)) {
            auto bri = reverse_bit_order(i, log2i(n));
            if (bri > i) {
                r.insert(i);
                r.insert(bri);
            }
        }
        for (auto i: stdv::iota(0U, n)) {
            auto bri = reverse_bit_order(i, log2i(n));
            if (bri == i) {
                r.insert(i);
            }
        }
    }
};

}    // namespace pcx::detail_
