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

private:
    template<uZ DestPackSize, uZ SrcPackSize>
    PCX_AINLINE static void fwd_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize, Width>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = 2> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_constant<Size> = {}) {
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
    PCX_AINLINE static void rev_impl(const dest_t& dest, auto src, auto get_tw) {
        auto data     = tupi::group_invoke(simd::cxload<SrcPackSize>, src);
        auto data_rep = tupi::group_invoke(simd::repack<Width>, data);
        auto res      = []<uZ Size = NodeSize> PCX_LAINLINE    //
            (this auto f, auto data, auto get_tw, uZ_constant<Size> = {}) {
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

template<uZ NodeSize, typename T, uZ Width>
struct subtransform {
    using btfly_node = btfly_node_dit<NodeSize, T, Width>;
    using vec_traits = simd::detail_::vec_traits<T, Width>;


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


private:
    template<uZ NodeSizeL>
    PCX_AINLINE auto iterate_lo_k(uZ max_size, uZ& size, auto data_ptr, auto tw_ptr) {
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
    PCX_AINLINE auto iterate(uZ max_size, uZ& size, auto data_ptr, auto& tw_ptr) {
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

    PCX_AINLINE auto single_load_lo_k(auto data_ptr, auto tw_ptr) {
        auto data = []<uZ... Is>(auto data_ptr, std::index_sequence<Is...>) {
            return tupi::make_tuple(simd::cxload<1, Width>(data_ptr + Width * 2 * Is)...);
        }(data_ptr, std::make_index_sequence<NodeSize>{});


        // quick maffs
        // GropsTuple = 0..NodeSize/2;
        //
        // template<uZ Count>
        // auto cl_btfly(tuple<...> lo, tuple<...> hi, const T* tw_ptr){
        // auto tw = tupi::group_invoke(load_tw<Count>, tupi::make_broadcast_tuple<NodeSize / 2>(tw_ptr), GroupTuple);
        // auto vs = tupi::group_invoke(regroup<16, NodeSize / Count>, v_lo, v_hi);
        // auto v_hi_tw = tupi::group_invoke(simd::mul, v_hi, tw);
        // auto btfly_res = tupi::group_invoke(simd::btfly, v_hi, tw);
        // auto new_lo    = tupi::group_invoke([](auto p) { return get<0>(p); }, btfly_res);
        // auto new_hi    = tupi::group_invoke([](auto p) { return get<1>(p); }, btfly_res);
        // return tupi::make_tuple(new_lo, new_hi, tw_ptr + Count * 2 * NodeSize / 2);
        // }
        //

        // single_load(v0, v1){
        // traits::repack<Width, Width / 2>(v0, v1);
        // btfly
        // traits::regroup<Width, Width / 4>(v0, v1);
        // ...
        // traits::reapck<Width, 1>
        // btfly
        // traits::repack<1, Width>
        // traits::repack<2, Width>
        // ...
        // traits::repack<Width / 2, Width>
        // simd::repack<pack_dest>(v0);
        // simd::repack<pack_dest>(v1);
        // }
        //
    }

    template<uZ GroupSize>
    auto foo(auto lo, auto hi, auto tw) {
        regroup<Width, GroupSize>(lo, hi);
        auto hi_tw   = simd::mul(hi, tw);
        auto newlohi = simd::btfly(lo, hi_tw);
    }

    template<uZ Count>
        requires(Count > 1)
    struct load_tw_t : tupi::compound_op_base {
        template<uZ IGroup>
        PCX_AINLINE auto operator()(const T* tw_ptr, uZ_constant<IGroup>) {
            auto re          = simd::load<Count>(tw_ptr + Count * 2 * IGroup);
            auto im          = simd::load<Count>(tw_ptr + Count * 2 * (IGroup + 1));
            auto re_upsample = vec_traits::upsample(re.value);
            auto im_upsample = vec_traits::upsample(im.value);
            return simd::cx_vec<T, false, false, Width>{.m_real = re_upsample, .m_imag = im_upsample};
        }
        template<uZ I>
        PCX_AINLINE constexpr friend auto get_stage(const load_tw_t&) {
            return stage_t<I>{};
        }

    private:
        template<uZ I>
        struct stage_t {
            template<uZ Offset>
            PCX_AINLINE auto operator()(const T* tw_ptr, uZ_constant<Offset>) {
                auto re = simd::load<Count>(tw_ptr + Offset);
                auto im = simd::load<Count>(tw_ptr + Count + Offset);
                return tupi::make_interim(re, im);
            };
            PCX_AINLINE auto operator()(auto re, auto im)
                requires(I == 1)
            {
                constexpr auto upsample = vec_traits::upsample;
                if constexpr (tupi::compound_op<decltype(upsample)>) {
                    auto stage = get_stage<I - 1>(upsample);
                    if constexpr (tupi::final_result<decltype(stage(re))>) {
                        auto re_upsample = stage(re.value);
                        auto im_upsample = stage(im.value);
                        return simd::cx_vec<T, false, false, Width>{.m_real = re_upsample,
                                                                    .m_imag = im_upsample};
                    } else {
                        return tupi::make_interim(stage(re.value), stage(im.value));
                    }
                } else {
                    auto re_upsample = upsample(re.value);
                    auto im_upsample = upsample(im.value);
                    return simd::cx_vec<T, false, false, Width>{.m_real = re_upsample,    //
                                                                .m_imag = im_upsample};
                }
            }
            PCX_AINLINE auto operator()(auto re, auto im)
                requires(I > 1)
            {
                constexpr auto upsample = vec_traits::upsample;
                auto           stage    = get_stage<I - 1>(upsample);
                if constexpr (tupi::final_result<decltype(stage(re))>) {
                    auto re_upsample = stage(re.value);
                    auto im_upsample = stage(im.value);
                    return simd::cx_vec<T, false, false, Width>{.m_real = re_upsample, .m_imag = im_upsample};
                } else {
                    return tupi::make_interim(stage(re.value), stage(im.value));
                }
            }
        };
    };

    template<uZ To, uZ From>
    struct regroup_t : tupi::compound_op_base {
        template<simd::any_cx_vec V>
            requires(To <= V::width()) && (From <= V::width())
        PCX_AINLINE auto operator()(V a, V b) const {
            constexpr auto repack = vec_traits::template repack<To, From>;
            auto [re_a, re_b]     = repack(a.real().native, b.real().native);
            auto [im_a, im_b]     = repack(a.imag().native, b.imag().native);
            return tupi::make_tuple(V{.m_real = re_a, .m_imag = im_a},    //
                                    V{.m_real = re_b, .m_imag = im_b});
        }
        template<uZ I>
        PCX_AINLINE constexpr friend auto get_stage(const regroup_t&) {
            return stage_t<I>{};
        }

    private:
        template<bool NReal, bool NImag, typename IR>
        struct interim_wrapper {
            IR result;
        };
        template<bool NReal, bool NImag, typename IR>
        PCX_AINLINE static constexpr auto wrap_interim(IR res) {
            return tupi::make_interim(interim_wrapper<NReal, NImag, IR>(res));
        }
        template<uZ I>
        struct stage_t {
            template<simd::any_cx_vec V>
                requires(I == 0)
            PCX_AINLINE auto operator()(V a, V b) const {
                constexpr auto repack = vec_traits::template repack<To, From>;
                if constexpr (tupi::compound_op<decltype(repack)>) {
                    auto stage = get_stage<I>(repack);
                    if constexpr (tupi::final_result<decltype(stage(a.real().native, b.real().native))>) {
                        auto [re_a, re_b] = stage(a.real().native, b.real().native);
                        auto [im_a, im_b] = stage(a.imag().native, b.imag().native);
                        return tupi::make_tuple(V{.m_real = re_a, .m_imag = im_a},
                                                V{.m_real = re_b, .m_imag = im_b});
                    } else {
                        return wrap_interim<V::neg_real(), V::neg_imag()>(
                            tupi::make_tuple(stage(a.real().native, b.real().native),
                                             stage(a.imag().native, b.imag().native)));
                    }
                } else {
                    auto [re_a, re_b] = repack(a.real().native, b.real().native);
                    auto [im_a, im_b] = repack(a.imag().native, b.imag().native);
                    return tupi::make_tuple(V{.m_real = re_a, .m_imag = im_a},
                                            V{.m_real = re_b, .m_imag = im_b});
                }
            }
            template<bool NReal, bool NImag, typename IR>
                requires(I > 0)
            PCX_AINLINE auto operator()(interim_wrapper<NReal, NImag, IR> wrapper) const {
                constexpr auto repack = vec_traits::template repack<To, From>;

                auto stage = get_stage<I>(repack);
                if constexpr (tupi::final_result<decltype(tupi::apply(stage, get<0>(wrapper.result)))>) {
                    using cx_vec      = simd::cx_vec<T, NReal, NImag, Width, To>;
                    auto [re_a, re_b] = tupi::apply(stage, get<0>(wrapper.result));
                    auto [im_a, im_b] = tupi::apply(stage, get<1>(wrapper.result));
                    return tupi::make_tuple(cx_vec{.m_real = re_a, .m_imag = im_a},
                                            cx_vec{.m_real = re_b, .m_imag = im_b});
                } else {
                    return wrap_interim<NReal, NImag>(
                        tupi::make_tuple(tupi::apply(stage, get<0>(wrapper.result)),
                                         tupi::apply(stage, get<1>(wrapper.result))));
                }
            }
        };
    };
    template<uZ To, uZ From>
    static constexpr auto regroup = regroup_t<To, From>{};
};

}    // namespace pcx::detail_
