#pragma once
#include "pcx/include/fft/util.hpp"
#include "pcx/include/meta.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"

namespace pcx::simd {
/**
 * @class br_permute_t
 * @brief performs bit-reverse permutation inside a tuple of simd complex vectors.
 *
 * Input data is a tuple of vectors containing `W` simd vectors of width `W`.
 *
 * Permutation consisnt of multiple iterations.
 * A single iteration consists of the following steps:
 * 1. Data is divided into pairs of vectors (`lo` and `hi`) using a `Stride`.
 *    Example: `W` = 8, `Stride` = 4
 *    data:  [lo0] [lo1] [hi0] [hi1] [lo2] [lo3] [hi2] [hi3]
 *    lo:    [lo0] [lo1] [lo2] [lo3]
 *    hi:    [hi0] [hi1] [hi2] [hi3]
 *    pairs: {[lo0][hi0]} {[lo1][hi1]} {[lo2][hi2]} {[lo3][hi3]}
 *
 * 2. Vectors in each pair are split into chunks, and interleaved pair-wise. 
 *    Example: `W` = 8, `Chunk` = 2
 *    pair before: [ 0  1  2  3  4  5  6  7] [ 8  9 10 11 12 13 14 15]
 *    pair after:  [ 0  1  8  9  4  5 12 13] [ 2  3 10 11  6  7 14 15]
 *
 * 3. Vector pairs are combined into a single data. This is the inverse of the step 1.
 *    Example: `W` = 8, `Stride` = 4
 *    lo:    [lo0] [lo1] [lo2] [lo3]
 *    hi:    [hi0] [hi1] [hi2] [hi3]
 *    data:  [lo0] [lo1] [hi0] [hi1] [lo2] [lo3] [hi2] [hi3]
 *
 * Starting `Stride` value is equal to `W`.
 * Starting `Chunk` value is 1.
 * If `Chunk` is equal to complex vector pack size (see `PackSize` in `simd::cx_vec<...>`), the iteration is skipped and `Chunk` is doubled.
 * If the `Stride` is equal to 2 the iteration is final.
 * After each iteration `Stride` is halved and `Chunk` is doubled.
 *
 * If the permutation is shifted, the data lower and upper halves are swapped.
 */
struct br_permute_t {
    template<eval_cx_vec... Vs>
        requires meta::equal_values<sizeof...(Vs), Vs::width()...> && meta::equal_values<Vs::pack_size()...>
    PCX_AINLINE static auto operator()(tupi::tuple<Vs...> data, meta::ce_of<bool> auto shifted) {
        constexpr auto width = uZ_ce<sizeof...(Vs)>{};
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
                           [] PCX_LAINLINE(auto v0, auto v1) {
                               return tupi::make_tuple(tupi::make_tuple(v0.real_v(), v1.real_v()),
                                                       tupi::make_tuple(v0.imag_v(), v1.imag_v()));
                           }
                           | tupi::group_invoke(tupi::apply | traits::template split_interleave<Chunk>)
                           | tupi::apply    //
                           | [] PCX_LAINLINE(auto re, auto im) {
                                 return tupi::make_tuple(cx_vec_t(get<0>(re), get<0>(im)),
                                                         cx_vec_t(get<1>(re), get<1>(im)));
                             };
                } else {
                    return tupi::pass    //
                           | [] PCX_LAINLINE(auto v0, auto v1) {
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
        auto br_data = [=]<uZ Stride, uZ Chunk = 1> PCX_LAINLINE(this auto     f,
                                                                 auto          l_data,
                                                                 uZ_ce<Stride> stride,
                                                                 uZ_ce<Chunk>  chunk = {}) {
            if constexpr (width == 1) {
                return l_data;
            } else if constexpr (chunk == pack) {
                return f(l_data, stride, uZ_ce<chunk * 2>{});
            } else if constexpr (stride == 2) {
                return pass(stride, chunk, l_data);
            } else {
                auto tmp = pass(stride, chunk, l_data);
                return f(tmp, uZ_ce<stride / 2>{}, uZ_ce<chunk * 2>{});
            }
        }(data, width);

        if constexpr (shifted && width > 1) {
            auto [lo, hi] = extract_halves<width>(br_data);
            return combine_halves<width>(hi, lo);
        } else {
            return br_data;
        }
    }
    template<uZ Stride, simd::any_cx_vec... Ts>
    PCX_AINLINE static auto extract_halves(tupi::tuple<Ts...> data) {
        constexpr auto count = sizeof...(Ts);
        auto get_half        = [=]<uZ... Grp> PCX_LAINLINE(uZ_seq<Grp...>, auto start) {
            auto iterate = [=]<uZ... Iters> PCX_LAINLINE(uZ_seq<Iters...>, auto offset) {
                return tupi::make_tuple(tupi::get<offset + Iters>(data)...);
            };
            return tupi::tuple_cat(iterate(make_uZ_seq<Stride / 2>{}, uZ_ce<start + Grp * Stride>{})...);
        };
        return tupi::make_tuple(get_half(make_uZ_seq<count / Stride>{}, uZ_ce<0>{}),
                                get_half(make_uZ_seq<count / Stride>{}, uZ_ce<Stride / 2>{}));
    }
    template<uZ Stride, typename... Tsl, typename... Tsh>
        requires(simd::any_cx_vec<Tsl> && ...) && (simd::any_cx_vec<Tsh> && ...)
    PCX_AINLINE static auto combine_halves(tupi::tuple<Tsl...> lo, tupi::tuple<Tsh...> hi) {
        constexpr auto        count = sizeof...(Tsl) * 2;
        return [=]<uZ... Grp> PCX_LAINLINE(uZ_seq<Grp...>) {
            auto iterate = [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>, auto offset) {
                return tupi::make_tuple(tupi::get<offset + Is>(lo)..., tupi::get<offset + Is>(hi)...);
            };
            return tupi::tuple_cat(iterate(make_uZ_seq<Stride / 2>{}, uZ_ce<Grp * Stride / 2>{})...);
        }(make_uZ_seq<count / Stride>{});
    }
};
inline constexpr auto br_permute = br_permute_t{};
}    // namespace pcx::simd

namespace pcx::detail_ {
struct br_permuter_base {};
inline constexpr struct identity_permuter_t : br_permuter_base {
    static auto coherent_permute(auto /* width */,
                                 auto /* batch_size */,
                                 auto /* reverse */,
                                 auto /* dst_pck */,
                                 auto /* src_pck */,
                                 auto /* dst_data */,
                                 auto src_data) {
        return src_data;
    };
    static auto permute(auto /* width */,
                        auto /* batch_size */,
                        auto /* reverse */,
                        auto /* dst_pck */,
                        auto /* src_pck */,
                        auto /* dst_data */,
                        auto src_data) {
        return src_data;
    };
    static auto small_permute(auto /* width */,
                              auto /* batch_size */,
                              auto /* reverse */,
                              auto /* dst_pck */,
                              auto /* src_pck */,
                              auto /* dst_data */,
                              auto src_data) {
        return src_data;
    };
    static auto sequential_permute(auto /* dst_pck */,    //
                                   auto /* src_pck */,
                                   auto /* dst_data */,
                                   auto src_data) {
        return src_data;
    };
    static constexpr auto empty() -> std::true_type {
        return {};
    }
    static constexpr auto insert_indexes(auto&&...) -> identity_permuter_t {
        return {};
    }
} identity_permuter;


/**
 *  @brief Permuter for nonsequential transform.
 *
 */
template<uZ NodeSize>
struct br_permuter_nonseq_base : public br_permuter_base {
    static constexpr auto empty() -> std::false_type {
        return {};
    }
    using idx_ptr_t = const u32*;
    PCX_AINLINE static auto perm_impl(auto                   width,
                                      auto                   batch_size,
                                      meta::ce_of<bool> auto reverse,
                                      auto                   dst_pck,
                                      auto                   src_pck,
                                      auto                   dst_data,
                                      auto                   src_data,
                                      idx_ptr_t&             idx_ptr,
                                      auto                   swap_cnt,
                                      auto                   nonswap_cnt,
                                      auto                   swap_grp_size) {
        const auto sequential = dst_data.sequential();
        static_assert(!sequential);
        const auto inplace = src_data.empty();
        static_assert(src_data.sequential() == sequential || inplace);
        constexpr auto rotate_groups = [] PCX_LAINLINE(tupi::tuple_like auto tuple, auto grp_size) {
            auto make_grp = [&] PCX_LAINLINE(auto i_grp) {
                return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    return tupi::make_tuple(tupi::get_copy<i_grp * grp_size + (Is + 1) % grp_size>(tuple)...);
                }(make_uZ_seq<grp_size>{});
            };
            return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                return tupi::make_flat_tuple(tupi::make_tuple(make_grp(uZ_ce<Is>{})...));
            }(make_uZ_seq<tupi::tuple_size_v<decltype(tuple)> / grp_size>{});
        };
        decltype(auto) l_src = [&] PCX_LAINLINE -> decltype(auto) {
            if constexpr (inplace)
                return dst_data;
            else
                return src_data;
        }();
        constexpr auto node_p = uZ_ce<log2i(NodeSize)>{};

        auto check_ns = [&] PCX_LAINLINE(auto p) {
            constexpr auto sort_node_size = uZ_ce<powi(2, node_p - p)>{};
            uZ             sns            = sort_node_size;
            if (swap_cnt % (sort_node_size / 2) != 0)
                return false;
            for (auto i: stdv::iota(0U, swap_cnt) | stdv::stride(sort_node_size / 2)) {
                const auto idxs0 = [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    if constexpr (reverse)
                        idx_ptr -= sort_node_size;
                    auto idxs = tupi::make_tuple(*(idx_ptr + Is)...);
                    if constexpr (!reverse)
                        idx_ptr += sort_node_size;
                    return idxs;
                }(make_uZ_seq<sort_node_size>{});
                const auto idxs1 = rotate_groups(idxs0, swap_grp_size);

                auto src_base =
                    tupi::group_invoke([&] PCX_LAINLINE(auto i) { return l_src.get_batch_base(i); }, idxs0);
                auto dst_base =
                    tupi::group_invoke([&] PCX_LAINLINE(auto i) { return dst_data.get_batch_base(i); },
                                       idxs1);
                for (auto ibs: stdv::iota(0U, batch_size) | stdv::stride(width)) {
                    auto src =
                        tupi::group_invoke([=] PCX_LAINLINE(auto base) { return base + ibs * 2; }, src_base);
                    auto dst =
                        tupi::group_invoke([=] PCX_LAINLINE(auto base) { return base + ibs * 2; }, dst_base);
                    auto data = tupi::group_invoke(simd::cxload<src_pck, width>, src);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst, data);
                }
            }
            return true;
        };

        auto check_nonswap_ns = [&](auto p) {
            constexpr auto sort_node_size = uZ_ce<powi(2, node_p - p)>{};
            if (nonswap_cnt % sort_node_size != 0)
                return false;
            for (auto i: stdv::iota(0U, nonswap_cnt) | stdv::stride(sort_node_size)) {
                const auto idxs = [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    if constexpr (reverse)
                        idx_ptr -= sort_node_size;
                    auto idxs = tupi::make_tuple(*(idx_ptr + Is)...);
                    if constexpr (!reverse)
                        idx_ptr += sort_node_size;
                    return idxs;
                }(make_uZ_seq<sort_node_size>{});

                auto src_base =
                    tupi::group_invoke([&] PCX_LAINLINE(auto i) { return l_src.get_batch_base(i); }, idxs);
                auto dst_base =
                    tupi::group_invoke([&] PCX_LAINLINE(auto i) { return dst_data.get_batch_base(i); }, idxs);
                for (auto ibs: stdv::iota(0U, batch_size) | stdv::stride(width)) {
                    auto src =
                        tupi::group_invoke([=] PCX_LAINLINE(auto base) { return base + ibs * 2; }, src_base);
                    auto dst =
                        tupi::group_invoke([=] PCX_LAINLINE(auto base) { return base + ibs * 2; }, dst_base);
                    auto data = tupi::group_invoke(simd::cxload<src_pck, width>, src);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst, data);
                }
            }
            return true;
        };
        if constexpr (reverse) {
            if constexpr (inplace) {
                idx_ptr -= nonswap_cnt;
            } else {
                [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    (void)(check_nonswap_ns(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<node_p + 1>{});
            }
        }

        [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
            (void)(check_ns(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<node_p - log2i(swap_grp_size) + 1>{});

        if constexpr (!reverse) {
            if constexpr (inplace) {
                idx_ptr += nonswap_cnt;
            } else {
                [=]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    (void)(check_nonswap_ns(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<node_p + 1>{});
            }
        }
    }
};
template<uZ NodeSize>
struct br_permuter : br_permuter_nonseq_base<NodeSize> {
    static constexpr auto node_size = uZ_ce<NodeSize>{};
    using base_t                    = br_permuter_nonseq_base<NodeSize>;

    const u32* idx_ptr;
    u32        coh_swap_cnt;
    u32        coh_nonswap_cnt;
    u32        noncoh_swap_cnt;
    u32        noncoh_nonswap_cnt;

    PCX_AINLINE auto permute(auto width,
                             auto batch_size,
                             auto reverse,
                             auto dst_pck,
                             auto src_pck,
                             auto dst_data,
                             auto src_data) {
        base_t::perm_impl(width,
                          batch_size,
                          reverse,
                          dst_pck,
                          src_pck,
                          dst_data,
                          src_data,
                          idx_ptr,
                          noncoh_swap_cnt,
                          noncoh_nonswap_cnt,
                          uZ_ce<2>{});
        if constexpr (reverse)
            return src_data;
        else
            return inplace_src;
    };
    PCX_AINLINE auto coherent_permute(auto width,
                                      auto batch_size,
                                      auto reverse,
                                      auto dst_pck,
                                      auto src_pck,
                                      auto dst_data,
                                      auto src_data) {
        base_t::perm_impl(width,
                          batch_size,
                          reverse,
                          dst_pck,
                          src_pck,
                          dst_data,
                          src_data,
                          idx_ptr,
                          coh_swap_cnt,
                          coh_nonswap_cnt,
                          uZ_ce<2>{});
        return inplace_src;
    };
    PCX_AINLINE auto small_permute(auto width,
                                   auto batch_size,
                                   auto reverse,
                                   auto dst_pck,
                                   auto src_pck,
                                   auto dst_data,
                                   auto src_data) {
        base_t::perm_impl(width,
                          batch_size,
                          reverse,
                          dst_pck,
                          src_pck,
                          dst_data,
                          src_data,
                          idx_ptr,
                          coh_swap_cnt,
                          coh_nonswap_cnt,
                          uZ_ce<2>{});
        return inplace_src;
    }

    static constexpr auto n_swaps(uZ fft_size) {
        uZ n_no_swap = powi(2, log2i(fft_size) / 2);
        return fft_size - n_no_swap;
    };
    static auto insert_indexes(auto& r, uZ fft_size, uZ coherent_size) -> br_permuter {
        auto rbo = [=] PCX_LAINLINE(auto i) { return reverse_bit_order(i, log2i(fft_size)); };
        u32  coh_swap_cnt{};
        u32  coh_nonswap_cnt{};
        bool coh_nonswap = true;
        for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
            auto l_cnt         = 0;
            auto l_nonswap_cnt = 0;
            auto coh_end       = coh_begin + std::min({coherent_size, fft_size});
            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br > i && br < coh_end)
                    ++l_cnt;
            }
            if (coh_begin == 0)
                coh_swap_cnt = l_cnt;
            if (l_cnt != coh_swap_cnt)
                throw std::runtime_error("Coherent swap count differs between coherent batches");

            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br == i)
                    ++l_nonswap_cnt;
            }
            if (coh_begin == 0)
                coh_nonswap_cnt = l_nonswap_cnt;
            if (l_nonswap_cnt != coh_nonswap_cnt)
                coh_nonswap = false;
        }
        coh_nonswap     = coh_nonswap && coh_nonswap_cnt != 0;
        coh_nonswap_cnt = coh_nonswap ? coh_nonswap_cnt : 0;

        for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
            auto coh_end = coh_begin + std::min({coherent_size, fft_size});
            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br > i && br < coh_end) {
                    r.push_back(i - coh_begin);
                    r.push_back(br - coh_begin);
                }
            }
            if (coh_nonswap) {
                for (uZ i: stdv::iota(coh_begin, coh_end)) {
                    auto br = rbo(i);
                    if (br == i)
                        r.push_back(i - coh_begin);
                }
            }
        }
        u32 noncoh_swap_cnt{};
        u32 noncoh_nonswap_cnt{};

        for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
            auto coh_end = coh_begin + std::min({coherent_size, fft_size});
            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br > i && br >= coh_end) {
                    r.push_back(i);
                    r.push_back(br);
                    ++noncoh_swap_cnt;
                }
            }
        }
        if (!coh_nonswap) {
            for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
                auto coh_end = coh_begin + std::min({coherent_size, fft_size});
                for (uZ i: stdv::iota(coh_begin, coh_end)) {
                    auto br = rbo(i);
                    if (br == i) {
                        r.push_back(i);
                        ++noncoh_nonswap_cnt;
                    }
                }
            }
        }
        return br_permuter{{}, nullptr, coh_swap_cnt, coh_nonswap_cnt, noncoh_swap_cnt, noncoh_nonswap_cnt};
    }
};
template<uZ NodeSize>
    requires(NodeSize >= 4)
struct br_permuter_shifted : public br_permuter_nonseq_base<NodeSize> {
    using base_t = br_permuter_nonseq_base<NodeSize>;
    const u32* idx_ptr;
    u32        swap_cnt;

    PCX_AINLINE auto permute(auto width,
                             auto batch_size,
                             auto reverse,
                             auto dst_pck,
                             auto src_pck,
                             auto dst_data,
                             auto src_data) {
        base_t::perm_impl(width,
                          batch_size,
                          reverse,
                          dst_pck,
                          src_pck,
                          dst_data,
                          src_data,
                          idx_ptr,
                          swap_cnt,
                          uZ_ce<0>{},
                          uZ_ce<4>{});
        return inplace_src;
    };
    PCX_AINLINE auto small_permute(auto width,
                                   auto batch_size,
                                   auto reverse,
                                   auto dst_pck,
                                   auto src_pck,
                                   auto dst_data,
                                   auto src_data) {
        base_t::perm_impl(width,
                          batch_size,
                          reverse,
                          dst_pck,
                          src_pck,
                          dst_data,
                          src_data,
                          idx_ptr,
                          swap_cnt,
                          uZ_ce<0>{},
                          uZ_ce<4>{});
        return inplace_src;
    };
    PCX_AINLINE auto coherent_permute(auto...) {
        return inplace_src;
    }

    static constexpr auto n_swaps_shifted(uZ fft_size) {
        return fft_size;
    }
    static auto insert_indexes(auto& r, uZ fft_size, uZ /*coherent_size*/ = 0) {
        auto rbo = [=](auto i) {
            auto br = reverse_bit_order(i, log2i(fft_size));
            return (br + fft_size / 2) % fft_size;
        };
        for (u32 i: stdv::iota(0U, fft_size)) {
            u32 br1 = rbo(i);
            u32 br2 = rbo(br1);
            u32 br3 = rbo(br2);
            if (i == std::min({i, br1, br2, br3})) {
                r.push_back(i);
                r.push_back(br1);
                r.push_back(br2);
                r.push_back(br3);
            }
        }
        return br_permuter_shifted{{}, nullptr, static_cast<u32>(fft_size / 2)};
    }
};

template<uZ Width, bool Shifted = false>
struct br_permuter_sequential : public br_permuter_base {
    static constexpr auto width   = uZ_ce<Width>{};
    static constexpr auto shifted = std::bool_constant<Shifted>{};

    const u32* idx_ptr;
    u32        swap_cnt;
    u32        nonswap_cnt;

    PCX_AINLINE auto sequential_permute(auto dst_pck,    //
                                        auto src_pck,
                                        auto dst_data,
                                        auto src_data) {
        const auto subsize = (swap_cnt * 2 + nonswap_cnt) * width;
        const auto inplace = src_data.empty();

        uZ lane_cnt = swap_cnt;
        if constexpr (inplace) {
            auto next_ptr_tup = [&] PCX_LAINLINE {
                return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    auto idx = *idx_ptr;
                    ++idx_ptr;
                    return tupi::make_tuple(
                        (dst_data.get_batch_base(0) + idx * width * 2 + Is * subsize * 2)...);
                }(make_uZ_seq<width>{});
            };
            if constexpr (Shifted && width == 1) {
                for (auto i: stdv::iota(0U, lane_cnt)) {
                    auto dst0  = next_ptr_tup();
                    auto dst1  = next_ptr_tup();
                    auto dst2  = next_ptr_tup();
                    auto dst3  = next_ptr_tup();
                    auto data0 = tupi::group_invoke(simd::cxload<src_pck, width>, dst0);
                    auto data1 = tupi::group_invoke(simd::cxload<src_pck, width>, dst1);
                    auto data2 = tupi::group_invoke(simd::cxload<src_pck, width>, dst2);
                    auto data3 = tupi::group_invoke(simd::cxload<src_pck, width>, dst3);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst1, data0);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst2, data1);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst3, data2);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data3);
                }
            } else {
                bool swap = true;
                while (true) {
                    for (auto i: stdv::iota(0U, lane_cnt)) {
                        auto dst0       = next_ptr_tup();
                        auto data0      = tupi::group_invoke(simd::cxload<src_pck, width>, dst0);
                        auto data_perm0 = simd::br_permute(data0, shifted);
                        if (!swap) {
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm0);
                        } else {
                            auto dst1       = next_ptr_tup();
                            auto data1      = tupi::group_invoke(simd::cxload<src_pck, width>, dst1);
                            auto data_perm1 = simd::br_permute(data1, shifted);
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm1);
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst1, data_perm0);
                        }
                    }
                    if (swap) {
                        swap     = false;
                        lane_cnt = nonswap_cnt;
                        continue;
                    }
                    break;
                }
            }
        } else {
            auto next_ptr_tup = [&] PCX_LAINLINE {
                return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    auto idx = *idx_ptr;
                    ++idx_ptr;
                    auto mktup = [=] PCX_LAINLINE(auto data) {
                        return tupi::make_tuple(
                            (data.get_batch_base(0) + idx * width * 2 + Is * subsize * 2)...);
                    };
                    return tupi::make_tuple(mktup(dst_data), mktup(src_data));
                }(make_uZ_seq<width>{});
            };
            if constexpr (Shifted && width == 1) {
                for (auto i: stdv::iota(0U, lane_cnt)) {
                    auto [dst0, src0] = next_ptr_tup();
                    auto [dst1, src1] = next_ptr_tup();
                    auto [dst2, src2] = next_ptr_tup();
                    auto [dst3, src3] = next_ptr_tup();
                    auto data0        = tupi::group_invoke(simd::cxload<src_pck, width>, src0);
                    auto data1        = tupi::group_invoke(simd::cxload<src_pck, width>, src1);
                    auto data2        = tupi::group_invoke(simd::cxload<src_pck, width>, src2);
                    auto data3        = tupi::group_invoke(simd::cxload<src_pck, width>, src3);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst1, data0);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst2, data1);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst3, data2);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data3);
                }
                lane_cnt = nonswap_cnt;
                for (auto i: stdv::iota(0U, lane_cnt)) {
                    auto [dst0, src0] = next_ptr_tup();
                    auto data0        = tupi::group_invoke(simd::cxload<src_pck, width>, src0);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data0);
                }
            } else {
                bool swap = true;
                while (true) {
                    for (auto i: stdv::iota(0U, lane_cnt)) {
                        auto [dst0, src0] = next_ptr_tup();
                        auto data0        = tupi::group_invoke(simd::cxload<src_pck, width>, src0);
                        auto data_perm0   = simd::br_permute(data0, shifted);
                        if (!swap) {
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm0);
                        } else {
                            auto [dst1, src1] = next_ptr_tup();
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst1, data_perm0);
                            auto data1      = tupi::group_invoke(simd::cxload<src_pck, width>, src1);
                            auto data_perm1 = simd::br_permute(data1, shifted);
                            tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm1);
                        }
                    }
                    if (swap) {
                        swap     = false;
                        lane_cnt = nonswap_cnt;
                        continue;
                    }
                    break;
                }
            }
        }
        return inplace_src;
    }
    static auto insert_indexes(auto& r, uZ fft_size, uZ /*coherent_size*/ = 0) -> br_permuter_sequential {
        u32 n = fft_size / width / width;
        u32 swap_cnt{};
        if constexpr (Shifted && width == 1) {
            if (fft_size == 2) {
                r.push_back(0);
                r.push_back(1);
                return br_permuter_sequential{{}, nullptr, 0, 2};
            }
            auto rbo = [=](auto i) {
                auto br = reverse_bit_order(i, log2i(fft_size));
                return (br + fft_size / 2) % fft_size;
            };
            for (u32 i: stdv::iota(0U, fft_size)) {
                u32 br1 = rbo(i);
                u32 br2 = rbo(br1);
                u32 br3 = rbo(br2);
                if (i == std::min({i, br1, br2, br3})) {
                    r.push_back(i);
                    r.push_back(br1);
                    r.push_back(br2);
                    r.push_back(br3);
                    ++swap_cnt;
                }
            }
            return br_permuter_sequential{{}, nullptr, swap_cnt, n - 4 * swap_cnt};
        } else {
            for (auto i: stdv::iota(0U, n)) {
                auto bri = reverse_bit_order(i, log2i(n));
                if (bri > i) {
                    r.push_back(i);
                    r.push_back(bri);
                    ++swap_cnt;
                }
            }
            for (auto i: stdv::iota(0U, n)) {
                auto bri = reverse_bit_order(i, log2i(n));
                if (bri == i)
                    r.push_back(i);
            }
            return br_permuter_sequential{{}, nullptr, swap_cnt, n - 2 * swap_cnt};
        }
    }
};
}    // namespace pcx::detail_
