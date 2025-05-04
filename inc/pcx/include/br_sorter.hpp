#pragma once
#include "pcx/include/meta.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"

#include <vector>

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

/**
 *  @brief sorter for nonsequential transform.
 *
 */
struct br_sorter {
    const u32* idx_ptr;
    u32        noncoh_swap_cnt;
    u32        coh_swap_cnt;

    const auto sort(auto sort_node_size,
                    auto width,
                    auto batch_size,
                    auto dst_pck,
                    auto src_pck,
                    auto dst_data,
                    auto src_data,
                    auto coh) {
        const auto sequential = dst_data.sequential();
        static_assert(!sequential);
        const auto inplace = src_data.empty();
        static_assert(src_data.sequential() == sequential || inplace);
        const auto     swap_cnt      = coh ? coh_swap_cnt : noncoh_swap_cnt;
        constexpr auto swap_grp_size = uZ_ce<2> {}
        constexpr auto rotate_groups = [](tupi::tuple_like auto tuple, auto grp_size) {
            auto make_grp = [&](auto i_grp) {
                return [&]<uZ... Is>(uZ_seq<Is...>) {
                    return tupi::make_tuple(tupi::get_copy<i_grp * grp_size + (Is + 1) % grp_size>(tuple)...);
                };
            };
            return [&]<uZ... Is>(uZ_seq<Is...>) {
                return tupi::make_flat_tuple(make_grp(uZ_ce<Is>)...);
            }(make_uZ_seq<tupi::tuple_size_v<decltype(tuple)> / grp_size>{});
        };
        auto& l_src = [&] {
            if constexpr (inplace)
                return dst_data;
            else
                return src_data;
        }();
        for (auto i: stdv::iota(0U, swap_cnt) | stdv::stride(sort_node_size / 2)) {
            const auto idxs0 = [&]<uZ... Is>(uZ_seq<Is...>) {
                idx_ptr += sort_node_size;
                return tupi::make_tuple(*(idx_ptr - sort_node_size + Is)...);
            }(make_uZ_seq<sort_node_size>{});
            const auto idxs1 = rotate_groups(idxs0, swap_grp_size);

            auto src_base = tupi::group_invoke([&](auto i) { return l_src.get_batch_base(i); }, idxs0);
            auto dst_base = tupi::group_invoke([&](auto i) { return dst_data.get_batch_base(i); }, idxs1);
            for (auto ibs: stdv::iota(0U, batch_size) | stdv::stride(width)) {
                auto src  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, src_base);
                auto dst  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, dst_base);
                auto data = tupi::group_invoke(simd::cxload<src_pck, width>, src);
                tupi::group_invoke(simd::cxstore<dst_pck>, dst, data);
            }
        }
    }

    static constexpr auto n_swaps(uZ fft_size) {
        uZ n_no_swap = powi(2, log2i(fft_size) / 2);
        return fft_size - n_no_swap;
    };
    static auto insert_indexes(auto& r, uZ fft_size, uZ coherent_size) -> uZ {
        auto rbo = [=](auto i) { return reverse_bit_order(i, log2i(fft_size)); };
        uZ   coherent_cnt{};
        for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
            auto l_cnt   = 0;
            auto coh_end = coh_begin + coherent_size;
            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br > i && br < coh_end) {
                    r.push_back(i);
                    r.push_back(br);
                    if (coh_begin == 0)
                        ++coherent_cnt;
                }
            }
        }
        for (auto coh_begin: stdv::iota(0U, fft_size) | stdv::stride(coherent_size)) {
            auto coh_end = coh_begin + coherent_size;
            for (uZ i: stdv::iota(coh_begin, coh_end)) {
                auto br = rbo(i);
                if (br > i && br >= coh_end) {
                    r.push_back(i);
                    r.push_back(br);
                }
            }
        }
        return coherent_cnt;
    }
};
struct br_sorter_shifted {
    const u32* idx_ptr;
    u32        swap_cnt;

    const auto sort(auto sort_node_size,
                    auto width,
                    auto batch_size,
                    auto dst_pck,
                    auto src_pck,
                    auto dst_data,
                    auto src_data,
                    auto coh) {
        const auto sequential = dst_data.sequential();
        static_assert(!sequential);
        const auto inplace = src_data.empty();
        static_assert(src_data.sequential() == sequential || inplace);
        static_assert(sort_node_size >= 4);
        if constexpr (coh) {
            return;
        } else {
            constexpr auto swap_grp_size = uZ_ce<4> {}
            constexpr auto rotate_groups = [](tupi::tuple_like auto tuple, auto grp_size) {
                auto make_grp = [&](auto i_grp) {
                    return [&]<uZ... Is>(uZ_seq<Is...>) {
                        return tupi::make_tuple(
                            tupi::get_copy<i_grp * grp_size + (Is + 1) % grp_size>(tuple)...);
                    };
                };
                return [&]<uZ... Is>(uZ_seq<Is...>) {
                    return tupi::make_flat_tuple(make_grp(uZ_ce<Is>)...);
                }(make_uZ_seq<tupi::tuple_size_v<decltype(tuple)> / grp_size>{});
            };
            auto& l_src = [&] {
                if constexpr (inplace)
                    return dst_data;
                else
                    return src_data;
            }();
            for (auto i: stdv::iota(0U, swap_cnt) | stdv::stride(sort_node_size / 2)) {
                const auto idxs0 = [&]<uZ... Is>(uZ_seq<Is...>) {
                    idx_ptr += sort_node_size;
                    return tupi::make_tuple(*(idx_ptr - sort_node_size + Is)...);
                }(make_uZ_seq<sort_node_size>{});
                const auto idxs1 = rotate_groups(idxs0, swap_grp_size);

                auto src_base = tupi::group_invoke([&](auto i) { return l_src.get_batch_base(i); }, idxs0);
                auto dst_base = tupi::group_invoke([&](auto i) { return dst_data.get_batch_base(i); }, idxs1);
                for (auto ibs: stdv::iota(0U, batch_size) | stdv::stride(width)) {
                    auto src  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, src_base);
                    auto dst  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, dst_base);
                    auto data = tupi::group_invoke(simd::cxload<src_pck, width>, src);
                    tupi::group_invoke(simd::cxstore<dst_pck>, dst, data);
                }
            }
        }
    }
    static constexpr auto n_swaps_shifted(uZ fft_size) {
        return fft_size;
    }
    static void insert_shifted_indexes(auto& r, uZ fft_size) {
        auto rbo = [=](auto i) {
            uZ br = reverse_bit_order(i, log2i(fft_size));
            return (br + fft_size / 2) % fft_size;
        };
        for (uZ i: stdv::iota(0U, fft_size)) {
            auto br1 = rbo(i);
            auto br2 = rbo(br1);
            auto br3 = rbo(br2);
            auto br4 = rbo(br3);
            assert(br4 != i);
            if (i == std::min({i, br1, br2, br3})) {
                r.push_back(i);
                r.push_back(br1);
                r.push_back(br2);
                r.push_back(br3);
            }
        }
    }
};
struct br_sorter_sequential {};
}    // namespace pcx::detail_
