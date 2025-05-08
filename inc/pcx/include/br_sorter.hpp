#pragma once
#include "pcx/include/meta.hpp"
#include "pcx/include/simd/common.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"


namespace pcx::simd {
struct br_permute_t {
    template<eval_cx_vec... Vs>
        requires meta::equal_values<sizeof...(Vs), Vs::width()...> && meta::equal_values<Vs::pack_size()...>
    PCX_AINLINE static auto operator()(tupi::tuple<Vs...> data) {
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
        return [pass]<uZ Stride, uZ Chunk = 1> PCX_LAINLINE(this auto     f,
                                                            auto          l_data,
                                                            uZ_ce<Stride> stride,
                                                            uZ_ce<Chunk>  chunk = {}) {
            if constexpr (chunk == pack) {
                return f(l_data, stride, uZ_ce<chunk * 2>{});
            } else if constexpr (stride == 2) {
                return pass(stride, chunk, l_data);
            } else {
                auto tmp = pass(stride, chunk, l_data);
                return f(tmp, uZ_ce<stride / 2>{}, uZ_ce<chunk * 2>{});
            }
        }(data, width);
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
};
inline constexpr auto br_permute = br_permute_t{};

}    // namespace pcx::simd
namespace pcx::detail_ {
template<typename T>
struct data_info_base {};

template<floating_point T, bool Contiguous, typename C = void>
struct data_info : public data_info_base<T> {
    using data_ptr_t    = std::conditional_t<Contiguous, T*, C*>;
    using data_offset_t = std::conditional_t<Contiguous, decltype([] {}), uZ>;
    using k_offset_t    = std::conditional_t<Contiguous, decltype([] {}), uZ>;
    using k_stride_t    = std::conditional_t<Contiguous, uZ, decltype([] {})>;

    data_ptr_t                          data_ptr;
    uZ                                  stride = 1;
    [[no_unique_address]] k_stride_t    k_stride;
    [[no_unique_address]] k_offset_t    k_offset{};
    [[no_unique_address]] data_offset_t data_offset{};

    static constexpr auto sequential() -> std::false_type {
        return {};
    }
    static constexpr auto contiguous() -> std::bool_constant<Contiguous> {
        return {};
    }
    static constexpr auto empty() -> std::false_type {
        return {};
    }

    constexpr auto mul_stride(uZ n) const -> data_info {
        auto new_info = *this;
        new_info.stride *= n;
        return new_info;
    }
    constexpr auto div_stride(uZ n) const -> data_info {
        auto new_info = *this;
        new_info.stride /= n;
        return new_info;
    }
    constexpr auto offset_k(uZ n) const -> data_info {
        auto new_info = *this;
        if constexpr (Contiguous) {
            new_info.data_ptr += k_stride * n * 2;
        } else {
            new_info.k_offset += n;
        }
        return new_info;
    }
    constexpr auto offset_contents(uZ n) const -> data_info {
        auto new_info = *this;
        if constexpr (Contiguous) {
            new_info.data_ptr += n * 2;
        } else {
            new_info.data_offset += n;
        }
        return new_info;
    }
    constexpr auto get_batch_base(uZ i) const -> T* {
        if constexpr (Contiguous) {
            return data_ptr + i * stride * 2;
        } else {
            auto ptr = reinterpret_cast<T*>((*data_ptr)[i * stride + k_offset].data());
            return ptr + data_offset * 2;
        }
    };
};
template<floating_point T>
struct sequential_data_info : public data_info_base<T> {
    T* data_ptr;
    uZ stride = 1;

    static constexpr auto sequential() -> std::true_type {
        return {};
    }
    static constexpr auto contiguous() -> std::true_type {
        return {};
    }
    static constexpr auto empty() -> std::false_type {
        return {};
    }

    constexpr auto mul_stride(uZ n) const -> sequential_data_info {
        auto new_info = *this;
        new_info.stride *= n;
        return new_info;
    }
    constexpr auto div_stride(uZ n) const -> sequential_data_info {
        auto new_info = *this;
        new_info.stride /= n;
        return new_info;
    }
    constexpr auto offset_k(uZ n) const -> sequential_data_info {
        return {{}, data_ptr + n * 2, stride};
    }
    constexpr auto offset_contents(uZ n) const -> sequential_data_info {
        return {{}, data_ptr + n * 2, stride};
    }
    constexpr auto get_batch_base(uZ i) const -> T* {
        return data_ptr + i * stride * 2;
    };
};
struct empty_data_info {
    static constexpr auto sequential() -> std::false_type {
        return {};
    }
    static constexpr auto contiguous() -> std::false_type {
        return {};
    }
    static constexpr auto empty() -> std::true_type {
        return {};
    }
    static constexpr auto mul_stride(uZ /*n*/) -> empty_data_info {
        return {};
    }
    static constexpr auto div_stride(uZ /*n*/) -> empty_data_info {
        return {};
    }
    static constexpr auto offset_k(uZ /*n*/) -> empty_data_info {
        return {};
    };
    static constexpr auto offset_contents(uZ /*n*/) -> empty_data_info {
        return {};
    };
};

template<typename T, typename U>
concept data_info_for = floating_point<U>
                        && (std::derived_from<T, data_info_base<U>>                            //
                            || std::derived_from<T, data_info_base<std::remove_const_t<U>>>    //
                            || std::same_as<T, empty_data_info>);

inline constexpr auto inplace_src = empty_data_info{};
constexpr auto        log2i(u64 num) -> uZ {
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

struct br_sorter_base {};
inline constexpr struct unsorted_t : br_sorter_base {
    static auto coherent_sort(auto width,
                              auto batch_size,
                              auto reverse,
                              auto dst_pck,
                              auto src_pck,
                              auto dst_data,
                              auto src_data) {
        return src_data;
    };
    static auto sort(auto width,
                     auto batch_size,
                     auto reverse,
                     auto dst_pck,
                     auto src_pck,
                     auto dst_data,
                     auto src_data) {
        return src_data;
    };
    static auto small_sort(auto width,
                           auto batch_size,
                           auto reverse,
                           auto dst_pck,
                           auto src_pck,
                           auto dst_data,
                           auto src_data) {
        return src_data;
    };
    static auto sequential_sort(auto dst_pck,    //
                                auto src_pck,
                                auto dst_data,
                                auto src_data) {
        return src_data;
    };
    static constexpr auto empty() -> std::true_type {
        return {};
    }
} blank_sorter;


/**
 *  @brief sorter for nonsequential transform.
 *
 */
template<uZ NodeSize>
struct br_sorter_nonseq : public br_sorter_base {
    static constexpr auto empty() -> std::false_type {
        return {};
    }
    using idx_ptr_t = const u32*;
    static auto sort_impl(auto                   width,
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
        constexpr auto rotate_groups = [](tupi::tuple_like auto tuple, auto grp_size) {
            auto make_grp = [&](auto i_grp) {
                return [&]<uZ... Is>(uZ_seq<Is...>) {
                    return tupi::make_tuple(tupi::get_copy<i_grp * grp_size + (Is + 1) % grp_size>(tuple)...);
                }(make_uZ_seq<grp_size>{});
            };
            return [&]<uZ... Is>(uZ_seq<Is...>) {
                return tupi::make_flat_tuple(tupi::make_tuple(make_grp(uZ_ce<Is>{})...));
            }(make_uZ_seq<tupi::tuple_size_v<decltype(tuple)> / grp_size>{});
        };
        decltype(auto) l_src = [&] -> decltype(auto) {
            if constexpr (inplace)
                return dst_data;
            else
                return src_data;
        }();
        constexpr auto node_p = uZ_ce<log2i(NodeSize)>{};

        auto check_ns = [&](auto p) {
            constexpr auto sort_node_size = uZ_ce<powi(2, node_p - p)>{};
            uZ             sns            = sort_node_size;
            if (swap_cnt % (sort_node_size / 2) != 0)
                return false;
            for (auto i: stdv::iota(0U, swap_cnt) | stdv::stride(sort_node_size / 2)) {
                const auto idxs0 = [&]<uZ... Is>(uZ_seq<Is...>) {
                    if constexpr (reverse)
                        idx_ptr -= sort_node_size;
                    auto idxs = tupi::make_tuple(*(idx_ptr + Is)...);
                    if constexpr (!reverse)
                        idx_ptr += sort_node_size;
                    return idxs;
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
            return true;
        };

        auto check_nonswap_ns = [&](auto p) {
            constexpr auto sort_node_size = uZ_ce<powi(2, node_p - p)>{};
            if (nonswap_cnt % sort_node_size != 0)
                return false;
            for (auto i: stdv::iota(0U, nonswap_cnt) | stdv::stride(sort_node_size)) {
                const auto idxs = [&]<uZ... Is>(uZ_seq<Is...>) {
                    if constexpr (reverse)
                        idx_ptr -= sort_node_size;
                    auto idxs = tupi::make_tuple(*(idx_ptr + Is)...);
                    if constexpr (!reverse)
                        idx_ptr += sort_node_size;
                    return idxs;
                }(make_uZ_seq<sort_node_size>{});

                auto src_base = tupi::group_invoke([&](auto i) { return l_src.get_batch_base(i); }, idxs);
                auto dst_base = tupi::group_invoke([&](auto i) { return dst_data.get_batch_base(i); }, idxs);
                for (auto ibs: stdv::iota(0U, batch_size) | stdv::stride(width)) {
                    auto src  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, src_base);
                    auto dst  = tupi::group_invoke([=](auto base) { return base + ibs * 2; }, dst_base);
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
                [=]<uZ... Is>(uZ_seq<Is...>) {
                    (void)(check_nonswap_ns(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<node_p + 1>{});
            }
        }

        [=]<uZ... Is>(uZ_seq<Is...>) {
            (void)(check_ns(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<node_p - log2i(swap_grp_size) + 1>{});

        if constexpr (!reverse) {
            if constexpr (inplace) {
                idx_ptr += nonswap_cnt;
            } else {
                [=]<uZ... Is>(uZ_seq<Is...>) {
                    (void)(check_nonswap_ns(uZ_ce<Is>{}) || ...);
                }(make_uZ_seq<node_p + 1>{});
            }
        }
    }
};
template<uZ NodeSize>
struct br_sorter : br_sorter_nonseq<NodeSize> {
    static constexpr auto node_size = uZ_ce<NodeSize>{};
    using impl_t                    = br_sorter_nonseq<NodeSize>;

    const u32* idx_ptr;
    u32        coh_swap_cnt;
    u32        coh_nonswap_cnt;
    u32        noncoh_swap_cnt;
    u32        noncoh_nonswap_cnt;

    auto sort(auto width,
              auto batch_size,
              auto reverse,
              auto dst_pck,
              auto src_pck,
              auto dst_data,
              auto src_data) {
        impl_t::sort_impl(width,
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
    auto coherent_sort(auto width,
                       auto batch_size,
                       auto reverse,
                       auto dst_pck,
                       auto src_pck,
                       auto dst_data,
                       auto src_data) {
        impl_t::sort_impl(width,
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
    auto small_sort(auto width,
                    auto batch_size,
                    auto reverse,
                    auto dst_pck,
                    auto src_pck,
                    auto dst_data,
                    auto src_data) {
        impl_t::sort_impl(width,
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
    static auto insert_indexes(auto& r, uZ fft_size, uZ coherent_size) -> br_sorter {
        auto rbo = [=](auto i) { return reverse_bit_order(i, log2i(fft_size)); };
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
        return br_sorter{{}, nullptr, coh_swap_cnt, coh_nonswap_cnt, noncoh_swap_cnt, noncoh_nonswap_cnt};
    }
};
struct br_sorter_shifted {
    const u32* idx_ptr;
    u32        swap_cnt;

    auto sort(auto width,
              auto batch_size,
              auto reverse,
              auto dst_pck,
              auto src_pck,
              auto dst_data,
              auto src_data) {
        sort_impl(width,
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
    auto small_sort(auto width,
                    auto batch_size,
                    auto reverse,
                    auto dst_pck,
                    auto src_pck,
                    auto dst_data,
                    auto src_data) {
        sort_impl(width,
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
    auto coherent_sort(auto...) {
        return inplace_src;
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

template<uZ Width>
struct br_sorter_sequential {
    static constexpr auto width = uZ_ce<Width>{};

    const u32* idx_ptr;
    u32        swap_cnt;
    u32        nonswap_cnt;

    void sequential_sort(auto dst_pck,    //
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
            bool swap = true;
            while (true) {
                for ([[maybe_unused]] auto i: stdv::iota(0U, lane_cnt)) {
                    auto dst0       = next_ptr_tup();
                    auto data0      = tupi::group_invoke(simd::cxload<src_pck, width>, dst0);
                    auto data_perm0 = simd::br_permute(data0);
                    if (!swap) {
                        tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm0);
                    } else {
                        auto dst1       = next_ptr_tup();
                        auto data1      = tupi::group_invoke(simd::cxload<src_pck, width>, dst1);
                        auto data_perm1 = simd::br_permute(data1);
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
        } else {
            auto next_ptr_tup = [&] PCX_LAINLINE {
                return [&]<uZ... Is> PCX_LAINLINE(uZ_seq<Is...>) {
                    auto idx = *idx_ptr;
                    ++idx_ptr;
                    auto mktup = [=](auto data) {
                        return tupi::make_tuple(
                            (data.get_batch_base(0) + idx * width * 2 + Is * subsize * 2)...);
                    };
                    return tupi::make_tuple(mktup(dst_data), mktup(src_data));
                }(make_uZ_seq<width>{});
            };
            bool swap = true;
            while (true) {
                for ([[maybe_unused]] auto i: stdv::iota(0U, lane_cnt)) {
                    auto [dst0, src0] = next_ptr_tup();
                    auto data0        = tupi::group_invoke(simd::cxload<src_pck, width>, src0);
                    auto data_perm0   = simd::br_permute(data0);
                    if (!swap) {
                        tupi::group_invoke(simd::cxstore<dst_pck>, dst0, data_perm0);
                    } else {
                        auto [dst1, src1] = next_ptr_tup();
                        tupi::group_invoke(simd::cxstore<dst_pck>, src1, data_perm0);
                        auto data1      = tupi::group_invoke(simd::cxload<src_pck, width>, src1);
                        auto data_perm1 = simd::br_permute(data1);
                        tupi::group_invoke(simd::cxstore<dst_pck>, src0, data_perm1);
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
    static auto insert_indexes(auto& r, uZ fft_size) -> br_sorter_sequential {
        u32 n = fft_size / width / width;
        u32 swap_cnt{};
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
        return br_sorter_sequential{nullptr, swap_cnt, n - 2 * swap_cnt};
    }
};
}    // namespace pcx::detail_
