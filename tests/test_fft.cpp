// #define PCX_AVX512
#include "test_fft.h"

#include "pcx/include/fft_impl.hpp"

#include <complex>
#include <cstddef>
#include <print>
#include <ranges>
#include <vector>
namespace stdv = std::views;

// NOLINTBEGIN (*pointer-arithmetic*)

using namespace pcx;
constexpr auto is_pow_of_two(uZ n) {
    return n > 0 && (n & (n - 1)) == 0;
}

template<typename T>
struct is_std_complex : std::false_type {};
template<typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template<typename fX>
void naive_fft(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");

    // constexpr auto vec_width = pcx::simd::max_width<fX>;


    auto fft_size         = 1;
    auto step             = rsize / 2;
    auto n_groups         = 1;
    auto single_load_size = vec_width * node_size;
    // while (step >= 1) {
    //     if (!(rsize / (fft_size * node_size) >= single_load_size)) {
    //         break;
    //     }
    //     for (auto l: stdv::iota(0U, log2i(node_size))) {
    //         fft_size *= 2;
    //         for (uZ k = 0; k < n_groups; ++k) {
    //             uZ start = k * step * 2;
    //             // auto rk    = pcx::detail_::reverse_bit_order(k, log2i(fft_size) - 1);
    //             auto tw = pcx::detail_::wnk_br<fX>(fft_size, k);
    //             for (uZ i = 0; i < step; ++i) {
    //                 pcxt::btfly(&data[start + i], &data[start + i + step], tw);    //
    //             }
    //         }
    //         step /= 2;
    //         n_groups *= 2;
    //     }
    // }
    // return;
    while (step >= 1) {
        if (step == vec_width * node_size / 2) {    // skip single load
            // break;
        }
        if (step < vec_width) {
            // return;
        }
        fft_size *= 2;
        for (uZ k = 0; k < n_groups; ++k) {
            uZ   start = k * step * 2;
            auto tw    = pcx::detail_::wnk_br<fX>(fft_size, k);
            for (uZ i = 0; i < step; ++i) {
                pcxt::btfly(&data[start + i], &data[start + i + step], tw);    //
            }
        }
        // if (n_groups >= powi(2, 2))
        //     break;
        step /= 2;
        n_groups *= 2;
        // break;
    }
}
template void naive_fft(std::vector<std::complex<f32>>& data, uZ, uZ);
template void naive_fft(std::vector<std::complex<f64>>& data, uZ, uZ);

template<uZ To, uZ From, typename R>
    requires stdr::random_access_range<R> && is_std_complex<stdr::range_value_t<R>>::value
void repack(R& data) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");
    using cx_t     = stdr::range_value_t<R>;
    using T        = cx_t::value_type;
    auto* data_ptr = reitnerpret_cast<T*>(data.begin());

    constexpr auto vec_width = simd::max_width<T>;
    for (auto i: stdv::iota(0U, data.size() / vec_width)) {
        auto ptr = data_ptr + i * vec_width * 2;
        auto x   = simd::cxload<From, vec_width>(ptr);
        simd::cxstore<To>(ptr, simd::repack<To>(x));
    }
}
template<typename fX>
auto make_subtform_tw_lok(uZ max_size,      //
                          uZ start_size,    // = 1
                          uZ vec_width,
                          uZ node_size) {
    auto tw_vec = std::vector<std::complex<fX>>();
    // tw_vec.reserve(4096);
    auto insert_tw = [&](uZ size, uZ k) {
        // auto rk = pcx::detail_::reverse_bit_order(k, log2i(size) - 1);
        // auto tw = pcx::detail_::wnk<fX>(size, rk);
        auto tw = pcx::detail_::wnk_br<fX>(size, k);
        tw_vec.push_back(tw);
    };

    auto single_load_size = vec_width * node_size;
    auto element_count    = max_size / start_size;
    // auto size             = start_size;

    auto lsize          = element_count / single_load_size;
    auto slog           = log2i(lsize);
    auto a              = slog / log2i(node_size);
    auto b              = a * log2i(node_size);
    auto pre_align_node = powi(2, slog - b);

    auto size = start_size;

    // pre-align
    for (uZ align_p2: stdv::iota(0U, log2i(node_size) - 1)) {
        auto local_node = powi(2, align_p2 + 1);
        if (local_node != pre_align_node)
            continue;
        for (auto li: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(local_node))) {
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_k = li * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_k);
                }
            }
        }
        size *= local_node;
    }


    uZ i = 0;
    while (max_size / (size * node_size) >= single_load_size) {
        // for (auto i: stdv::iota(0U, size)) {
        for (; i < size; ++i) {
            for (auto pow2: stdv::iota(0U, log2i(node_size))) {
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_k = i * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_k);
                }
            }
        }
        size *= node_size;
    }

    // post-align
    auto post_align_node = 0;
    for (uZ align_p2: stdv::iota(0U, log2i(node_size) - 1)) {
        auto local_node = powi(2, align_p2 + 1);
        if (local_node != post_align_node)
            continue;
        for (auto li: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(local_node))) {
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_k = li * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_k);
                }
            }
        }
        size *= local_node;
    }

    auto x = 0;
    for (auto i_sl: stdv::iota(0U, size * 2)) {
        auto start_offset = i_sl;

        auto fft_size = size;
        for (auto pow2: stdv::iota(0U, log2i(node_size))) {
            for (auto k: stdv::iota(0U, powi(2, pow2))) {
                if (k % 2 == 1) {
                    continue;
                }
                insert_tw(fft_size * 2, start_offset + k);
            }
            start_offset *= 2;
            fft_size *= 2;
        }
        // auto fft_size   = size * node_size;
        uZ tw_per_vec = 2;
        while (tw_per_vec <= vec_width) {
            auto tw_idx = start_offset;
            for (auto i_vec: stdv::iota(0U, node_size / 2)) {
                for (auto i_tw: stdv::iota(0U, tw_per_vec)) {
                    insert_tw(fft_size * 2, tw_idx);
                    ++tw_idx;
                }
            }
            start_offset *= 2;
            tw_per_vec *= 2;
            fft_size *= 2;
        }
    }
    return tw_vec;
};

template<typename fX>
auto make_subtform_tw(uZ max_size,      //
                      uZ start_size,    // = 1
                      uZ start_idx,     // = 0
                      uZ vec_width,
                      uZ node_size) -> std::vector<std::complex<fX>> {
    auto tw_vec = std::vector<std::complex<fX>>();

    auto element_count = max_size / start_size;

    auto insert_tw = [&](uZ size, uZ k) {
        // auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
        // auto tw = pcx::detail_::wnk<fX>(size, rk);
        auto tw = pcx::detail_::wnk_br<fX>(size, k);
        tw_vec.push_back(tw);
    };
    auto single_load_size = vec_width * node_size;

    auto lsize      = element_count / single_load_size;
    auto slog       = log2i(lsize);
    auto a          = slog / log2i(node_size);
    auto b          = a * log2i(node_size);
    auto align_node = powi(2, slog - b);

    auto size = start_size;

    // pre-align
    for (uZ align_p2: stdv::iota(0U, log2i(node_size) - 1)) {
        auto local_node = powi(2, align_p2 + 1);
        if (local_node != align_node)
            continue;
        for (auto li: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(local_node))) {
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_k = li * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_k);
                }
            }
        }
        size *= local_node;
    }

    while (max_size / size > single_load_size * node_size) {
        for (auto i: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(node_size))) {
                auto l_start = start_idx * powi(2, pow2);
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_i = l_start + i * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_i);
                }
            }
        }
        size *= node_size;
        start_idx *= node_size;
    }
    for (uZ align_p2: stdv::iota(0U, log2i(node_size))) {
        if (max_size / (size * powi(2, align_p2 + 1)) != single_load_size) {
            continue;
        }
        auto local_node = powi(2, align_p2 + 1);
        for (auto i: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(local_node))) {
                auto l_start = start_idx * powi(2, pow2);
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_i = l_start + i * powi(2, pow2) + k;
                    if (k % 2 == 1) {
                        continue;
                    }
                    insert_tw(size * powi(2, pow2 + 1), l_i);
                }
            }
        }
        size *= local_node;
        start_idx *= local_node;
    }
    for (auto i_sl: stdv::iota(0U, element_count / single_load_size)) {
        auto start_offset = i_sl + start_idx;
        auto fft_size     = size * 2;

        for (auto i_node: stdv::iota(0U, log2i(node_size))) {
            for (auto k: stdv::iota(0U, powi(2, i_node))) {
                if (k % 2 == 1) {
                    continue;
                }
                insert_tw(fft_size, start_offset + k);
            }
            start_offset *= 2;
            fft_size *= 2;
        }
        uZ tw_per_vec = 2;
        uZ pow2       = 0;
        while (tw_per_vec <= vec_width) {
            uZ tw_idx = start_offset;
            for (auto i_vec: stdv::iota(0U, node_size / 2)) {
                for (auto i_tw: stdv::iota(0U, tw_per_vec)) {
                    insert_tw(fft_size, tw_idx);
                    ++tw_idx;
                }
            }
            start_offset *= 2;
            tw_per_vec *= 2;
            fft_size *= 2;
        }
    }
    return tw_vec;
}
template<typename fX>
auto make_tw_vec(uZ fft_size, uZ vec_width, uZ node_size, bool low_k) -> std::vector<std::complex<fX>> {
    if (low_k)
        return make_subtform_tw_lok<fX>(fft_size, 1, vec_width, node_size);
    else
        return make_subtform_tw<fX>(fft_size, 1, 0, vec_width, node_size);
}
template auto make_tw_vec<f32>(uZ fft_size, uZ vec_width, uZ node_size, bool low_k)
    -> std::vector<std::complex<f32>>;
template auto make_tw_vec<f64>(uZ fft_size, uZ vec_width, uZ node_size, bool low_k)
    -> std::vector<std::complex<f64>>;

// template<typename fX>
// auto make_tw_vec_lok(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<fX>> {
//     return make_subtform_tw_lok<fX>(fft_size, 1, vec_width, node_size);
// }
// template auto make_tw_vec_lok<f32>(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<f32>>;
// template auto make_tw_vec_lok<f64>(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<f64>>;


// NOLINTEND (*pointer-arithmetic*)
