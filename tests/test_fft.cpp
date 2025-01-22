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
void naive_fft(std::vector<std::complex<fX>>& data) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");

    constexpr auto vec_with = pcx::simd::max_width<fX>;

    auto fft_size = 2;
    auto step     = rsize / 2;
    auto n_groups = 1;
    while (step >= 1) {
        if (step == vec_with * 8) {
            // return;
        }
        if (step == 8) {
            // return;
        }
        if (step == 2) {
            // return;
        }
        for (uZ k = 0; k < n_groups; ++k) {
            uZ   start = k * step * 2;
            auto rk    = pcx::detail_::reverse_bit_order(k, log2i(fft_size) - 1);
            auto tw    = pcx::detail_::wnk<fX>(fft_size, rk);
            for (uZ i = 0; i < step; ++i) {
                pcxt::btfly(&data[start + i], &data[start + i + step], tw);    //
            }
        }
        step /= 2;
        n_groups *= 2;
        fft_size *= 2;
    }
}
template void naive_fft(std::vector<std::complex<f32>>& data);
template void naive_fft(std::vector<std::complex<f64>>& data);

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
auto make_subtform_tw(uZ max_size,      //
                      uZ start_size,    // = 1
                      uZ start_idx,     // = 0
                      uZ vec_width,
                      uZ node_size) -> std::vector<std::complex<fX>> {
    auto tw_vec = std::vector<std::complex<fX>>();

    auto element_count = max_size / start_size;

    auto insert_tw = [&](uZ size, uZ i) {
        auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
        auto tw = pcx::detail_::wnk<fX>(size, rk);
        tw_vec.push_back(tw);
    };
    auto single_load_size = vec_width * node_size;
    auto size             = start_size;
    while (max_size / size > single_load_size * node_size) {
        for (auto i: stdv::iota(0U, size)) {
            for (auto pow2: stdv::iota(0U, log2i(node_size))) {
                auto l_start = start_idx * powi(2, pow2);
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    auto l_i = l_start + i * powi(2, pow2) + k;
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
auto make_tw_vec(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<fX>> {
    return make_subtform_tw<fX>(fft_size, 1, 0, vec_width, node_size);
    // auto tw_vec    = std::vector<std::complex<fX>>();
    // auto insert_tw = [&](uZ size, uZ i) {
    //     auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
    //     auto tw = pcx::detail_::wnk<fX>(size, rk);
    //     tw_vec.push_back(tw);
    // };
    // auto single_load_size = vec_width * node_size;
    // auto size             = 2UZ;
    // while (fft_size / size >= single_load_size * node_size) {
    //     for (auto i: stdv::iota(0U, size / 2)) {
    //         for (auto ns: stdv::iota(0U, log2i(node_size))) {
    //             for (auto k: stdv::iota(0U, powi(2, ns))) {
    //                 insert_tw(size * powi(2, ns), k + i * powi(2, ns));
    //             }
    //         }
    //     }
    //     size *= node_size;
    // }
    // for (uZ tp: stdv::iota(0U, log2i(node_size))) {
    //     if (fft_size / (size * powi(2, tp)) != single_load_size) {
    //         continue;
    //     }
    //     auto local_node = powi(2, tp + 1);
    //     for (auto i: stdv::iota(0U, size / 2)) {
    //         for (auto pow2: stdv::iota(0U, log2i(local_node))) {
    //             for (auto k: stdv::iota(0U, powi(2, pow2))) {
    //                 insert_tw(size * powi(2, pow2), k + i * powi(2, pow2));
    //             }
    //         }
    //     }
    //     size *= local_node;
    //     break;
    // }
    // for (auto i_sl: stdv::iota(0U, fft_size / single_load_size)) {
    //     auto start_offset = i_sl;
    //     auto fft_size     = size;
    //     uZ   n_tw         = 1;
    //     // uZ   n_tw     = VecCount;
    //     while (n_tw <= single_load_size / 2) {
    //         for (auto i: stdv::iota(0U, n_tw)) {
    //             insert_tw(fft_size, start_offset + i);
    //         }
    //         start_offset *= 2;
    //         fft_size *= 2;
    //         n_tw *= 2;
    //     }
    // }
    // return tw_vec;
}
template auto make_tw_vec<f32>(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<f32>>;
template auto make_tw_vec<f64>(uZ fft_size, uZ vec_width, uZ node_size) -> std::vector<std::complex<f64>>;


// NOLINTEND (*pointer-arithmetic*)
