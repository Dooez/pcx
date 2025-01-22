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

template<typename R>
    requires stdr::random_access_range<R> && is_std_complex<stdr::range_value_t<R>>::value
void naive_fft(R& data) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");
    using cx_t              = stdr::range_value_t<R>;
    using T                 = cx_t::value_type;
    constexpr auto vec_with = pcx::simd::max_width<T>;

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
            auto tw    = pcx::detail_::wnk<T>(fft_size, rk);
            for (uZ i = 0; i < step; ++i) {
                pcxt::btfly(&data[start + i], &data[start + i + step], tw);    //
            }
        }
        step /= 2;
        n_groups *= 2;
        fft_size *= 2;
    }
}
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
auto make_tw_vec2(uZ fft_size, uZ node_size = 8) {
    auto tw_vec    = std::vector<std::complex<fX>>();
    auto insert_tw = [&](uZ size, uZ i) {
        auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
        auto tw = pcx::detail_::wnk<fX>(size, rk);
        tw_vec.push_back(tw);
    };
    constexpr auto vec_size         = pcx::simd::max_width<fX>;
    auto           single_load_size = vec_size * node_size;
    auto           size             = 2UZ;
    while (fft_size / size >= single_load_size * node_size) {
        for (auto i: stdv::iota(0U, size / 2)) {
            for (auto ns: stdv::iota(0U, log2i(node_size))) {
                for (auto k: stdv::iota(0U, powi(2, ns))) {
                    insert_tw(size * powi(2, ns), k + i * powi(2, ns));
                }
            }
        }
        size *= node_size;
    }
    for (uZ tp: stdv::iota(0U, log2i(node_size))) {
        if (fft_size / (size * powi(2, tp)) != single_load_size) {
            continue;
        }
        auto local_node = powi(2, tp + 1);
        for (auto i: stdv::iota(0U, size / 2)) {
            for (auto pow2: stdv::iota(0U, log2i(local_node))) {
                for (auto k: stdv::iota(0U, powi(2, pow2))) {
                    insert_tw(size * powi(2, pow2), k + i * powi(2, pow2));
                }
            }
        }
        size *= local_node;
        break;
    }
    for (auto i_sl: stdv::iota(0U, fft_size / single_load_size)) {
        auto start_offset = i_sl;
        auto fft_size     = size;
        uZ   n_tw         = 1;
        // uZ   n_tw     = VecCount;
        while (n_tw <= single_load_size / 2) {
            for (auto i: stdv::iota(0U, n_tw)) {
                insert_tw(fft_size, start_offset + i);
            }
            start_offset *= 2;
            fft_size *= 2;
            n_tw *= 2;
        }
    }
    return tw_vec;
}

void bit_reverse_sort(stdr::random_access_range auto& range) {
    auto rsize = stdr::size(range);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Range size is not a power of two.");
    auto depth = log2i(rsize);
    for (auto i: stdv::iota(0U, rsize)) {
        auto irev = pcx::detail_::reverse_bit_order(i, depth);
        if (i < irev)
            std::swap(range[i], range[irev]);
    }
}

template<typename fX>
int test_subtranform(uZ fft_size) {
    uZ   freq_n  = fft_size / 2;
    auto datavec = [=]() {
        auto vec = std::vector<std::complex<fX>>(fft_size);
        for (auto [i, v]: stdv::enumerate(vec)) {
            v = std::exp(std::complex<fX>(0, 1)                 //
                         * static_cast<fX>(2)                   //
                         * static_cast<fX>(std::numbers::pi)    //
                         * static_cast<fX>(i)                   //
                         * static_cast<fX>(freq_n)              //
                         / static_cast<fX>(fft_size));
        }
        return vec;
    }();
    auto datavec2 = datavec;
    naive_fft(datavec);

    constexpr auto vec_width = pcx::simd::max_width<fX>;
    constexpr auto vec_count = 8;

    using fimpl = pcx::detail_::subtransform<vec_count, fX, vec_width>;
    auto twvec  = make_tw_vec2<fX>(fft_size);
    // template<uZ DestPackSize, uZ SrcPackSize, bool LowK>
    auto* data_ptr = reinterpret_cast<fX*>(datavec2.data());
    auto* tw_ptr   = reinterpret_cast<fX*>(twvec.data());
    fimpl::template perform<1, 1, false>(2, fft_size, data_ptr, tw_ptr);
    auto subtform_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
                                       [](auto v) { return std::get<0>(v) != std::get<1>(v); });

    if (subtform_error) {
        std::println("error during subtform of size {}", fft_size);
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2) | stdv::take(999)) {
            std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                         i,
                         (naive),
                         (pcx),
                         (naive - pcx));
            // }
        }
        return -1;
    }
    std::println("successful during subtform of size {}", fft_size);
    return 0;
}
int test_subtranform_f32(uZ fft_size) {
    return test_subtranform<f32>(fft_size);
}
int test_subtranform_f64(uZ fft_size) {
    return test_subtranform<f64>(fft_size);
}

// NOLINTEND (*pointer-arithmetic*)
