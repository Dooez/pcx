#include "common.hpp"
#include "pcx/fft.hpp"
#include "pcx/par_fft.hpp"

#include <generator>
#include <print>
namespace stdv = std::views;
namespace stdr = std::ranges;

using pcx::f32;
using pcx::f64;
using pcx::uZ;
namespace pcxt = pcx::testing;

template<typename fX>
bool check_parc_br(uZ fft_size, uZ data_size, f64 freq_n) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::bit_reversed};
    auto           fft = pcx::par_fft_plan<fX, ops>(fft_size);

    auto signal_raw = std::vector<std::complex<fX>>(fft_size * data_size);
    auto s1_raw     = signal_raw;

    auto signal = [&](uZ i = 0) -> std::generator<std::span<std::complex<fX>>> {
        while (true)
            co_yield {signal_raw.data() + data_size * (i++), data_size};
    };
    auto check = std::vector<std::complex<fX>>(fft_size);
    for (auto [i, v, vcf]: stdv::zip(stdv::iota(0U), signal(), check)) {
        auto cx = std::exp(std::complex<fX>(0, 1)                 //
                           * static_cast<fX>(2)                   //
                           * static_cast<fX>(std::numbers::pi)    //
                           * static_cast<fX>(i)                   //
                           * static_cast<fX>(freq_n)              //
                           / static_cast<fX>(fft_size));

        vcf = cx;
        stdr::fill(v, cx);
    }

    pcxt::naive_fft(check, 8, 8);
    fft.fft_raw(signal_raw.data(), data_size, data_size);

    for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), signal(), check)) {
        if (!pcxt::par_check_correctness(check_v, sv, fft_size, i, ops.simd_width, ops.node_size, false))
            return false;
    }
    std::println("[Success][Parc] size: {}", fft_size);
    return true;
}


template<typename fX>
bool check_br(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::bit_reversed};
    auto           fft = pcx::fft_plan<fX, ops>(fft_size);

    auto data = std::vector<std::complex<fX>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<fX>{0, static_cast<fX>(i)}    //
                     * static_cast<fX>(2.)                      //
                     * std::numbers::pi_v<fX>                   //
                     / static_cast<fX>(2.));
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    std::print("[BitRev][Fwd]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    pcx::testing::naive_reverse(data_c, 16, 8);
    fft.ifft(data);
    std::print("[BitRev][rev]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    return true;
}

template<typename fX>
bool check_normal(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::normal};
    auto           fft = pcx::fft_plan<fX, ops>(fft_size);

    auto data = std::vector<std::complex<fX>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<fX>{0, static_cast<fX>(i)}    //
                     * static_cast<fX>(2.)                      //
                     * std::numbers::pi_v<fX>                   //
                     / static_cast<fX>(2.));
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    pcx::testing::bit_reverse(data_c);
    std::print("[Normal][Fwd]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    pcx::testing::bit_reverse(data_c);
    pcx::testing::naive_reverse(data_c, 16, 8);
    fft.ifft(data);
    std::print("[Normal][Rev]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    return true;
}

template<typename fX>
bool check_shiftd(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::shifted};
    auto           fft = pcx::fft_plan<fX, ops>(fft_size);

    auto data = std::vector<std::complex<fX>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<fX>{0, static_cast<fX>(i)}    //
                     * static_cast<fX>(2.)                      //
                     * std::numbers::pi_v<fX>                   //
                     / static_cast<fX>(2.));
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    pcx::testing::shifted_bit_reverse(data_c);
    std::print("[Shiftd][Fwd]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    pcx::testing::shifted_bit_reverse(data_c);
    pcx::testing::naive_reverse(data_c, 16, 8);
    fft.ifft(data);
    std::print("[Shiftd][Rev]");
    if (!pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true))
        return false;
    return true;
}

int main() {
    size_t fft_size = 2;
    while (fft_size < 2048 * 256) {
        if (!check_parc_br<f32>(fft_size, 31, 13.001))
            return -1;
        // if (!check_br<f32>(fft_size))
        //     return -1;
        // if (!check_normal<f32>(fft_size))
        //     return -1;
        // if (!check_shiftd<f32>(fft_size))
        //     return -1;
        fft_size *= 2;
    }
    fft_size = 4;
    while (fft_size < 2048 * 128) {
        if (!check_parc_br<f32>(fft_size, 31, 13.001))
            return -1;
        // if (!check_br<f64>(fft_size))
        //     return -1;
        // if (!check_normal<f64>(fft_size))
        //     return -1;
        // if (!check_shiftd<f64>(fft_size))
        //     return -1;
        fft_size *= 2;
    }

    return 0;
}
