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

namespace {
// constexpr auto permutation  = pcx::meta::val_seq<pcx::fft_permutation::normal>{};
constexpr auto permutations = pcx::meta::val_seq<pcx::fft_permutation::normal,
                                                 pcx::fft_permutation::bit_reversed,
                                                 pcx::fft_permutation::shifted>{};
template<auto Perm>
void naive_permute(auto& check) {
    using enum pcx::fft_permutation;
    if constexpr (Perm == bit_reversed) {
        return;
    } else if constexpr (Perm == normal) {
        pcxt::bit_reverse(check);
    } else if constexpr (Perm == shifted) {
        pcxt::shifted_bit_reverse(check);
    }
}
template<auto Perm>
void print_perm() {
    using enum pcx::fft_permutation;
    if constexpr (Perm == bit_reversed) {
        std::print("[BitRev]");
    } else if constexpr (Perm == normal) {
        std::print("[Normal]");
    } else if constexpr (Perm == shifted) {
        std::print("[Shiftd]");
    }
}

template<typename fX, auto Perm>
bool check_par_perm(uZ fft_size, uZ data_size, f64 freq_n) {
    constexpr auto ops = pcx::fft_options{.pt = Perm};
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
    naive_permute<Perm>(check);
    auto data_range = signal() | stdv::take(fft_size) | stdr::to<std::vector<std::span<std::complex<fX>>>>();
    fft.fft(data_range);

    print_perm<Perm>();
    std::print("[Par ][Fwd]");
    for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), signal(), check)) {
        if (!pcxt::par_check_correctness(check_v, sv, fft_size, i, ops.simd_width, ops.node_size, false))
            return false;
    }
    std::println("[Success] {}×{}×{}, width {}, node size {}.",
                 pcx::meta::types<fX>{},
                 fft_size,
                 data_size,
                 ops.simd_width,
                 ops.node_size);
    return true;
}
template<typename fX, auto... Perms>
bool check_par(uZ fft_size, uZ data_size, f64 freq_n, pcx::meta::val_seq<Perms...>) {
    return (check_par_perm<fX, Perms>(fft_size, data_size, freq_n) && ...);
}

template<typename fX, auto Perm>
bool check_parc_perm(uZ fft_size, uZ data_size, f64 freq_n) {
    constexpr auto ops = pcx::fft_options{.pt = Perm};
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
    naive_permute<Perm>(check);
    fft.fft_raw(signal_raw.data(), data_size, data_size);

    print_perm<Perm>();
    std::print("[Parc][Fwd]");
    for (auto [i, sv, check_v]: stdv::zip(stdv::iota(0U), signal(), check)) {
        if (!pcxt::par_check_correctness(check_v, sv, fft_size, i, ops.simd_width, ops.node_size, false))
            return false;
    }
    std::println("[Success] {}×{}×{}, width {}, node size {}.",
                 pcx::meta::types<fX>{},
                 fft_size,
                 data_size,
                 ops.simd_width,
                 ops.node_size);
    return true;
}
template<typename fX, auto... Perms>
bool check_parc(uZ fft_size, uZ data_size, f64 freq_n, pcx::meta::val_seq<Perms...>) {
    return (check_parc_perm<fX, Perms>(fft_size, data_size, freq_n) && ...);
}

template<typename fX, auto Perm>
bool check_seq_perm(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = Perm};
    auto           fft = pcx::fft_plan<fX, ops>(fft_size);

    auto data = std::vector<std::complex<fX>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<fX>{0, static_cast<fX>(i)}    //
                     * static_cast<fX>(2.)                      //
                     * std::numbers::pi_v<fX>                   //
                     / static_cast<fX>(2.));
    }

    auto check = data;
    fft.fft(data);
    pcxt::naive_fft(check, 16, 8);
    naive_permute<Perm>(check);
    print_perm<Perm>();
    std::print("[Seq ][Fwd]");
    if (!pcxt::check_correctness(check, data, 16, 8, true, true, true))
        return false;

    naive_permute<Perm>(check);
    pcxt::naive_reverse(check, 16, 8);
    fft.ifft(data);
    print_perm<Perm>();
    std::print("[Seq ][Rev]");
    if (!pcxt::check_correctness(check, data, 16, 8, true, true, true))
        return false;
    //
    return true;
}
template<typename fX, auto... Perms>
bool check_seq(uZ fft_size, pcx::meta::val_seq<Perms...>) {
    return (check_seq_perm<fX, Perms>(fft_size) && ...);
}
}    // namespace

int main() {
    size_t fft_size = 2;
    while (fft_size < 2048 * 256) {
        if (!check_par<f32>(fft_size, 31, 13.001, permutations))
            return -1;
        if (!check_parc<f32>(fft_size, 31, 13.001, permutations))
            return -1;
        if (!check_seq<f32>(fft_size, permutations))
            return -1;
        fft_size *= 2;
    }
    fft_size = 4;
    while (fft_size < 2048 * 128) {
        if (!check_par<f64>(fft_size, 31, 13.001, permutations))
            return -1;
        if (!check_parc<f64>(fft_size, 31, 13.001, permutations))
            return -1;
        if (!check_seq<f64>(fft_size, permutations))
            return -1;
        fft_size *= 2;
    }

    return 0;
}
