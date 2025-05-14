#include "common.hpp"
#include "pcx/fft.hpp"

#include <print>
namespace stdv = std::views;
namespace stdr = std::ranges;

using pcx::f32;
using pcx::f64;
using pcx::uZ;

bool check_br(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::bit_reversed};
    auto           fft = pcx::fft_plan<float, ops>(fft_size);

    auto data = std::vector<std::complex<float>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<float>{0, static_cast<float>(i)}    //
                     * 2.F                                            //
                     * std::numbers::pi_v<float>                      //
                     / 2.F);
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    std::print("[BitRvd]");
    return pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true);
}

bool check_normal(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::normal};
    auto           fft = pcx::fft_plan<float, ops>(fft_size);

    auto data = std::vector<std::complex<float>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<float>{0, static_cast<float>(i)}    //
                     * 2.F                                            //
                     * std::numbers::pi_v<float>                      //
                     / 2.F);
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    pcx::testing::bit_reverse(data_c);
    std::print("[Normal]");
    return pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true);
}

bool check_shiftd(uZ fft_size) {
    constexpr auto ops = pcx::fft_options{.pt = pcx::fft_permutation::shifted};
    auto           fft = pcx::fft_plan<float, ops>(fft_size);

    auto data = std::vector<std::complex<float>>(fft_size);
    for (auto [i, v]: stdv::enumerate(data)) {
        v = std::exp(std::complex<float>{0, static_cast<float>(i)}    //
                     * 2.F                                            //
                     * std::numbers::pi_v<float>                      //
                     / 2.F);
    }

    auto data_c = data;
    fft.fft(data);
    pcx::testing::naive_fft(data_c, 16, 8);
    pcx::testing::shifted_bit_reverse(data_c);
    std::print("[Shiftd]");
    return pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true);
}

int main() {
    size_t fft_size = 2;
    while (fft_size < 2048 * 2048) {
        if (!check_br(fft_size))
            return -1;
        if (!check_normal(fft_size))
            return -1;
        if (!check_shiftd(fft_size))
            return -1;
        fft_size *= 2;
    }

    return 0;
}
