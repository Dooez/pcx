#include "common.hpp"
#include "pcx/fft.hpp"

#include <print>
namespace stdv = std::views;
namespace stdr = std::ranges;

using pcx::f32;
using pcx::f64;
using pcx::uZ;

int main() {
    size_t fft_size = 2;

    while (fft_size < 2048 * 2048) {
        auto fft  = pcx::fft_plan<float>(fft_size);
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
        pcx::testing::check_correctness(data_c, data, 16, 8, true, true, true);
        // for (auto v: data) {
        //     std::print("{:.2f} ", abs(v));
        // }
        fft_size *= 2;
    }

    return 0;
}
