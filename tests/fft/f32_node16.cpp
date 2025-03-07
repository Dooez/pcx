
#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f32, 16>(uZ fft_size, f64 freq_n) {
    return run_tests<f32, 16>(f32_widths, low_k, local_tw, half_tw, fft_size, freq_n);
};
template bool test_fft<f32, 16>(uZ, f64);

}    // namespace pcx::testing
