
#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f32, 4>(uZ fft_size, uZ freq_n) {
    return run_tests<f32, 4>(f32_widths, low_k, local_tw, half_tw, fft_size, freq_n);
};
template bool test_fft<f32, 4>(uZ, uZ);

}    // namespace pcx::testing
