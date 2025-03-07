
#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f64, 2>(uZ fft_size, f64 freq_n) {
    return run_tests<f64, 2>(f64_widths, low_k, local_tw, half_tw, fft_size, freq_n);
};
template bool test_fft<f64, 2>(uZ, f64);

}    // namespace pcx::testing
