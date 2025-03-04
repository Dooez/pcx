

#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f64, 16>(uZ fft_size) {
    return run_tests<f64, 16>(f64_widths, low_k, local_tw, half_tw, fft_size);
};
template bool test_fft<f64, 16>(uZ);

}    // namespace pcx::testing
