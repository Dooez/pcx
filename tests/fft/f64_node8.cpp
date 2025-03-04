

#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f64, 8>(uZ fft_size) {
    return run_tests<f64, 8>(f64_widths, low_k, local_tw, fft_size);
};
template bool test_fft<f64, 8>(uZ);

}    // namespace pcx::testing
