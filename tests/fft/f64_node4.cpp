

#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f64, 4>(uZ fft_size) {
    return run_tests<f64, 4>(f64_widths, low_k, local_tw, fft_size);
};
template bool test_fft<f64, 4>(uZ);

}    // namespace pcx::testing
