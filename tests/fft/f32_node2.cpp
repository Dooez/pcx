#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f32, 2>(uZ fft_size) {
    return run_tests<f32, 2>(f32_widths, low_k, local_tw, half_tw, fft_size);
};
template bool test_fft<f32, 2>(uZ);

}    // namespace pcx::testing
