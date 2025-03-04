
#include "common.hpp"

namespace pcx::testing {
template<>
bool test_fft<f32, 8>(uZ fft_size) {
    return run_tests<f32, 8>(f32_widths, low_k, local_tw, fft_size);
};
template bool test_fft<f32, 8>(uZ);

}    // namespace pcx::testing
