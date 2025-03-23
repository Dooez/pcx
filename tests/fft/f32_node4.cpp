#include "common.hpp"
namespace pcx::testing {
using fX                      = f32;
static constexpr uZ node_size = 4;
static constexpr auto fX_widths = f32_widths;
template<>
bool test_fft<fX, node_size>(const std::vector<std::complex<fX>>& signal,
                             const std::vector<std::complex<fX>>& check,
                             std::vector<std::complex<fX>>&       s1,
                             std::vector<std::complex<fX>>&       s2) {
    return run_tests<fX, node_size>(fX_widths, low_k, local_tw, half_tw, signal, check, s1, s2);
};
template bool test_fft<fX, node_size>(const std::vector<std::complex<fX>>&,
                                      const std::vector<std::complex<fX>>&,
                                      std::vector<std::complex<fX>>&,
                                      std::vector<std::complex<fX>>&);
}    // namespace pcx::testing
