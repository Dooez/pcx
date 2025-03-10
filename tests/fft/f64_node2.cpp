#include "common.hpp"
namespace pcx::testing {
using fX                      = f64;
static constexpr uZ node_size = 2;
template<>
bool test_fft<fX, node_size>(const std::vector<std::complex<fX>>& signal,
                             const std::vector<std::complex<fX>>& check) {
    return run_tests<fX, node_size>(f64_widths, low_k, local_tw, half_tw, signal, check);
};
template bool test_fft<fX, node_size>(const std::vector<std::complex<fX>>&,
                                      const std::vector<std::complex<fX>>&);

}    // namespace pcx::testing
