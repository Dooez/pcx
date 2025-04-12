#include "common.hpp"
namespace pcx::testing {
using fX                  = f64;
static constexpr uZ width = 4;
template<>
bool test_fft<fX, width>(const std::vector<std::complex<fX>>& signal,
                         const std::vector<std::complex<fX>>& chk_fwd,
                         const std::vector<std::complex<fX>>& chk_rev,
                         std::vector<std::complex<fX>>&       s1,
                         std::vector<fX>&                     twvec) {
    return run_tests<fX, width>(node_sizes, low_k, local_tw, half_tw, signal, chk_fwd, chk_rev, s1, twvec);
};
template bool test_fft<fX, width>(const std::vector<std::complex<fX>>&,
                                  const std::vector<std::complex<fX>>&,
                                  const std::vector<std::complex<fX>>&,
                                  std::vector<std::complex<fX>>&,
                                  std::vector<fX>&);
}    // namespace pcx::testing
