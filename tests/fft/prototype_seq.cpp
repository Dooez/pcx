#include "../../tests/fft/impl-test.hpp"
namespace pcx::testing {
using fX                      = ${FLOAT_TYPE};
static constexpr uZ width     = ${VECTOR_WIDTH};
static constexpr uZ node_size = ${NODE_SIZE};
template<>
bool test_seq<fX, width, node_size>(const std::vector<std::complex<fX>>& signal,
                                    const chk_t<fX>&                     chk_fwd,
                                    const chk_t<fX>&                     chk_rev,
                                    std::vector<std::complex<fX>>&       s1,
                                    std::vector<fX>&                     twvec,
                                    bool                                 local_check,
                                    bool                                 fwd,
                                    bool                                 rev,
                                    bool                                 inplace,
                                    bool                                 external) {
    return seq_run_tests<fX, width, node_size>(low_k,
                                               local_tw,
                                               half_tw,
                                               perm_types,
                                               signal,
                                               chk_fwd,
                                               chk_rev,
                                               s1,
                                               twvec,
                                               local_check,
                                               fwd,
                                               rev,
                                               inplace,
                                               external);
};
template bool test_seq<fX, width, node_size>(const std::vector<std::complex<fX>>&,
                                             const chk_t<fX>&,
                                             const chk_t<fX>&,
                                             std::vector<std::complex<fX>>&,
                                             std::vector<fX>&,
                                             bool,
                                             bool,
                                             bool,
                                             bool,
                                             bool);
}    // namespace pcx::testing
