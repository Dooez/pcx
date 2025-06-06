#include "../../tests/fft/impl-test.hpp"
namespace pcx::testing {
using fX                      = ${FLOAT_TYPE};
static constexpr uZ width     = ${VECTOR_WIDTH};
static constexpr uZ node_size = ${NODE_SIZE};
template<>
bool test_par<fX, width, node_size>(const std_vec2d<fX>& signal,
                                    std_vec2d<fX>&       s1,
                                    const chk_t<fX>&     chk_fwd,
                                    const chk_t<fX>&     chk_rev,
                                    std::vector<fX>&     twvec,
                                    bool                 local_check,
                                    bool                 fwd,
                                    bool                 rev,
                                    bool                 inplace,
                                    bool                 external) {
    return par_run_tests<fX, width, node_size>(low_k,
                                               local_tw,
                                               perm_types,
                                               signal,
                                               s1,
                                               chk_fwd,
                                               chk_rev,
                                               twvec,
                                               local_check,
                                               fwd,
                                               rev,
                                               inplace,
                                               external);
};
template bool test_par<fX, width, node_size>(const std_vec2d<fX>&,
                                             std_vec2d<fX>&,
                                             const chk_t<fX>&,
                                             const chk_t<fX>&,
                                             std::vector<fX>&,
                                             bool,
                                             bool,
                                             bool,
                                             bool,
                                             bool);
}    // namespace pcx::testing
