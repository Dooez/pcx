#include "impl-test.hpp"
namespace pcx::testing {
using fX                  = f32;
static constexpr uZ width = 8;
template<>
bool test_parc<fX, width>(parc_data<const fX> signal,
                          parc_data<fX>       s1,
                          uZ                  data_size,
                          const chk_t<fX>&    chk_fwd,
                          const chk_t<fX>&    chk_rev,
                          std::vector<fX>&    twvec,
                          bool                local_check,
                          bool                fwd,
                          bool                rev,
                          bool                inplace,
                          bool                external) {
    return parc_run_tests<fX, width>(node_sizes,
                                     low_k,
                                     local_tw,
                                     perm_types,
                                     signal,
                                     s1,
                                     data_size,
                                     chk_fwd,
                                     chk_rev,
                                     twvec,
                                     local_check,
                                     fwd,
                                     rev,
                                     inplace,
                                     external);
};
template bool test_parc<fX, width>(parc_data<const fX>,
                                   parc_data<fX>,
                                   uZ,
                                   const chk_t<fX>&,
                                   const chk_t<fX>&,
                                   std::vector<fX>&,
                                   bool,
                                   bool,
                                   bool,
                                   bool,
                                   bool);
}    // namespace pcx::testing
