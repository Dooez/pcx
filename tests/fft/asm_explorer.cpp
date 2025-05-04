#include "pcx/include/fft_impl.hpp"

#include <cmath>
#include <print>
#include <vector>

namespace pcx::testing {
void foo(f32* data, f32* tw) {
    using coh = pcx::detail_::contiguous_subtransform<8, f32, 16>;

    auto           tw_data = detail_::tw_data_t<f32, false>{tw};
    constexpr auto src_pck = cxpack<16, f32>{};
    constexpr auto dst_pck = cxpack<1, f32>{};

    // PCX_AINLINE static void perform(cxpack_for<T> auto         dst_pck,
    //                                 cxpack_for<T> auto         src_pck,
    //                                 meta::any_ce_of<bool> auto lowk,
    //                                 meta::any_ce_of<bool> auto half_tw,
    //                                 uZ                         data_size,
    //                                 T*                         dest_ptr,
    //                                 meta::maybe_ce_of<uZ> auto align_node,
    //                                 tw_data_for<T> auto&       tw);
    // coh::perform(dst_pck, src_pck, std::false_type{}, std::true_type{}, 2048, data, uZ_ce<2>{}, tw_data);

    // PCX_AINLINE static auto single_load(cxpack_for<T> auto         dst_pck,
    //                                     cxpack_for<T> auto         src_pck,
    //                                     meta::any_ce_of<bool> auto lowk,
    //                                     meta::any_ce_of<bool> auto half_tw,
    //                                     T*                         data_ptr,
    //                                     const T*                   src_ptr,
    //                                     tw_data_t<T, LocalTw>&     tw_data) {
    // coh::single_load(dst_pck, src_pck, std::false_type{}, std::true_type{}, data, data, tw_data);

    using fimpl = pcx::detail_::transform<8, f32, 16>;
    fimpl::template perform<1, 1>(4096, data, tw_data, std::true_type{}, std::true_type{});
}
}    // namespace pcx::testing

int main() {
    return 0;
}
