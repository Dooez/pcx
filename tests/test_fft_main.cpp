#include "test_fft.h"

#include <print>

// void foo(f32* data_ptr, f32* tw_ptr) {
//     using fimpl = pcx::detail_::subtransform<16, f32, 16>;
//     fimpl::template perform<1, 1, false>(2048, data_ptr, tw_ptr);
// }

int main() {
    constexpr auto exec_sl_test =
        []<uZ... NodeSizes, uZ... VecWidth, typename fX>(std::index_sequence<NodeSizes...>,
                                                         std::index_sequence<VecWidth...>,
                                                         pcx::meta::t_id<fX>) {
            auto ns_passed = [=]<uZ NS>(uZ_ce<NS>) {
                return ((test_single_load<fX, VecWidth, NS>() == 0) && ...);
            };
            auto passed = (ns_passed(uZ_ce<NodeSizes>{}) && ...);
            return passed;
        };
    constexpr auto exec_test =
        []<uZ... NodeSizes, uZ... VecWidth, bool... low_k, bool... local_tw, typename fX>(
            std::index_sequence<NodeSizes...>,
            std::index_sequence<VecWidth...>,
            pcx::meta::val_seq<low_k...>,
            pcx::meta::val_seq<local_tw...>,
            pcx::meta::t_id<fX>,
            uZ fft_size) {
            auto ns_passed = [=]<uZ NS>(uZ_ce<NS>) {
                auto lk_passed = [=]<bool LowK>(val_ce<LowK>) {
                    auto ltw_passed = [=]<bool LocalTw>(val_ce<LocalTw>) {
                        return ((fft_size <= NS * VecWidth
                                 || test_tform<fX, VecWidth, NS, LowK, LocalTw>(fft_size) == 0)
                                && ...);
                    };
                    return (ltw_passed(val_ce<local_tw>{}) && ...);
                };
                return (lk_passed(val_ce<low_k>{}) && ...);
            };
            return (ns_passed(uZ_ce<NodeSizes>{}) && ...);
        };
    constexpr auto exec_subtf_test =
        []<uZ... NodeSizes, uZ... VecWidth, bool... low_k, bool... local_tw, typename fX>(
            std::index_sequence<NodeSizes...>,
            std::index_sequence<VecWidth...>,
            pcx::meta::val_seq<low_k...>,
            pcx::meta::val_seq<local_tw...>,
            pcx::meta::t_id<fX>,
            uZ fft_size) {
            auto ns_passed = [=]<uZ NS>(uZ_ce<NS>) {
                auto lk_passed = [=]<bool LowK>(val_ce<LowK>) {
                    auto ltw_passed = [=]<bool LocalTw>(val_ce<LocalTw>) {
                        return ((fft_size <= NS * VecWidth
                                 || test_subtranform<fX, VecWidth, NS, LowK, LocalTw>(fft_size) == 0)
                                && ...);
                    };
                    return (ltw_passed(val_ce<local_tw>{}) && ...);
                };
                return (lk_passed(val_ce<low_k>{}) && ...);
            };
            return (ns_passed(uZ_ce<NodeSizes>{}) && ...);
        };
    constexpr auto node_sizes = std::index_sequence<2>{};
    constexpr auto f64_widths = std::index_sequence<8>{};
    constexpr auto f32_widths = std::index_sequence<16>{};
    constexpr auto low_k      = pcx::meta::val_seq<false>{};
    constexpr auto local_tw   = pcx::meta::val_seq<true>{};

    // constexpr auto node_sizes = std::index_sequence<2, 4, 8, 16>{};
    // constexpr auto f64_widths = std::index_sequence<2, 4, 8>{};
    // constexpr auto f32_widths = std::index_sequence<4, 8, 16>{};
    // constexpr auto low_k    = pcx::meta::val_seq<false, true>{};
    // constexpr auto local_tw = pcx::meta::val_seq<false, true>{};
    constexpr auto f32t = pcx::meta::t_id<f32>{};
    constexpr auto f64t = pcx::meta::t_id<f64>{};

    // int test_single_load(uZ fft_size);
    // int test_subtranform(uZ fft_size);
    std::println("testing f32:");
    // if (!exec_sl_test(node_sizes, f32_widths, f32t))
    //     return -1;
    std::println();
    uZ fft_size = 256;
    while (fft_size <= 8192UZ) {
        // if (!exec_subtf_test(node_sizes, f32_widths, low_k, local_tw, f32t, fft_size))
        //     return -1;
        fft_size *= 2;
    }
    fft_size = 2048 * 2;
    while (fft_size <= 2048 * 4) {
        if (!exec_test(node_sizes, f32_widths, low_k, local_tw, f32t, fft_size))
            return -1;
        fft_size *= 2;
    }

    std::println("\ntesting f64:");
    std::println();
    // if (!exec_sl_test(node_sizes, f64_widths, f64t))
    //     return -1;
    fft_size = 512;
    while (fft_size <= 8192UZ) {
        // if (!exec_subtf_test(node_sizes, f64_widths, low_k, local_tw, f64t, fft_size))
        //     return -1;
        fft_size *= 2;
    }

    return 0;
}
