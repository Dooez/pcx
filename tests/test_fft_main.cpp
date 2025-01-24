#include "test_fft.h"

#include <print>


int main() {
    constexpr auto exec_sl_test = []<uZ... NodeSizes, uZ... VecWidth, typename fX>(
                                      std::index_sequence<NodeSizes...>,
                                      std::index_sequence<VecWidth...>,
                                      pcx::meta::types<fX>) {
        auto ns_passed = [=]<uZ NS>(uZc<NS>) { return ((test_single_load<fX, VecWidth, NS>() == 0) && ...); };
        auto passed    = (ns_passed(uZc<NodeSizes>{}) && ...);
        return passed;
    };
    constexpr auto exec_test = []<uZ... NodeSizes, uZ... VecWidth, typename fX>(
                                   std::index_sequence<NodeSizes...>,
                                   std::index_sequence<VecWidth...>,
                                   pcx::meta::types<fX>,
                                   uZ fft_size) {
        auto ns_passed = [=]<uZ NS>(uZc<NS>) {
            return ((fft_size <= NS * VecWidth || test_subtranform<fX, VecWidth, NS>(fft_size) == 0) && ...);
        };
        auto passed = (ns_passed(uZc<NodeSizes>{}) && ...);
        return passed;
    };
    constexpr auto node_sizes = std::index_sequence<4>{};
    // constexpr auto f64_widths = std::index_sequence<8>{};
    constexpr auto f32_widths = std::index_sequence<4>{};
    // constexpr auto node_sizes = std::index_sequence<2, 4, 8>{};
    constexpr auto f64_widths = std::index_sequence<2, 4, 8>{};
    // constexpr auto f32_widths = std::index_sequence<4, 8, 16>{};
    constexpr auto f32t       = pcx::meta::types<f32>{};
    constexpr auto f64t       = pcx::meta::types<f64>{};

    // int test_single_load(uZ fft_size);
    // int test_subtranform(uZ fft_size);
    std::println("testing f32:");
    // if (!exec_sl_test(node_sizes, f32_widths, f32t))
    //     return -1;
    std::println();
    uZ fft_size = 128;
    while (fft_size <= 8192UZ * 4) {
        if (!exec_test(node_sizes, f32_widths, f32t, fft_size))
            return -1;
        fft_size *= 2;
    }

    std::println("\ntesting f64:");
    std::println();
    // if (!exec_sl_test(node_sizes, f64_widths, f64t))
    //     return -1;
    fft_size = 512;
    while (fft_size < 8192UZ) {
        // if (!exec_test(node_sizes, f64_widths, f64t, fft_size))
        //     return -1;
        fft_size *= 2;
    }

    return 0;
}
