#include "test_fft.h"

#include <print>


int main() {
    // int test_single_load(uZ fft_size);
    // int test_subtranform(uZ fft_size);
    uZ fft_size = 256;
    while (fft_size < 8192UZ * 2) {
        // test_subtranform(fft_size);
        fft_size *= 2;
    }
    std::println();
    // test_single_load<16, 16>();
    // test_single_load<16, 8>();
    // test_single_load<16, 4>();
    // test_single_load<16, 2>();
    // test_single_load<8, 16>();
    // test_single_load<8, 8>();
    // test_single_load<8, 4>();
    // test_single_load<8, 2>();
    // test_single_load<4, 16>();
    // test_single_load<4, 8>();
    // test_single_load<4, 4>();
    // test_single_load<4, 2>();

    std::println();
    // test_single_load<8, 8, f64>();
    // test_single_load<8, 4, f64>();
    // test_single_load<8, 2, f64>();
    test_single_load<4, 8, f64>();
    test_single_load<4, 4, f64>();
    test_single_load<4, 2, f64>();

    return 0;
}
