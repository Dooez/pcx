#include "test_fft.h"

#include <print>


int main() {
    // int test_single_load(uZ fft_size);
    // int test_subtranform(uZ fft_size);
    std::println("testing f32:");
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
    uZ fft_size = 256;
    while (fft_size < 8192UZ * 2) {
        if (fft_size > 16 * 16)
            test_subtranform<f32, 16, 16>(fft_size);
        test_subtranform<f32, 8, 16>(fft_size);
        test_subtranform<f32, 4, 16>(fft_size);
        test_subtranform<f32, 2, 16>(fft_size);
        test_subtranform<f32, 16, 8>(fft_size);
        test_subtranform<f32, 8, 8>(fft_size);
        test_subtranform<f32, 4, 8>(fft_size);
        test_subtranform<f32, 2, 8>(fft_size);
        test_subtranform<f32, 16, 4>(fft_size);
        test_subtranform<f32, 8, 4>(fft_size);
        test_subtranform<f32, 4, 4>(fft_size);
        test_subtranform<f32, 2, 4>(fft_size);
        test_subtranform<f32, 16, 2>(fft_size);
        test_subtranform<f32, 8, 2>(fft_size);
        test_subtranform<f32, 4, 2>(fft_size);
        test_subtranform<f32, 2, 2>(fft_size);
        fft_size *= 2;
    }

    std::println("\ntesting f64:");
    // test_single_load<8, 8, f64>();
    // test_single_load<8, 4, f64>();
    // test_single_load<8, 2, f64>();
    // test_single_load<4, 8, f64>();
    // test_single_load<4, 4, f64>();
    // test_single_load<4, 2, f64>();
    // test_single_load<2, 8, f64>();
    // test_single_load<2, 4, f64>();
    // test_single_load<2, 2, f64>();
    std::println();
    fft_size = 128;
    while (fft_size < 8192UZ * 2) {
        if (fft_size > 8 * 16)
            test_subtranform<f64, 8, 16>(fft_size);
        test_subtranform<f64, 4, 16>(fft_size);
        test_subtranform<f64, 2, 16>(fft_size);
        test_subtranform<f64, 8, 8>(fft_size);
        test_subtranform<f64, 4, 8>(fft_size);
        test_subtranform<f64, 2, 8>(fft_size);
        test_subtranform<f64, 8, 4>(fft_size);
        test_subtranform<f64, 4, 4>(fft_size);
        test_subtranform<f64, 2, 4>(fft_size);
        test_subtranform<f64, 8, 2>(fft_size);
        test_subtranform<f64, 4, 2>(fft_size);
        test_subtranform<f64, 2, 2>(fft_size);
        fft_size *= 2;
    }

    return 0;
}
