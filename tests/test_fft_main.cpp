#include "test_fft.h"

#include <print>


int main() {
    // int test_single_load(uZ fft_size);
    // int test_subtranform(uZ fft_size);
    uZ fft_size = 256;
    while (fft_size < 8192UZ * 2) {
        test_subtranform(fft_size);
        fft_size *= 2;
    }


    return 0;
}
