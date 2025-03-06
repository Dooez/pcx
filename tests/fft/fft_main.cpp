#include "common.hpp"

using pcx::f32;
using pcx::f64;
using pcx::uZ;

namespace pcx::testing {

template<typename T>
auto cmul(std::complex<T> a, std::complex<T> b, bool conj_b = false) {
    auto* ca = reinterpret_cast<T*>(&a);
    auto* cb = reinterpret_cast<T*>(&b);
    auto  va = simd::cxbroadcast<1>(ca);
    auto  vb = simd::cxbroadcast<1>(cb);

    using vec_t = decltype(va);

    // auto rva = simd::repack<vec_t::width()>(va);
    // auto rvb = simd::repack<vec_t::width()>(vb);
    using ct = std::complex<T>;
    if (b == ct{1, 0})
        return a;
    if (b == ct{-1, 0})
        return -a;
    if (b == ct{0, 1})
        return ct{-a.imag(), a.real()};
    if (b == ct{0, -1})
        return ct{a.imag(), -a.real()};

    using resarr = std::array<std::complex<T>, vec_t::width()>;
    auto res     = resarr{};
    auto resptr  = reinterpret_cast<T*>(res.data());

    if (conj_b) {
        auto mulr = simd::mul(va, conj(vb));
        auto vres = simd::repack<1>(mulr);
        simd::cxstore<1>(resptr, vres);
    } else {
        auto mulr = simd::mul(va, vb);
        auto vres = simd::repack<1>(mulr);
        simd::cxstore<1>(resptr, vres);
    }
    return res[0];
}
template<typename T>
void btfly(std::complex<T>* a, std::complex<T>* b, std::complex<T> tw, bool conj_b = false) {
    using ct = std::complex<T>;

    auto b_tw = cmul(*b, tw, conj_b);
    auto a_c  = *a;
    *a        = a_c + b_tw;
    *b        = a_c - b_tw;
}
constexpr auto is_pow_of_two(uZ n) {
    return n > 0 && (n & (n - 1)) == 0;
}
template<typename fX>
void naive_fft(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");

    auto fft_size         = 1;
    auto step             = rsize / 2;
    auto n_groups         = 1;
    auto single_load_size = vec_width * node_size;
    while (step >= 1) {
        if (step == vec_width / 2) {    // skip single load small vector
            // break;
        }
        if (step == vec_width * node_size / 2) {    // skip single load
            // break;
        }
        if (step == rsize / 4) {
            // break;
        }
        if (step == 2048 / 2) {    // skip coherent
            // break;
        }
        if (step < vec_width / 2) {
            break;
        }
        fft_size *= 2;
        if (step < vec_width) {
            for (uZ k = 0; k < n_groups; ++k) {
                uZ   start = k * step * 2;
                auto tw    = pcx::detail_::wnk_br<fX>(fft_size, k);
                if (k % 2 == 1) {
                    // tw = pcx::detail_::wnk_br<fX>(fft_size, k - 1);
                }
                for (uZ i = 0; i < step; ++i) {
                    btfly(&data[start + i], &data[start + i + step], tw);    //
                }
            }
        } else {
            for (uZ k = 0; k < n_groups; ++k) {
                uZ   start = k * step * 2;
                auto tw    = pcx::detail_::wnk_br<fX>(fft_size, k);
                for (uZ i = 0; i < step; ++i) {
                    btfly(&data[start + i], &data[start + i + step], tw);    //
                }
            }
        }
        // if (n_groups >= powi(2, 2))
        //     break;
        step /= 2;
        n_groups *= 2;
        // break;
    }
}
template void naive_fft(std::vector<std::complex<f32>>& data, uZ, uZ);
template void naive_fft(std::vector<std::complex<f64>>& data, uZ, uZ);
}    // namespace pcx::testing
#ifdef FULL_FFT_TEST
inline constexpr auto node_sizes = pcx::uZ_seq<2, 4, 8, 16>{};
#else
inline constexpr auto node_sizes = pcx::uZ_seq<8>{};
#endif

int main() {
    auto test_size = []<uZ... Is>(pcx::uZ_seq<Is...>, uZ fft_size, uZ freq_n) {
        return (pcx::testing::test_fft<f32, Is>(fft_size, freq_n) && ...)
               && (pcx::testing::test_fft<f64, Is>(fft_size, freq_n) && ...);
    };
    uZ fft_size = 2048 * 2;
    while (fft_size <= 2048 * 64) {
        // if (!test_size(node_sizes, fft_size, fft_size / 64))
        if (!test_size(node_sizes, fft_size, fft_size / 64))
            return -1;
        fft_size *= 2;
    }
    return 0;
}
