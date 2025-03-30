#include "common.hpp"

#include <cstdlib>

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
template<typename T>
void rbtfly(std::complex<T>* a, std::complex<T>* b, std::complex<T> tw) {
    using ct = std::complex<T>;
    auto c_a = *a + *b;
    auto c_b = *a - *b;
    *a       = c_a;
    *b       = cmul(c_b, tw);
}
constexpr auto is_pow_of_two(uZ n) {
    return n > 0 && (n & (n - 1)) == 0;
}
template<typename fX>
void naive_reverse(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");

    auto fft_size         = rsize / 2;
    auto step             = 1;
    auto n_groups         = rsize / 2;
    auto single_load_size = vec_width * node_size;
    while (step <= rsize / 2) {
        bool skip = false;
        if (step <= vec_width / 2) {    // skip sub width
            // skip = true;
        }
        if (step <= vec_width * node_size / 2) {    // skip single load
            skip = true;
        }
        if (step <= 2048 / 2) {    // skip coherent
            // skip = true;
        }
        if (step < vec_width / 4) {
            // skip = true;
        }
        if (!skip) {
            for (uZ k = 0; k < n_groups; ++k) {
                uZ   start = k * step * 2;
                auto tw    = conj(pcx::detail_::wnk_br<fX>(fft_size * 2, k));
                for (uZ i = 0; i < step; ++i) {
                    rbtfly(&data[start + i], &data[start + i + step], tw);    //
                    data[start + i] /= 2;
                    data[start + i + step] /= 2;
                }
            }
        }
        fft_size /= 2;
        step *= 2;
        n_groups /= 2;
    }
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
        if (step == vec_width / 2) {    // skip sub width
            // break;
        }
        if (step == vec_width * node_size / 2) {    // skip single load
            break;
        }
        if (step == rsize / 4) {
            // break;
        }
        if (step <= 2048 / 2) {    // skip coherent
            // break;
        }
        if (step < vec_width / 4) {
            // break;
        }
        for (uZ k = 0; k < n_groups; ++k) {
            uZ   start = k * step * 2;
            auto tw    = pcx::detail_::wnk_br<fX>(fft_size * 2, k);
            for (uZ i = 0; i < step; ++i) {
                btfly(&data[start + i], &data[start + i + step], tw);    //
            }
        }
        fft_size *= 2;
        step /= 2;
        n_groups *= 2;
    }
}
// template void naive_fft(std::vector<std::complex<f32>>& data, uZ, uZ);
// template void naive_fft(std::vector<std::complex<f64>>& data, uZ, uZ);
}    // namespace pcx::testing
#ifdef FULL_FFT_TEST
inline constexpr auto node_sizes = pcx::uZ_seq<2, 4, 8, 16>{};
#ifdef NO_SPLIT_COMPILE
namespace pcx::testing {
template<typename fX, uZ NodeSize>
bool test_fft(const std::vector<std::complex<fX>>& signal,
              const std::vector<std::complex<fX>>& check,
              std::vector<std::complex<fX>>&       s1,
              std::vector<std::complex<fX>>&       s2) {
    return run_tests<fX, NodeSize>(f32_widths, low_k, local_tw, half_tw, signal, check, s1, s2);
};
}    // namespace pcx::testing
#endif
#else
inline constexpr auto node_sizes = pcx::uZ_seq<FFT_NODE_SIZE>{};
#endif

int main() {
    namespace stdv = std::views;
    namespace stdr = std::ranges;

    auto test_size = []<uZ... Is>(pcx::uZ_seq<Is...>, uZ fft_size, f64 freq_n) {
        constexpr uZ node_size = 8;
        constexpr uZ width     = 16;

        auto make_signal_check = [=]<typename fX>(pcx::meta::t_id<fX>) {
            auto vec = std::vector<std::complex<fX>>(fft_size);
            for (auto [i, v]: stdv::enumerate(vec)) {
                v = std::exp(std::complex<fX>(0, 1)                 //
                             * static_cast<fX>(2)                   //
                             * static_cast<fX>(std::numbers::pi)    //
                             * static_cast<fX>(i)                   //
                             * static_cast<fX>(freq_n)              //
                             / static_cast<fX>(fft_size));
            }

            auto check = vec;
            pcx::testing::naive_fft(check, node_size, width);
            return std::make_tuple(vec, check);
        };
        auto [s32, check32] = make_signal_check(pcx::meta::t_id<f32>{});
        auto [s64, check64] = make_signal_check(pcx::meta::t_id<f64>{});
        auto s32_1          = std::vector<std::complex<f32>>(fft_size);
        auto s32_2          = std::vector<f32>(fft_size);
        auto s64_1          = std::vector<std::complex<f64>>(fft_size);
        auto s64_2          = std::vector<f64>(fft_size);

        // auto r32 = check32;
        // pcx::testing::naive_reverse(r32, node_size, width);
        // f32 delta = std::numeric_limits<f32>::epsilon() * static_cast<f32>(pcx::detail_::log2i(fft_size)) / 2;
        // f32 max_diff = 0;
        // for (auto [i, s, c, r]: stdv::zip(stdv::iota(0U), s32, check32, r32)) {
        //     // std::println("{}, {}, {}", s, c, r);
        //     // continue;
        //     auto diff = abs(s - r);
        //     max_diff  = std::max(diff, max_diff);
        //     if (diff > delta) {
        //         std::println("[Failure] reverse. fft size: {}, freq_n {}, i: {}, s: {}, r: {}",
        //                      fft_size,
        //                      freq_n,
        //                      i,
        //                      s,
        //                      r);
        //         return false;
        //     }
        // }
        // std::println("[Success] reverse. fft size: {}, freq_n: {}, max diff: {} epsilon",
        //              fft_size,
        //              freq_n,
        //              max_diff / std::numeric_limits<f32>::epsilon());
        // return true;

        // auto mag  = stdr::to<std::vector<f32>>(check32 | stdv::transform([](auto v) { return abs(v); }));
        // auto x    = stdr::max_element(mag);
        // auto brid = x - mag.begin();
        // auto id   = pcx::detail_::reverse_bit_order(brid, pcx::detail_::log2i(fft_size));
        // if (id == std::round(freq_n)) {
        //     std::println("[Success] fft size {}, freq_n {}", fft_size, freq_n);
        //     return true;
        // }
        // std::println("[Failure] fft size {}, freq_n {}, detected n {}", fft_size, freq_n, id);
        // return false;


        return (pcx::testing::test_fft<f32, Is>(s32, check32, s32_1, s32_2) && ...)
               && (pcx::testing::test_fft<f64, Is>(s64, check64, s64_1, s64_2) && ...);
    };
    // uZ fft_size = 2048 * 256;
    uZ fft_size = 512;
    // for (auto i: stdv::iota(0U, fft_size)) {
    //     if (!test_size(node_sizes, fft_size, i + .01))
    //         return -1;
    // }
    while (fft_size <= 2048 * 256 * 4) {
        // if (!test_size(node_sizes, fft_size, 31.01 * fft_size / 64))
        if (!test_size(node_sizes, fft_size, fft_size / 2 * 1.001))
            return -1;
        fft_size *= 2;
    }
    return 0;
}

namespace pcx::testing {
template<typename fX>
bool check_correctness(const std::vector<std::complex<fX>>& naive,
                       const std::vector<std::complex<fX>>& pcx,
                       uZ                                   width,
                       uZ                                   node_size,
                       bool                                 lowk,
                       bool                                 local_tw,
                       bool                                 half_tw) {
    auto fft_size = naive.size();

    auto subtform_error = stdr::any_of(stdv::zip(naive, pcx),    //
                                       [](auto v) { return std::get<0>(v) != std::get<1>(v); });
    if (subtform_error) {
        std::println("[Error] {}×{}, width {}, node size {}{}{}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     width,
                     node_size,
                     lowk ? ", low k" : "",
                     local_tw ? ", local tw" : "",
                     half_tw ? ", half tw" : "");
        uZ err_cnt = 0;
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), naive, pcx) | stdv::take(999999999)) {
            if (naive != pcx) {
                std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                             i,
                             (naive),
                             (pcx),
                             abs(naive - pcx));
                ++err_cnt;
            }
            if (err_cnt > 100)
                break;
            if (std::abs(naive - pcx) > 1) {
                // std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                //              i,
                //              (naive),
                //              (pcx),
                //              (naive - pcx));
                std::println("Over 1 found.");
                // break;
            }
            // }
        }
        return false;
    }
    std::println("[Success] {}×{}, width {}, node size {}{}{}{}.",
                 pcx::meta::types<fX>{},
                 fft_size,
                 width,
                 node_size,
                 lowk ? ", low k" : "",
                 local_tw ? ", local tw" : "",
                 half_tw ? ", half tw" : "");
    return true;
}

template bool check_correctness<f32>(const std::vector<std::complex<f32>>&,
                                     const std::vector<std::complex<f32>>&,
                                     uZ,
                                     uZ,
                                     bool,
                                     bool,
                                     bool);
template bool check_correctness<f64>(const std::vector<std::complex<f64>>&,
                                     const std::vector<std::complex<f64>>&,
                                     uZ,
                                     uZ,
                                     bool,
                                     bool,
                                     bool);
}    // namespace pcx::testing
