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
        // while (step < rsize / 2) {
        if (step > 2048 / 2) {
            // break;
        }
        if (step >= node_size * vec_width) {
            // break;
        }
        // while (step < rsize / 2) {
        bool skip = false;
        if (step <= vec_width / 2) {    // skip sub width
            // skip = true;
        }
        if (step <= vec_width * node_size / 2) {    // skip single load
            // skip = true;
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
                    // data[start + i] /= 2;
                    // data[start + i + step] /= 2;
                }
            }
            auto chekc = 0;
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
            // break;
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
template void naive_fft(std::vector<std::complex<f32>>& data, uZ, uZ);
template void naive_fft(std::vector<std::complex<f64>>& data, uZ, uZ);
template void naive_reverse(std::vector<std::complex<f32>>& data, uZ, uZ);
template void naive_reverse(std::vector<std::complex<f64>>& data, uZ, uZ);
}    // namespace pcx::testing

int main() {
    namespace stdv = std::views;
    namespace stdr = std::ranges;
    auto test_par  = []<uZ... Ws, typename fX>(pcx::uZ_seq<Ws...>,
                                              pcx::meta::t_id<fX>,
                                              uZ  fft_size,
                                              uZ  data_size,
                                              f64 freq_n) {
        constexpr uZ   width     = 16;
        constexpr auto node_size = 8;

        bool local_check = false;
        bool fwd         = true;
        bool rev         = true;
        bool inplace     = true;
        bool external    = true;

        auto signal = pcx::testing::std_vec2d<fX>(fft_size);
        using enum pcx::testing::permute_t;
        auto chk_fwd          = pcx::testing::chk_t<fX>{};
        auto chk_rev          = pcx::testing::chk_t<fX>{};
        chk_fwd(bit_reversed) = std::vector<std::complex<fX>>(fft_size);
        for (auto [i, v, vcf]: stdv::zip(stdv::iota(0U), signal, chk_fwd(bit_reversed))) {
            auto cx = std::exp(std::complex<fX>(0, 1)                 //
                               * static_cast<fX>(2)                   //
                               * static_cast<fX>(std::numbers::pi)    //
                               * static_cast<fX>(i)                   //
                               * static_cast<fX>(freq_n)              //
                               / static_cast<fX>(fft_size));

            vcf = cx;
            v.resize(data_size);
            stdr::fill(v, cx);
        }
        chk_rev(bit_reversed) = chk_fwd(bit_reversed);
        {
            pcx::testing::naive_fft(chk_fwd(bit_reversed), node_size, width);
            chk_fwd(normal)  = chk_fwd(bit_reversed);
            chk_fwd(shifted) = chk_fwd(bit_reversed);
            auto& norm       = chk_fwd(normal);
            for (auto i: stdv::iota(0U, norm.size())) {
                auto br = pcx::detail_::reverse_bit_order(i, pcx::detail_::log2i(fft_size));
                if (br > i)
                    std::swap(norm[i], norm[br]);
            }
            auto& shft   = chk_fwd(shifted);
            auto  rbo_sh = [=](auto k) {
                auto br = pcx::detail_::reverse_bit_order(k, pcx::detail_::log2i(fft_size));
                return (br + fft_size / 2) % fft_size;
            };
            for (uZ i: stdv::iota(0U, shft.size())) {
                uZ br0 = rbo_sh(i);
                uZ br1 = rbo_sh(br0);
                uZ br2 = rbo_sh(br1);
                if (std::min({i, br0, br1, br2}) == i) {
                    auto vi   = shft[i];
                    auto v0   = shft[br0];
                    auto v1   = shft[br1];
                    auto v2   = shft[br2];
                    shft[i]   = v2;
                    shft[br0] = vi;
                    shft[br1] = v0;
                    shft[br2] = v1;
                }
            }
        }
        {
            chk_rev(normal)  = chk_rev(bit_reversed);
            chk_rev(shifted) = chk_rev(bit_reversed);
            auto& norm       = chk_rev(normal);
            for (auto i: stdv::iota(0U, norm.size())) {
                auto br = pcx::detail_::reverse_bit_order(i, pcx::detail_::log2i(fft_size));
                if (br > i)
                    std::swap(norm[i], norm[br]);
            }
            auto& shft   = chk_rev(shifted);
            auto  rbo_sh = [=](auto k) {
                auto br = pcx::detail_::reverse_bit_order(k, pcx::detail_::log2i(fft_size));
                return (br + fft_size / 2) % fft_size;
            };
            for (uZ i: stdv::iota(0U, shft.size())) {
                uZ br0 = rbo_sh(i);
                uZ br1 = rbo_sh(br0);
                uZ br2 = rbo_sh(br1);
                if (std::min({i, br0, br1, br2}) == i) {
                    auto vi   = shft[i];
                    auto v0   = shft[br0];
                    auto v1   = shft[br1];
                    auto v2   = shft[br2];
                    shft[i]   = v2;
                    shft[br0] = vi;
                    shft[br1] = v0;
                    shft[br2] = v1;
                }
            }
            pcx::testing::naive_reverse(chk_rev(bit_reversed), node_size, width);
            pcx::testing::naive_reverse(norm, node_size, width);
            pcx::testing::naive_reverse(shft, node_size, width);
        }
        auto s1    = signal;
        auto twvec = std::vector<fX>{};
        return (pcx::testing::test_par<fX, Ws>(signal,
                                               s1,
                                               chk_fwd,
                                               chk_rev,
                                               twvec,
                                               local_check,
                                               fwd,
                                               rev,
                                               inplace,
                                               external)
                && ...);
    };

    auto test_size =
        []<uZ... Is, typename fX>(pcx::uZ_seq<Is...>, pcx::meta::t_id<fX>, uZ fft_size, f64 freq_n) {
            constexpr uZ node_size = 8;
            constexpr uZ width     = 16;

            bool local_check = false;
            bool fwd         = true;
            bool rev         = true;
            bool inplace     = true;
            bool external    = true;

            auto tw     = std::vector<fX>();
            auto signal = std::vector<std::complex<fX>>(fft_size);
            for (auto [i, v]: stdv::enumerate(signal)) {
                v = std::exp(std::complex<fX>(0, 1)                 //
                             * static_cast<fX>(2)                   //
                             * static_cast<fX>(std::numbers::pi)    //
                             * static_cast<fX>(i)                   //
                             * static_cast<fX>(freq_n)              //
                             / static_cast<fX>(fft_size));
            }

            using enum pcx::testing::permute_t;
            auto chk_fwd = pcx::testing::chk_t<fX>{};
            auto chk_rev = pcx::testing::chk_t<fX>{};

            chk_fwd(bit_reversed) = signal;
            chk_rev(bit_reversed) = signal;
            {
                pcx::testing::naive_fft(chk_fwd(bit_reversed), node_size, width);
                chk_fwd(normal)  = chk_fwd(bit_reversed);
                chk_fwd(shifted) = chk_fwd(bit_reversed);
                auto& norm       = chk_fwd(normal);
                for (auto i: stdv::iota(0U, norm.size())) {
                    auto br = pcx::detail_::reverse_bit_order(i, pcx::detail_::log2i(fft_size));
                    if (br > i)
                        std::swap(norm[i], norm[br]);
                }
                auto& shft   = chk_fwd(shifted);
                auto  rbo_sh = [=](auto k) {
                    auto br = pcx::detail_::reverse_bit_order(k, pcx::detail_::log2i(fft_size));
                    return (br + fft_size / 2) % fft_size;
                };
                for (uZ i: stdv::iota(0U, shft.size())) {
                    uZ br0 = rbo_sh(i);
                    uZ br1 = rbo_sh(br0);
                    uZ br2 = rbo_sh(br1);
                    if (std::min({i, br0, br1, br2}) == i) {
                        auto vi   = shft[i];
                        auto v0   = shft[br0];
                        auto v1   = shft[br1];
                        auto v2   = shft[br2];
                        shft[i]   = v2;
                        shft[br0] = vi;
                        shft[br1] = v0;
                        shft[br2] = v1;
                    }
                }
            }
            {
                chk_rev(normal)  = chk_rev(bit_reversed);
                chk_rev(shifted) = chk_rev(bit_reversed);
                auto& norm       = chk_rev(normal);
                for (auto i: stdv::iota(0U, norm.size())) {
                    auto br = pcx::detail_::reverse_bit_order(i, pcx::detail_::log2i(fft_size));
                    if (br > i)
                        std::swap(norm[i], norm[br]);
                }
                auto& shft   = chk_rev(shifted);
                auto  rbo_sh = [=](auto k) {
                    auto br = pcx::detail_::reverse_bit_order(k, pcx::detail_::log2i(fft_size));
                    return (br + fft_size / 2) % fft_size;
                };
                for (uZ i: stdv::iota(0U, shft.size())) {
                    uZ br0 = rbo_sh(i);
                    uZ br1 = rbo_sh(br0);
                    uZ br2 = rbo_sh(br1);
                    if (std::min({i, br0, br1, br2}) == i) {
                        auto vi   = shft[i];
                        auto v0   = shft[br0];
                        auto v1   = shft[br1];
                        auto v2   = shft[br2];
                        shft[i]   = v2;
                        shft[br0] = vi;
                        shft[br1] = v0;
                        shft[br2] = v1;
                    }
                }
                pcx::testing::naive_reverse(chk_rev(bit_reversed), node_size, width);
                pcx::testing::naive_reverse(norm, node_size, width);
                pcx::testing::naive_reverse(shft, node_size, width);
            }
            auto s1 = std::vector<std::complex<fX>>(fft_size);
            auto s2 = std::vector<std::complex<fX>>(fft_size);
            return (pcx::testing::test_fft<fX, Is>(signal,
                                                   chk_fwd,
                                                   chk_rev,
                                                   s1,
                                                   tw,
                                                   local_check,
                                                   fwd,
                                                   rev,
                                                   inplace,
                                                   external)
                    && ...);
        };
    // uZ fft_size = 2048 * 256;
    // uZ fft_size = 32768;
    uZ fft_size = 512;
    // uZ fft_size = 256;
    // uZ fft_size = 128 * 128 * 2;
    // uZ fft_size = 2048;
    // uZ fft_size = 131072 * 4;

    // for (auto i: stdv::iota(0U, fft_size)) {
    //     if (!test_size(node_sizes, fft_size, i + .01))
    //         return -1;
    // }
    //
    constexpr auto f32_tid = pcx::meta::t_id<f32>{};
    constexpr auto f64_tid = pcx::meta::t_id<f64>{};
    while (fft_size <= 2048 * 2048 * 2) {
        if (!test_par(pcx::testing::f32_widths, f32_tid, fft_size, 31, 13.001))
            return -1;
        if (!test_size(pcx::testing::f32_widths, f32_tid, fft_size, fft_size / 2 * 13.0001))
            return -1;
        // if (!test_par(pcx::testing::f64_widths, f64_tid, fft_size, 31, 13.001))
        //     return -1;
        if (!test_size(pcx::testing::f64_widths, f64_tid, fft_size, fft_size / 2 * 13.0001))
            return -1;
        fft_size *= 2;
    }
    return 0;
}

namespace pcx::testing {
template<typename fX>
bool par_check_correctness(std::complex<fX>                     val,
                           const std::vector<std::complex<fX>>& pcx,
                           uZ                                   fft_size,
                           uZ                                   fft_id,
                           uZ                                   width,
                           uZ                                   node_size,
                           bool                                 local_tw) {
    for (auto [i, v]: stdv::enumerate(pcx)) {
        if (v == val)
            continue;
        std::println("[Error] {}×{}@{}:{}, width {}, node size {}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     fft_id,
                     i,
                     width,
                     node_size,
                     local_tw ? ", local tw" : "");
        std::println("expected: {}", val);
        for (auto [ei, ev]: stdv::enumerate(pcx | stdv::drop(i)) | stdv::take(100)) {
            std::println("{:>3}| pcx:{: >6.4f}, diff:{}",    //
                         ei + i,
                         ev,
                         (ev - val));
        }
        return false;
    }
    return true;
}
template bool
par_check_correctness(std::complex<f32>, const std::vector<std::complex<f32>>&, uZ, uZ, uZ, uZ, bool);
template bool
par_check_correctness(std::complex<f64>, const std::vector<std::complex<f64>>&, uZ, uZ, uZ, uZ, bool);

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
            if (err_cnt > 1000)
                break;
            // if (std::abs(naive - pcx) > 1) {
            //     // std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
            //     //              i,
            //     //              (naive),
            //     //              (pcx),
            //     //              (naive - pcx));
            //     std::println("Over 1 found.");
            //     // break;
            // }
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
