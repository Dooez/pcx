#include "impl-test.hpp"

#include <cstdlib>

using pcx::f32;
using pcx::f64;
using pcx::uZ;

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
        if (!test_par(pcx::testing::f64_widths, f64_tid, fft_size, 31, 13.001))
            return -1;
        if (!test_size(pcx::testing::f64_widths, f64_tid, fft_size, fft_size / 2 * 13.0001))
            return -1;
        fft_size *= 2;
    }
    return 0;
}
