#pragma once
#include "pcx/include/fft_impl.hpp"

#include <print>
#include <vector>
using namespace pcx;

template<typename T>
struct std::formatter<std::complex<T>, char> {
    std::formatter<T, char> m_fmt;
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        return m_fmt.parse(ctx);
    }

    template<class FmtContext>
    FmtContext::iterator format(std::complex<T> cxv, FmtContext& ctx) const {
        *ctx.out() = "(";
        auto out   = m_fmt.format(cxv.real(), ctx);
        *(out++)   = ",";
        *out       = " ";
        out        = m_fmt.format(cxv.imag(), ctx);
        *(out++)   = ")";
        return out;
    }
};
namespace pcxt {
template<typename T>
auto cmul(std::complex<T> a, std::complex<T> b) {
    auto* ca    = reinterpret_cast<T*>(&a);
    auto* cb    = reinterpret_cast<T*>(&b);
    auto  va    = simd::cxbroadcast<1>(ca);
    auto  vb    = simd::cxbroadcast<1>(cb);
    using vec_t = decltype(va);

    // auto rva = simd::repack<vec_t::width()>(va);
    // auto rvb = simd::repack<vec_t::width()>(vb);

    using resarr = std::array<std::complex<T>, vec_t::width()>;
    auto res     = resarr{};
    auto resptr  = reinterpret_cast<T*>(res.data());

    auto mulr = simd::mul(va, vb);
    auto vres = simd::repack<1>(mulr);
    simd::cxstore<1>(resptr, vres);
    return res[0];
}
template<typename T>
void btfly(std::complex<T>* a, std::complex<T>* b, std::complex<T> tw) {
    auto b_tw = cmul(*b, tw);
    auto a_c  = *a;
    *a        = a_c + b_tw;
    *b        = a_c - b_tw;
}
};    // namespace pcxt
constexpr auto powi(u64 num, u64 pow) -> u64 {    // NOLINT(*recursion*)
    auto res = (pow % 2) == 1 ? num : 1UL;
    if (pow > 1) {
        auto half_pow = powi(num, pow / 2UL);
        res *= half_pow * half_pow;
    }
    return res;
}

constexpr auto log2i(u64 num) -> uZ {
    u64 order = 0;
    for (u8 shift = 32; shift > 0; shift /= 2) {
        if (num >> shift > 0) {
            order += num >> shift > 0 ? shift : 0;
            num >>= shift;
        }
    }
    return order;
}

template<uZ VecSize, uZ VecCount, typename fX = f32>    // 32 for avx512
auto make_tw_vec_single_load(uZ start_offset = 0, uZ start_size = 2) {
    auto twvec    = std::vector<std::complex<fX>>();
    auto fft_size = start_size;
    uZ   n_tw     = 1;
    while (n_tw <= VecSize * VecCount / 2) {
        for (auto i: stdv::iota(0U, n_tw)) {
            auto rk = pcx::detail_::reverse_bit_order(start_offset + i, log2i(fft_size) - 1);
            auto tw = pcx::detail_::wnk<fX>(fft_size, rk);
            twvec.push_back(tw);
        }
        start_offset *= 2;
        fft_size *= 2;
        n_tw *= 2;
    }
    return twvec;
}

template<uZ VecSize, uZ VecCount, typename T>    // 32 for avx512
void naive_single_load(std::complex<T>* data, const std::complex<T>* tw_ptr) {
    auto fft_size = 2;
    auto step     = VecSize * VecCount / 2;
    auto n_groups = 1;
    auto stop_cnt = 1;
    while (step >= 1) {
        // if (fft_size == VecCount * 2)
        //     return;
        for (uZ k = 0; k < n_groups; ++k) {
            uZ start = k * step * 2;
            for (uZ i = 0; i < step; ++i) {
                pcxt::btfly(data + start + i, data + start + i + step, *tw_ptr);    //
            }
            ++tw_ptr;
        }
        step /= 2;
        n_groups *= 2;
        fft_size *= 2;
    }
}
template<uZ VecSize, uZ VecCount, typename fX = f32, bool LowK = false>
int test_single_load() {
    constexpr auto fft_size = VecSize * VecCount;
    auto           freq_n   = fft_size / 2;

    auto twvec   = make_tw_vec_single_load<VecSize, VecCount, fX>();
    auto datavec = [=]() {
        auto vec = std::vector<std::complex<fX>>(fft_size);
        for (auto [i, v]: stdv::enumerate(vec)) {
            v = std::exp(std::complex<fX>(0, 1)                 //
                         * static_cast<fX>(2)                   //
                         * static_cast<fX>(std::numbers::pi)    //
                         * static_cast<fX>(i)                   //
                         * static_cast<fX>(freq_n)              //
                         / static_cast<fX>(fft_size));
        }
        return vec;
    }();
    auto datavec2 = datavec;
    naive_single_load<VecSize, VecCount>(datavec.data(), twvec.data());
    //
    // auto twvec_lo_k  = make_tw_vec_lo_k<vec_size, vec_count>();
    using fimpl    = pcx::detail_::subtransform<VecCount, fX, VecSize>;
    auto* data_ptr = reinterpret_cast<fX*>(datavec2.data());
    auto* tw_ptr   = reinterpret_cast<fX*>(twvec.data());
    // auto* tw_lok_ptr = reinterpret_cast<fX*>(twvec_lo_k.data());
    fimpl::template single_load<1, 1, false>(data_ptr, data_ptr, tw_ptr);
    // fimpl::single_load<1, 1, true>(data_ptr, data_ptr, tw_lok_ptr);

    // bit_reverse_sort(datavec);
    // bit_reverse_sort(datavec2);
    auto single_load_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
                                          [](auto v) { return std::get<0>(v) != std::get<1>(v); });

    if (single_load_error) {
        std::println("Error in single load for {} simd vectors of width {}.", VecCount, VecSize);
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2)) {
            std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                         i,
                         (naive),
                         (pcx),
                         (naive - pcx));
        }
        return -1;
    }

    std::println("Successful single load for {} simd vectors of width {}.", VecCount, VecSize);
    // for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2)) {
    //     std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
    //                  i,
    //                  (naive),
    //                  (pcx),
    //                  (naive - pcx));
    // }
    return 0;
};

int test_single_load(uZ fft_size);
int test_subtranform_f32(uZ fft_size);
int test_subtranform_f64(uZ fft_size);
