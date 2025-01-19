// #define PCX_AVX512
#include "pcx/include/fft_impl.hpp"

#include <complex>
#include <cstddef>
#include <print>
#include <ranges>
#include <vector>
namespace stdv = std::views;

// NOLINTBEGIN (*pointer-arithmetic*)

using namespace pcx;
constexpr auto is_pow_of_two(uZ n) {
    return n > 0 && (n & (n - 1)) == 0;
}

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

void foo(std::complex<f32>* dest, const std::array<std::complex<f32>, 7>& tw) {
    auto dest_r     = reinterpret_cast<f32*>(dest);
    using node_t    = detail_::btfly_node_dit<8, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * 2 * Is)...);
    }(dest_r, std::make_index_sequence<8>());
    auto tw_tuple = []<uZ... Is>(auto&& tw, std::index_sequence<Is...>) {
        return tupi::make_tuple(         //
            simd::repack<16>(            //
                simd::cxbroadcast<1>(    //
                    reinterpret_cast<const f32*>(&tw[Is])))...);
    }(tw, std::make_index_sequence<7>{});

    constexpr auto settings = node_t::settings{
        .pack_dest = 1,
        .pack_src  = 1,
        .reverse   = false,
    };
    node_t::perform<settings>(dest_tuple, tw_tuple);
};

template<uZ VecSize = 16>
void bar8(std::complex<f32>* dest, auto&& tw) {
    for (auto i: stdv::iota(0U, VecSize)) {
        auto* start = dest + i;

        pcxt::btfly(start + 0 * VecSize, start + 4 * VecSize, tw[0]);
        pcxt::btfly(start + 1 * VecSize, start + 5 * VecSize, tw[0]);
        pcxt::btfly(start + 2 * VecSize, start + 6 * VecSize, tw[0]);
        pcxt::btfly(start + 3 * VecSize, start + 7 * VecSize, tw[0]);

        pcxt::btfly(start + 0 * VecSize, start + 2 * VecSize, tw[1]);
        pcxt::btfly(start + 1 * VecSize, start + 3 * VecSize, tw[1]);
        pcxt::btfly(start + 4 * VecSize, start + 6 * VecSize, tw[2]);
        pcxt::btfly(start + 5 * VecSize, start + 7 * VecSize, tw[2]);
        //
        pcxt::btfly(start + 0 * VecSize, start + 1 * VecSize, tw[3]);
        pcxt::btfly(start + 2 * VecSize, start + 3 * VecSize, tw[4]);
        pcxt::btfly(start + 4 * VecSize, start + 5 * VecSize, tw[5]);
        pcxt::btfly(start + 6 * VecSize, start + 7 * VecSize, tw[6]);
    }
};


auto log2i(u64 num) -> uZ {
    u64 order = 0;
    for (u8 shift = 32; shift > 0; shift /= 2) {
        if (num >> shift > 0) {
            order += num >> shift > 0 ? shift : 0;
            num >>= shift;
        }
    }
    return order;
}
template<typename T>
struct is_std_complex : std::false_type {};
template<typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template<typename R>
    requires stdr::random_access_range<R> && is_std_complex<stdr::range_value_t<R>>::value
void naive_fft(R& data) {
    auto rsize = stdr::size(data);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Data size is not a power of two.");
    using cx_t    = stdr::range_value_t<R>;
    auto fft_size = 2;
    auto step     = rsize / 2;
    auto n_groups = 1;
    while (step >= 1) {
        if (step == 16 * 8 / 2) {
            return;
        }
        for (uZ k = 0; k < n_groups; ++k) {
            uZ   start = k * step * 2;
            auto rk    = pcx::detail_::reverse_bit_order(k, log2i(fft_size) - 1);
            auto tw    = pcx::detail_::wnk<f32>(fft_size, rk);
            for (uZ i = 0; i < step; ++i) {
                pcxt::btfly(&data[start + i], &data[start + i + step], tw);    //
            }
        }
        step /= 2;
        n_groups *= 2;
        fft_size *= 2;
    }
}

template<uZ VecSize, uZ VecCount, typename T>    // 32 for avx512
void naive_single_load(std::complex<T>* data, const std::complex<T>* tw_ptr) {
    auto fft_size = 2;
    auto step     = VecSize * VecCount / 2;
    auto n_groups = 1;
    auto stop_cnt = 1;
    while (step >= 1) {
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
constexpr auto powi(u64 num, u64 pow) -> u64 {    // NOLINT(*recursion*)
    auto res = (pow % 2) == 1 ? num : 1UL;
    if (pow > 1) {
        auto half_pow = powi(num, pow / 2UL);
        res *= half_pow * half_pow;
    }
    return res;
}
auto make_tw_vec2(uZ fft_size, uZ node_size = 8) {
    auto tw_vec = std::vector<std::complex<f32>>();
    tw_vec.reserve(4096);
    auto insert_tw = [&](uZ size, uZ i) {
        auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
        auto tw = pcx::detail_::wnk<f32>(size, rk);
        tw_vec.push_back(tw);
    };
    auto vec_size         = 16;
    auto single_load_size = vec_size * node_size;
    auto size             = 2UZ;
    while (fft_size / size >= single_load_size * node_size) {
        for (auto i: stdv::iota(0U, size / 2)) {
            for (auto ns: stdv::iota(0U, log2i(node_size))) {
                for (auto k: stdv::iota(0U, powi(2, ns))) {
                    insert_tw(size * powi(2, ns), k + i * powi(2, ns));
                }
            }
        }
        size *= node_size;
    }
    for (uZ tp: stdv::iota(0U, log2i(node_size))) {
        if (fft_size / (size * powi(2, tp)) != single_load_size) {
            continue;
        }
        auto local_node = powi(2, tp);
        for (auto i: stdv::iota(0U, size / 2)) {
            for (auto ns: stdv::iota(0U, log2i(local_node))) {
                for (auto k: stdv::iota(0U, powi(2, ns))) {
                    insert_tw(size * powi(2, ns), k + i * powi(2, ns));
                }
            }
        }
    }
    return tw_vec;

    // [&]<uZ... Is>(std::index_sequence<Is...>) {
    //     auto try_iter = [&]<uZ I>(uZc<I>) {
    //         auto si = size;
    //         auto sz = (size * powi(2, I));
    //         if (max_size / (size * powi(2, I)) == single_load_size) {
    //             fft_iteration<powi(2, I + 1), Width, Width, LowK>(max_size,
    //                                                               size,
    //                                                               dest_ptr,
    //                                                               tw_ptr);
    //             return true;
    //         }
    //         return false;
    //     };
    //     (try_iter(uZc<Is>{}) || ...);
    //
}
auto make_tw_vec(uZ fft_size) {
    auto twvec        = std::vector<std::complex<f32>>();
    auto current_size = 2;
    uZ   n_tw         = 1;
    // uZ   n_tw     = VecCount;
    while (n_tw <= fft_size / 2) {
        for (auto i: stdv::iota(0U, n_tw)) {
            auto rk = pcx::detail_::reverse_bit_order(i, log2i(current_size) - 1);
            auto tw = pcx::detail_::wnk<f32>(current_size, rk);
            // auto tw = pcx::detail_::wnk_br<f32>(fft_size, start_offset + i);
            twvec.push_back(tw);
        }
        current_size *= 2;
        n_tw *= 2;
    }
    return twvec;
}

template<uZ VecSize, uZ VecCount>    // 32 for avx512
auto make_tw_vec_single_load(uZ start_offset = 0, uZ start_size = 2) {
    // skip steps that don't cross simd vector boundary
    auto twvec    = std::vector<std::complex<f32>>();
    auto fft_size = start_size;
    uZ   n_tw     = 1;
    // uZ   n_tw     = VecCount;
    while (n_tw <= VecSize * VecCount / 2) {
        for (auto i: stdv::iota(0U, n_tw)) {
            auto rk = pcx::detail_::reverse_bit_order(start_offset + i, log2i(fft_size) - 1);
            auto tw = pcx::detail_::wnk<f32>(fft_size, rk);
            // auto tw = pcx::detail_::wnk_br<f32>(fft_size, start_offset + i);
            twvec.push_back(tw);
        }
        start_offset *= 2;
        fft_size *= 2;
        n_tw *= 2;
    }
    return twvec;
}
template<uZ VecSize, uZ VecCount>    // 32 for avx512
auto make_tw_vec_lo_k() {
    // skip steps that don't cross simd vector boundary
    auto twvec    = std::vector<std::complex<f32>>();
    auto fft_size = VecSize * VecCount;
    uZ   n_tw     = VecSize * VecCount / 2;
    for (auto i: stdv::iota(0U, n_tw)) {
        auto rk = pcx::detail_::reverse_bit_order(i, log2i(fft_size) - 1);
        auto tw = pcx::detail_::wnk<f32>(fft_size, rk);
        twvec.push_back(tw);
    }
    return twvec;
}

void bit_reverse_sort(stdr::random_access_range auto& range) {
    auto rsize = stdr::size(range);
    if (!is_pow_of_two(rsize))
        throw std::invalid_argument("Range size is not a power of two.");
    auto depth = log2i(rsize);
    for (auto i: stdv::iota(0U, rsize)) {
        auto irev = pcx::detail_::reverse_bit_order(i, depth);
        if (i < irev)
            std::swap(range[i], range[irev]);
    }
}

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

void foo_single_load(f32* data_ptr, f32* tw_lok_ptr) {
    constexpr auto vec_size  = 16;
    constexpr auto vec_count = 8;
    using fimpl              = pcx::detail_::subtransform<vec_count, f32, vec_size>;
    fimpl::single_load<1, 16, true>(data_ptr, data_ptr, tw_lok_ptr);
}

// NOLINTEND (*pointer-arithmetic*)
int main() {
    constexpr auto tw = [] {
        auto arr = std::array<std::complex<f32>, 7>{};
        for (auto [i, v]: stdv::enumerate(arr)) {
            // v = std::complex(i, -7 + i);
            v = 1;
        }
        return arr;
    }();
    auto data_bar = [] {
        auto arr = std::array<std::complex<f32>, static_cast<std::size_t>(16 * 8)>{};
        for (auto [i, v]: stdv::enumerate(arr))
            // v = std::complex(i * i, -i * i * i);
            v = i;

        return arr;
    }();
    auto data_foo = data_bar;
    // foo(data_foo.data(), tw);
    // bar8(data_bar.data(), tw);
    // for (auto [f, b]: stdv::zip(data_foo, data_bar)) {
    //     auto v = f - b;
    //     std::print("({}, {}) ", v.real(), v.imag());
    //     std::print("({}, {}) ", f.real(), f.imag());
    //     std::print("({}, {})\n", b.real(), b.imag());
    // }
    // std::println();

    // {
    //     using fX                 = f32;
    //     constexpr auto vec_size  = 16;
    //     constexpr auto vec_count = 8;
    //     constexpr auto fft_size  = vec_size * vec_count;
    //     constexpr auto freq_n    = fft_size / 2;
    //
    //     auto twvec   = make_tw_vec_single_load<vec_size, vec_count>();
    //     auto datavec = [=]() {
    //         auto vec = std::vector<std::complex<fX>>(fft_size);
    //         for (auto [i, v]: stdv::enumerate(vec)) {
    //             v = std::exp(std::complex<fX>(0, 1)                 //
    //                          * static_cast<fX>(2)                   //
    //                          * static_cast<fX>(std::numbers::pi)    //
    //                          * static_cast<fX>(i)                   //
    //                          * static_cast<fX>(freq_n)              //
    //                          / static_cast<fX>(fft_size));
    //         }
    //         return vec;
    //     }();
    //     auto datavec2 = datavec;
    //     naive_single_load<vec_size, vec_count>(datavec.data(), twvec.data());
    //     //
    //     auto twvec_lo_k  = make_tw_vec_lo_k<vec_size, vec_count>();
    //     using fimpl      = pcx::detail_::subtransform<vec_count, f32, vec_size>;
    //     auto* data_ptr   = reinterpret_cast<fX*>(datavec2.data());
    //     auto* tw_ptr     = reinterpret_cast<fX*>(twvec.data());
    //     auto* tw_lok_ptr = reinterpret_cast<fX*>(twvec_lo_k.data());
    //     // fimpl::single_load<1, 1>(data_ptr, data_ptr, tw_ptr);
    //     fimpl::single_load<1, 1, true>(data_ptr, data_ptr, tw_lok_ptr);
    //
    //     // bit_reverse_sort(datavec);
    //     // bit_reverse_sort(datavec2);
    //     std::println();
    //
    //     auto single_load_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
    //                                           [](auto v) { return std::get<0>(v) != std::get<1>(v); });
    //
    //     if (single_load_error) {
    //         std::println("Error in single load.");
    //         return -1;
    //     }
    //     // for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2) | stdv::take(999)) {
    //     //     std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
    //     //                  i,
    //     //                  (naive),
    //     //                  (pcx),
    //     //                  (naive - pcx));
    //     // }
    // }
    {
        using fX      = f32;
        uZ   fft_size = 256;
        uZ   freq_n   = fft_size / 2;
        auto datavec  = [=]() {
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
        naive_fft(datavec);

        constexpr auto vec_size  = 16;
        constexpr auto vec_count = 8;

        using fimpl = pcx::detail_::subtransform<vec_count, f32, vec_size>;
        auto twvec  = make_tw_vec2(fft_size);
        // template<uZ DestPackSize, uZ SrcPackSize, bool LowK>
        auto* data_ptr = reinterpret_cast<fX*>(datavec2.data());
        auto* tw_ptr   = reinterpret_cast<fX*>(twvec.data());
        fimpl::perform<1, 1, false>(2, fft_size, data_ptr, tw_ptr);
        auto subtform_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
                                           [](auto v) { return std::get<0>(v) != std::get<1>(v); });

        if (subtform_error) {
            for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2) | stdv::take(999)) {
                std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                             i,
                             (naive),
                             (pcx),
                             (naive - pcx));
                // }
            }
            return -1;
        }
        std::println("subtform successful");
    }
    return 0;
}
