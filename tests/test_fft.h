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
static auto get_type_name(pcx::meta::types<f32>) {
    return std::string_view("f32");
}
static auto get_type_name(pcx::meta::types<f64>) {
    return std::string_view("f64");
}
template<typename T>
struct std::formatter<pcx::meta::types<T>> {
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        auto it = ctx.begin();
        if (it == ctx.end())
            return it;
        if (it != ctx.end() && *it != '}')
            throw std::format_error("Invalid format args for pcx::meta::types<T>.");

        return it;
    }

    template<class FmtContext>
    FmtContext::iterator format(pcx::meta::types<T> mt, FmtContext& ctx) const {
        return std::ranges::copy(get_type_name(mt), ctx.out()).out;
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

    auto mulr = simd::mul(va, vb);
    auto vres = simd::repack<1>(mulr);
    simd::cxstore<1>(resptr, vres);
    return res[0];
}
template<typename T>
void btfly(std::complex<T>* a, std::complex<T>* b, std::complex<T> tw) {
    using ct = std::complex<T>;

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

template<typename fX>
auto make_tw_vec_single_load(uZ start_offset,    // = 0
                             uZ start_size,      // = 1
                             uZ vec_width,
                             uZ node_size) {
    auto tw_vec    = std::vector<std::complex<fX>>();
    auto insert_tw = [&](uZ size, uZ i) {
        auto rk = pcx::detail_::reverse_bit_order(i, log2i(size) - 1);
        auto tw = pcx::detail_::wnk<fX>(size, rk);
        tw_vec.push_back(tw);
    };
    // auto fft_size = start_size;
    // uZ   n_tw     = 1;
    // while (n_tw <= VecSize * VecCount / 2) {
    //     for (auto i: stdv::iota(0U, n_tw)) {
    //         auto rk = pcx::detail_::reverse_bit_order(start_offset + i, log2i(fft_size) - 1);
    //         auto tw = pcx::detail_::wnk<fX>(fft_size, rk);
    //         twvec.push_back(tw);
    //     }
    //     start_offset *= 2;
    //     fft_size *= 2;
    //     n_tw *= 2;
    // }
    // return twvec;
    //
    auto fft_size = start_size * 2;
    for (auto i_node: stdv::iota(0U, log2i(node_size))) {
        for (auto k: stdv::iota(0U, powi(2, i_node))) {
            if (k % 2 == 1) {
                continue;
            }
            insert_tw(fft_size, start_offset + k);
        }
        start_offset *= 2;
        fft_size *= 2;
    }
    uZ tw_per_vec = 2;
    while (tw_per_vec <= vec_width) {
        uZ tw_idx = start_offset;
        for (auto i_vec: stdv::iota(0U, node_size / 2)) {
            for (auto i_tw: stdv::iota(0U, tw_per_vec)) {
                insert_tw(fft_size, tw_idx);
                ++tw_idx;
            }
        }
        start_offset *= 2;
        tw_per_vec *= 2;
        fft_size *= 2;
    }
    return tw_vec;
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
            uZ   start = k * step * 2;
            auto rk    = pcx::detail_::reverse_bit_order(k, log2i(fft_size) - 1);
            auto tw    = pcx::detail_::wnk<T>(fft_size, rk);
            for (uZ i = 0; i < step; ++i) {
                // pcxt::btfly(data + start + i, data + start + i + step, *tw_ptr);    //
                pcxt::btfly(data + start + i, data + start + i + step, tw);    //
            }
            // ++tw_ptr;
        }
        step /= 2;
        n_groups *= 2;
        fft_size *= 2;
    }
}
template<typename fX, uZ Width, uZ NodeSize, bool LowK = false>
int test_single_load() {
    constexpr auto fft_size = Width * NodeSize;
    auto           freq_n   = fft_size / 2;

    auto twvec   = make_tw_vec_single_load<fX>(0, 1, Width, NodeSize);
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
    naive_single_load<Width, NodeSize>(datavec.data(), twvec.data());
    //
    // auto twvec_lo_k  = make_tw_vec_lo_k<vec_size, vec_count>();
    using fimpl    = pcx::detail_::subtransform<NodeSize, fX, Width>;
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
        std::println("Error in single load for {} simd vectors of width {}.", NodeSize, Width);
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2)) {
            std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                         i,
                         (naive),
                         (pcx),
                         (naive - pcx));
        }
        return -1;
    }

    std::println("Successful single load for {} simd vectors of width {}.", NodeSize, Width);
    // for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2)) {
    //     std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
    //                  i,
    //                  (naive),
    //                  (pcx),
    //                  (naive - pcx));
    // }
    return 0;
};
template<typename fX>
void naive_fft(std::vector<std::complex<fX>>& data);
template<typename fX>
auto make_tw_vec(uZ fft_size, uZ vec_width, uZ node_size, bool low_k) -> std::vector<std::complex<fX>>;
template<typename fX>
auto make_tw_vec_lok(uZ fft_size, uZ vec_width, uZ node_size, bool low_k) -> std::vector<std::complex<fX>>;

// void bit_reverse_sort(stdr::random_access_range auto& range) {
//     auto rsize = stdr::size(range);
//     if (!is_pow_of_two(rsize))
//         throw std::invalid_argument("Range size is not a power of two.");
//     auto depth = log2i(rsize);
//     for (auto i: stdv::iota(0U, rsize)) {
//         auto irev = pcx::detail_::reverse_bit_order(i, depth);
//         if (i < irev)
//             std::swap(range[i], range[irev]);
//     }
// }
template<typename fX, uZ Width, uZ NodeSize, bool LowK = true>
int test_subtranform(uZ fft_size) {
    // uZ   freq_n  = 2;
    uZ   freq_n  = 1;
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
    naive_fft(datavec);

    using fimpl = pcx::detail_::subtransform<NodeSize, fX, Width>;
    auto twvec  = make_tw_vec<fX>(fft_size, Width, NodeSize, LowK);
    // auto  twvec    = make_tw_vec_lok<fX>(fft_size, Width, NodeSize);
    auto* data_ptr = reinterpret_cast<fX*>(datavec2.data());
    auto* tw_ptr   = reinterpret_cast<fX*>(twvec.data());
    // fimpl::template perform<1, 1, false>(2, fft_size, data_ptr, tw_ptr);
    fimpl::template perform<1, 1, LowK>(1, fft_size, data_ptr, tw_ptr);
    auto subtform_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
                                       [](auto v) { return std::get<0>(v) != std::get<1>(v); });

    if (subtform_error) {
        std::println(
            "Error during {} subtform of size {} of type {} with vector width {:>2} and node size {:>2}.",
            LowK ? "low k" : "",
            fft_size,
            pcx::meta::types<fX>{},
            Width,
            NodeSize);
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2)) {
            // if (naive != pcx)
                std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                             i,
                             (naive),
                             (pcx),
                             (naive - pcx));
            if (std::abs(naive - pcx) > 1) {
                // // std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                // //              i,
                // //              (naive),
                // //              (pcx),
                // //              (naive - pcx));
                // std::println("Over 1 found.");
                // break;
            }
            // }
        }
        return -1;
    }
    std::println("Successful {} subtform of size {} of type {} with width {:>2} and node size {:>2}.",
                 LowK ? "low k" : "",
                 fft_size,
                 pcx::meta::types<fX>{},
                 Width,
                 NodeSize);
    return 0;
}
