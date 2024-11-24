// #define PCX_AVX512
#include "pcx/include/fft_impl.hpp"

#include <complex>
#include <print>
#include <ranges>
namespace stdv = std::views;


using namespace pcx;

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
    // *a        = a_c + b_tw;
    // *b        = a_c - b_tw;
    *b = b_tw;
}


};    // namespace pcxt

void foo(std::complex<f32>* dest, auto&& tw) {
    auto dest_r     = reinterpret_cast<f32*>(dest);
    using node_t    = detail_::btfly_node<8, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * Is)...);
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
        .conj_tw   = false,
        .dit       = true,
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

        // pcxt::btfly(start + 0 * VecSize, start + 2 * VecSize, tw[1]);
        // pcxt::btfly(start + 1 * VecSize, start + 3 * VecSize, tw[1]);
        // pcxt::btfly(start + 4 * VecSize, start + 6 * VecSize, tw[2]);
        // pcxt::btfly(start + 5 * VecSize, start + 7 * VecSize, tw[2]);
        //
        // pcxt::btfly(start + 0 * VecSize, start + 1 * VecSize, tw[3]);
        // pcxt::btfly(start + 2 * VecSize, start + 3 * VecSize, tw[4]);
        // pcxt::btfly(start + 4 * VecSize, start + 5 * VecSize, tw[5]);
        // pcxt::btfly(start + 6 * VecSize, start + 7 * VecSize, tw[6]);
    }
};

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
        auto arr = std::array<std::complex<f32>, 16 * 8>{};
        for (auto [i, v]: stdv::enumerate(arr))
            // v = std::complex(i * i, -i * i * i);
            v = i;

        return arr;
    }();
    auto data_foo = data_bar;
    foo(data_foo.data(), tw);
    bar8(data_bar.data(), tw);
    for (auto [f, b]: stdv::zip(data_foo, data_bar)) {
        auto v = f - b;
        std::print("({}, {}) ", v.real(), v.imag());
        std::print("({}, {}) ", f.real(), f.imag());
        std::print("({}, {})\n", b.real(), b.imag());
    }
    std::println();
    return 0;
}
