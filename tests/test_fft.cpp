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

    auto rva = simd::repack<vec_t::width()>(va);
    auto rvb = simd::repack<vec_t::width()>(vb);

    using resarr = std::array<std::complex<T>, vec_t::width()>;
    auto res     = resarr{};
    auto resptr  = reinterpret_cast<T*>(res.data());

    auto vres = simd::repack<1>(simd::mul(rva, rvb));
    simd::cxstore<1>(resptr, vres);
}
template<typename T>
void btfly(std::complex<T>* a, std::complex<T>* b, std::complex<T> tw) {
    auto b_tw = cmul(*b, tw);
    auto a_c  = *a;
    *a        = a_c + b_tw;
    *b        = a_c - b_tw;
}


};    // namespace pcxt

void foo(f32* dest, const f32* tw) {
    using node_t    = detail_::btfly_node<4, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * Is)...);
    }(dest, std::make_index_sequence<4>());
    auto tw_tuple = []<uZ... Is>(auto* tw, std::index_sequence<Is...>) {
        return tupi::make_tuple(simd::cxload<16>(tw + 16 * Is)...);
    }(tw, std::make_index_sequence<3>{});

    constexpr auto settings = node_t::settings{
        .pack_dest = 1,
        .pack_src  = 1,
        .conj_tw   = false,
        .dit       = false,
    };
    node_t::perform<settings>(dest_tuple, tw_tuple);
};

template<uZ VecSize>
void bar8(std::complex<f32>* dest, const std::complex<f32>* tw) {
    for (auto i: stdv::iota(VecSize)) {
        auto* start = dest + i;

        pcxt::btfly(start + 0 * VecSize, start + 4 * VecSize, tw[0]);
        pcxt::btfly(start + 1 * VecSize, start + 5 * VecSize, tw[0]);
        pcxt::btfly(start + 2 * VecSize, start + 6 * VecSize, tw[0]);
        pcxt::btfly(start + 3 * VecSize, start + 7 * VecSize, tw[0]);

        pcxt::btfly(start + 0 * VecSize, start + 2 * VecSize, tw[1]);
        pcxt::btfly(start + 1 * VecSize, start + 3 * VecSize, tw[1]);
        pcxt::btfly(start + 4 * VecSize, start + 6 * VecSize, tw[2]);
        pcxt::btfly(start + 5 * VecSize, start + 7 * VecSize, tw[2]);

        pcxt::btfly(start + 0 * VecSize, start + 1 * VecSize, tw[3]);
        pcxt::btfly(start + 2 * VecSize, start + 3 * VecSize, tw[4]);
        pcxt::btfly(start + 4 * VecSize, start + 5 * VecSize, tw[5]);
        pcxt::btfly(start + 6 * VecSize, start + 7 * VecSize, tw[6]);
    }
};

int main() {
    auto data0 = []<uZ... Is>(std::index_sequence<Is...>) {
        return std::array{f32{Is}...};
    }(std::make_index_sequence<16 * 8 * 2>{});
    auto data1 = data0;
    auto tw    = std::array<float, 16 * 8 * 2>{};
    tw.fill(1);
    foo(data0.data(), tw.data());
    for (auto v: data0) {
        std::print("{} ", v);
    }
    std::println();
    return 0;
}
