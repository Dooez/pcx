// #define PCX_AVX512
#include "pcx/include/fft_impl.hpp"

#include <print>
#include <ranges>
namespace stdv = std::views;


using namespace pcx;
void foo(f32* dest, const f32* tw) {
    using node_t    = detail_::btfly_node<4, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * Is)...);
    }(dest, std::make_index_sequence<4>());
    auto tw_tuple = []<uZ... Is>(auto* tw, std::index_sequence<Is...>) {
        return tupi::make_tuple(simd::cxload<16>(tw + 16 * Is)...);
    }(tw, std::make_index_sequence<3>{});

    constexpr auto settings = node_t::settings{
        .pack_dest = 16,
        .pack_src  = 16,
        .conj_tw   = false,
        .dit       = false,
    };
    node_t::perform<settings>(dest_tuple, tw_tuple);
};

auto direct_btfly_0(auto dest, auto tw) {
    auto a0 = *get<0>(dest);
    auto a1 = *get<1>(dest);
    auto a2 = *get<2>(dest);
    auto a3 = *get<3>(dest);

    auto b0 = *get<4>(dest);
    auto b1 = *get<5>(dest);
    auto b2 = *get<6>(dest);
    auto b3 = *get<7>(dest);

    b0 *= get<0>(tw);
    b1 *= get<0>(tw);
    b2 *= get<0>(tw);
    b3 *= get<0>(tw);

    auto bf0 = a0 + b0;
    auto bf1 = a1 + b1;
    auto bf2 = a2 + b2;
    auto bf3 = a3 + b3;

    auto bf4 = a0 - b0;
    auto bf5 = a1 - b1;
    auto bf6 = a2 - b2;
    auto bf7 = a3 - b3;

    *get<0>(dest) = bf0;
    *get<1>(dest) = bf1;
    *get<2>(dest) = bf2;
    *get<3>(dest) = bf3;
    *get<4>(dest) = bf4;
    *get<5>(dest) = bf5;
    *get<6>(dest) = bf6;
    *get<7>(dest) = bf7;
}
auto direct_btfly_1(auto dest, auto tw) {
    auto a0 = *get<0>(dest);
    auto a1 = *get<1>(dest);
    auto a2 = *get<4>(dest);
    auto a3 = *get<5>(dest);

    auto b0 = *get<2>(dest);
    auto b1 = *get<3>(dest);
    auto b2 = *get<6>(dest);
    auto b3 = *get<7>(dest);

    b0 *= get<0>(tw);
    b1 *= get<0>(tw);
    b2 *= get<1>(tw);
    b3 *= get<2>(tw);

    auto bf0 = a0 + b0;
    auto bf1 = a1 + b1;
    auto bf2 = a0 - b0;
    auto bf3 = a1 - b1;

    auto bf4 = a2 + b2;
    auto bf5 = a3 + b3;
    auto bf6 = a2 - b2;
    auto bf7 = a3 - b3;

    *get<0>(dest) = bf0;
    *get<1>(dest) = bf1;
    *get<2>(dest) = bf2;
    *get<3>(dest) = bf3;
    *get<4>(dest) = bf4;
    *get<5>(dest) = bf5;
    *get<6>(dest) = bf6;
    *get<7>(dest) = bf7;
}
auto direct_btfly_2(auto dest, auto tw) {
    auto a0 = *get<0>(dest);
    auto a1 = *get<2>(dest);
    auto a2 = *get<4>(dest);
    auto a3 = *get<6>(dest);

    auto b0 = *get<1>(dest);
    auto b1 = *get<3>(dest);
    auto b2 = *get<5>(dest);
    auto b3 = *get<7>(dest);

    b0 *= get<0>(tw);
    b1 *= get<1>(tw);
    b2 *= get<2>(tw);
    b3 *= get<3>(tw);

    auto bf0 = a0 + b0;
    auto bf1 = a0 - b0;
    auto bf2 = a1 + b1;
    auto bf3 = a1 - b1;

    auto bf4 = a2 + b2;
    auto bf5 = a2 - b2;
    auto bf6 = a3 + b3;
    auto bf7 = a3 - b3;

    *get<0>(dest) = bf0;
    *get<1>(dest) = bf1;
    *get<2>(dest) = bf2;
    *get<3>(dest) = bf3;
    *get<4>(dest) = bf4;
    *get<5>(dest) = bf5;
    *get<6>(dest) = bf6;
    *get<7>(dest) = bf7;
}

void foo(f32* dest, const f32* tw) {
    using node_t    = detail_::btfly_node<4, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * Is)...);
    }(dest, std::make_index_sequence<4>());
    auto tw_tuple = []<uZ... Is>(auto* tw, std::index_sequence<Is...>) {
        return tupi::make_tuple(simd::cxload<16>(tw + 16 * Is)...);
    }(tw, std::make_index_sequence<3>{});

    constexpr auto settings = node_t::settings{
        .pack_dest = 16,
        .pack_src  = 16,
        .conj_tw   = false,
        .dit       = false,
    };
    node_t::perform<settings>(dest_tuple, tw_tuple);
};

void bar(f32* dest, const f32* tw) {
    using node_t    = detail_::btfly_node<4, float, 16>;
    auto dest_tuple = []<uZ... Is>(auto* dest, std::index_sequence<Is...>) {
        return pcx::tupi::make_tuple((dest + 16 * Is)...);
    }(dest, std::make_index_sequence<4>());
    auto tw_tuple = []<uZ... Is>(auto* tw, std::index_sequence<Is...>) {
        return tupi::make_tuple(simd::cxload<16>(tw + 16 * Is)...);
    }(tw, std::make_index_sequence<3>{});

    constexpr auto settings = node_t::settings{
        .pack_dest = 16,
        .pack_src  = 16,
        .conj_tw   = false,
        .dit       = false,
    };
    node_t::perform<settings>(dest_tuple, tw_tuple);
};


int main() {
    auto data = []<uZ... Is>(std::index_sequence<Is...>) {
        return std::array{f32{Is}...};
    }(std::make_index_sequence<16 * 8 * 2>{});
    auto tw = std::array<float, 16 * 8 * 2>{};
    tw.fill(1);
    foo(data.data(), tw.data());
    for (auto v: data) {
        std::print("{} ", v);
    }
    std::println();
    return 0;
}
