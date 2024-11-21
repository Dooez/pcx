// #define PCX_AVX512
#include "pcx/include/fft_impl.hpp"

#include <print>


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
