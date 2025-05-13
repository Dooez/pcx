#include "pcx/include/fft/fft_impl.hpp"

#include <vector>

using namespace pcx;

auto main() -> int {
    using impl_t            = detail_::sequential_subtransform<2, f32, 1>;
    auto           tw       = std::vector<f32>{};
    auto           tw_data0 = detail_::tw_data_t<f32, true>{};
    constexpr auto lowk     = std::true_type{};
    constexpr auto half_tw  = std::true_type{};
    impl_t::insert_single_load_tw(tw, tw_data0, lowk, half_tw);

    using T                = f32;
    constexpr auto dst_pck = cxpack<1, T>{};
    constexpr auto src_pck = cxpack<1, T>{};
    constexpr auto reverse = std::false_type{};
    constexpr auto conj_tw = std::false_type{};

    auto data = std::vector<std::complex<T>>(1024);
    auto dst  = reinterpret_cast<f32*>(data.data());

    auto tw_data = detail_::tw_data_t<T, false>{.tw_ptr = tw.data()};
    impl_t::single_load(dst_pck, src_pck, lowk, half_tw, conj_tw, reverse, dst, dst, tw_data);

    return 0;
}
