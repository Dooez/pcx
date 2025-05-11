#pragma once
#include "pcx/include/fft/fft_impl.hpp"

#include <vector>


namespace pcx {

enum class fft_permutation {
    bit_reversed,
    normal,
    shifted,
};
struct fft_options {
    fft_permutation pt = fft_permutation::normal;

    uZ coherent_size = 0;
    uZ lane_size     = 0;
    uZ node_size     = 0;
};

/**
 * @brief FFT plan for performing fft over sequential data.
 */
template<floating_point T, fft_options Opts = {}>
class fft_plan {
    static constexpr auto lowk         = std::true_type{};
    static constexpr auto half_tw      = std::true_type{};
    static constexpr auto bit_reversed = std::bool_constant<Opts.pt == fft_permutation::bit_reversed>{};

public:
private:
    using member_impl_t  = auto (fft_plan::*)(T*) -> void;
    using permute_idxs_t = std::conditional_t<bit_reversed, decltype([] {}), std::vector<u32>>;

    uZ                                   fft_size{};
    std::vector<T>                       tw{};
    [[no_unique_address]] permute_idxs_t idxs{};
    member_impl_t                        ileave_impl_ptr;

    void tform_inplace_ileave(T* dst);
    void coherent_tform_inplace_ileave(T* dst);
    void single_load_tform_inplace_ileave(T* dst);

    template<uZ Width, uZ DstPck, uZ SrcPck>
    void tform_inplace(T* dst) {
        using impl_t           = detail_::transform<Opts.node_size, T, Width, Opts.coherent_size, 0>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto reverse = std::false_type{};
        constexpr auto conj_tw = std::false_type{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = tw.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = 1};
        impl_t::perform(dst_pck,
                        src_pck,
                        half_tw,
                        lowk,
                        dst_data,
                        detail_::inplace_src,
                        fft_size,
                        tw_data,
                        detail_::identity_permuter);
    }


    template<uZ Width, uZ DstPck, uZ SrcPck, uZ Align, uZ DataSize>
    void coherent_tform_inplace(T* dst) {
        using impl_t             = detail_::sequential_subtransform<Opts.node_size, T, Width>;
        constexpr auto dst_pck   = cxpack<DstPck, T>{};
        constexpr auto src_pck   = cxpack<SrcPck, T>{};
        constexpr auto align     = detail_::align_param<Align, true>{};
        constexpr auto reverse   = std::false_type{};
        constexpr auto conj_tw   = std::false_type{};
        constexpr auto data_size = uZ_ce<DataSize>{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = tw.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};
        impl_t::perform_impl(dst_pck,
                             src_pck,
                             align,
                             lowk,
                             half_tw,
                             reverse,
                             conj_tw,
                             data_size,
                             dst_data,
                             detail_::inplace_src,
                             tw_data);
    }

    template<uZ Width, uZ NodeSize, uZ DstPck, uZ SrcPck>
    void single_load_tform_inplace(T* dst) {
        using impl_t           = detail_::sequential_subtransform<NodeSize, T, Width>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto reverse = std::false_type{};
        constexpr auto conj_tw = std::false_type{};

        auto tw_data = detail_::tw_data_t<T, false>{.tw_ptr = tw.data()};
        impl_t::single_load(dst_pck, src_pck, lowk, half_tw, conj_tw, reverse, dst, dst, tw_data);
    }
};


}    // namespace pcx
