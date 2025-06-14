#pragma once
#include "pcx/include/fft/fft_impl.hpp"
#include "pcx/include/fft/fft_interface_common.hpp"

#include <variant>
#include <vector>

namespace pcx {

/**
 * @brief FFT plan for performing fft over sequential data.
 */
template<floating_point T, fft_options Opts = {}>
class par_fft_plan {
    static constexpr auto lowk           = std::true_type{};
    static constexpr auto half_tw        = std::false_type{};
    static constexpr auto bit_reversed   = std::bool_constant<Opts.pt == fft_permutation::bit_reversed>{};
    static constexpr auto not_sequential = std::false_type{};
    static constexpr auto width = uZ_ce<Opts.simd_width != 0 ? Opts.simd_width : simd::max_width<T>>{};
    static constexpr auto coherent_size = uZ_ce<Opts.coherent_size != 0    //
                                                    ? Opts.coherent_size
                                                    : 8192 / sizeof(T)>{};
    static constexpr auto lane_size     = uZ_ce<std::max(Opts.lane_size != 0    //
                                                         ? Opts.lane_size
                                                         : 64 / sizeof(T) / 2,
                                                         uZ(width))>{};

    using permuter_t = std::conditional_t<bit_reversed,
                                          detail_::identity_permuter_t,
                                          std::conditional_t<Opts.pt == fft_permutation::normal,
                                                             detail_::br_permuter<Opts.node_size>,
                                                             detail_::br_permuter_shifted<Opts.node_size>>>;

    using tw_t           = std::vector<T>;
    using permute_idxs_t = std::conditional_t<bit_reversed, decltype([] {}), std::vector<u32>>;

public:
    explicit par_fft_plan(uZ fft_size);

    [[nodiscard]] auto fft_size() const -> uZ {
        return fft_size_;
    }

    void fft_raw(std::complex<T>* data_ptr, uZ stride, uZ data_size) const {
        (this->*inplace_ptr_)(reinterpret_cast<T*>(data_ptr), stride, data_size);
    };

    template<stdr::random_access_range R>
        requires stdr::contiguous_range<stdr::range_value_t<R>>
                 && std::same_as<std::complex<T>,    //
                                 stdr::range_value_t<stdr::range_value_t<R>>>
    void fft(R& data) const {
        if (stdr::size(data) != fft_size_)
            throw std::runtime_error("Range size not equal to fft size");
        using impl_t           = detail_::transform<Opts.node_size, T, width, coherent_size, lane_size>;
        constexpr auto dst_pck = cxpack<1, T>{};
        constexpr auto src_pck = cxpack<1, T>{};

        auto data_size = stdr::size(data[0]);
        for (auto [i, slice]: stdv::enumerate(data) | stdv::drop(1)) {
            if (stdr::size(slice) != data_size)
                throw std::runtime_error("Subrange sizes not equal");
        }
        auto dst_data = detail_::data_info<T, false, R>{.data_ptr = &data};
        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = tw_.data()};

        impl_t::perform_auto_size(dst_pck,
                                  src_pck,
                                  half_tw,
                                  lowk,
                                  dst_data,
                                  detail_::inplace_src,
                                  fft_size_,
                                  tw_data,
                                  permuter_,
                                  data_size);
    }
    template<stdr::random_access_range R>
        requires stdr::contiguous_range<stdr::range_value_t<R>>
                 && std::same_as<std::complex<T>,    //
                                 stdr::range_value_t<stdr::range_value_t<R>>>
    void ifft(R& data) const {
        if (stdr::size(data) != fft_size_)
            throw std::runtime_error("Range size not equal to fft size");
        using impl_t           = detail_::transform<Opts.node_size, T, width, coherent_size, lane_size>;
        constexpr auto dst_pck = cxpack<1, T>{};
        constexpr auto src_pck = cxpack<1, T>{};

        auto data_size = stdr::size(data[0]);
        for (auto [i, slice]: stdv::enumerate(data) | stdv::drop(1)) {
            if (stdr::size(slice) != data_size)
                throw std::runtime_error("Subrange sizes not equal");
        }
        auto dst_data = detail_::data_info<T, false, R>{.data_ptr = &data[0]};
        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = &*tw_.end()};

        impl_t::perform_rev_auto_size(dst_pck,
                                      src_pck,
                                      half_tw,
                                      lowk,
                                      dst_data,
                                      detail_::inplace_src,
                                      fft_size_,
                                      tw_data,
                                      permuter_,
                                      data_size);
    }

private:
    using inplace_impl_t  = auto (par_fft_plan::*)(T*, uZ, uZ) const -> void;
    using external_impl_t = auto (par_fft_plan::*)(T*, uZ, const T*, uZ, uZ) const -> void;
    uZ                                   fft_size_;
    tw_t                                 tw_{};
    [[no_unique_address]] permute_idxs_t idxs_{};
    [[no_unique_address]] permuter_t     permuter_{};
    uZ                                   align{};
    inplace_impl_t                       inplace_ptr_;
    inplace_impl_t                       inplace_r_ptr_;
    external_impl_t                      external_ptr_;
    external_impl_t                      external_r_ptr_;


    template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
    void inplace(T* dst, uZ stride, uZ data_size) const;
    template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
    void external(T* dst, uZ dst_stride, const T* src, uZ src_stride, uZ data_size) const;
    template<uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
    void inplace_coh(T* dst, uZ stride, uZ data_size) const;
    template<uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
    void external_coh(T* dst, uZ dst_stride, const T* src, uZ src_stride, uZ data_size) const;
    template<uZ NodeSize, uZ DstPck, uZ SrcPck, bool Reverse>
    void inplace_single_node(T* dst, uZ stride, uZ data_size) const;
    template<uZ NodeSize, uZ DstPck, uZ SrcPck, bool Reverse>
    void external_single_node(T* dst, uZ dst_stride, const T* src, uZ src_stride, uZ data_size) const;

    template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
    PCX_AINLINE void impl(detail_::data_info_for<T> auto       dst_data,
                          detail_::data_info_for<const T> auto src_data,
                          uZ                                   data_size) const;
    template<bool SingleNode, uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
    PCX_AINLINE void coh_impl(detail_::data_info_for<T> auto       dst_data,
                              detail_::data_info_for<const T> auto src_data,
                              uZ                                   data_size) const;
};
}    // namespace pcx
