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
class fft_plan {
    static constexpr auto lowk         = val_ce<true>{};
    static constexpr auto half_tw      = val_ce<false>{};
    static constexpr auto bit_reversed = val_ce<Opts.pt == fft_permutation::bit_reversed>{};
    static constexpr auto sequential   = val_ce<true>{};
    static constexpr auto coherent_size =
        uZ_ce<Opts.coherent_size != 0 ? Opts.coherent_size : 8192 / sizeof(T)>{};
    static constexpr auto width = uZ_ce<Opts.simd_width != 0 ? Opts.simd_width : simd::max_width<T>>{};
    static constexpr auto max_perm_width = std::min(Opts.node_size, width.value);

    // The sequential permutation requires that transform size is not less
    // than the square of the SIMD vector width.
    // Permutation of transforms larger than `coherent_size` is assumed to use maximum width
    // because `coherent_size` is expected to be significantly bigger than `width`.
    // This assert is required to check for an oddly specified `Opts`.
    // Sizes smaller than `coherent_size` are handled explicitly.
    static_assert(coherent_size >= width * width);

    template<uZ AlignNode>
    using align_param = detail_::align_param<AlignNode, true>;
    template<uZ Width>
    using permuter_t =
        std::conditional_t<bit_reversed,
                           detail_::identity_permuter_t,
                           std::conditional_t<Opts.pt == fft_permutation::normal,
                                              detail_::br_permuter_sequential<max_perm_width, false>,
                                              detail_::br_permuter_sequential<max_perm_width, true>>>;
    using tw_t = std::vector<T>;

public:
    explicit fft_plan(uZ fft_size);

    template<stdr::contiguous_range R>
        requires std::same_as<std::complex<T>, stdr::range_value_t<R>>
    void fft(R& data) const {
        if (stdr::size(data) != fft_size_)
            throw std::runtime_error("Data size not equal to fft size");

        auto dst_ptr = reinterpret_cast<T*>(stdr::data(data));
        (this->*ileave_inplace_ptr_)(dst_ptr);
    }
    template<stdr::contiguous_range Dst, stdr::contiguous_range Src>
        requires std::same_as<std::complex<T>, stdr::range_value_t<Dst>>
                 && std::same_as<std::complex<T>, stdr::range_value_t<Src>>
    void fft(Dst& dst, const Src& src) const {
        if (src.size() != fft_size_)
            throw std::runtime_error("Source size not equal to fft size");
        if (dst.size() != fft_size_)
            throw std::runtime_error("Destination size not equal to fft size");

        auto dst_ptr = reinterpret_cast<T*>(stdr::data(dst));
        auto src_ptr = reinterpret_cast<const T*>(stdr::data(src));
        if (src_ptr == dst_ptr)
            return (this->*ileave_inplace_ptr_)(dst_ptr);
        return (this->*ileave_external_ptr_)(dst_ptr, src_ptr);
    }

    template<stdr::contiguous_range R>
        requires std::same_as<std::complex<T>, stdr::range_value_t<R>>
    void ifft(R& data) const {
        if (stdr::size(data) != fft_size_)
            throw std::runtime_error("Data size not equal to fft size");

        auto dst_ptr = reinterpret_cast<T*>(stdr::data(data));
        (this->*ileave_inplace_r_ptr_)(dst_ptr);
    }
    template<stdr::contiguous_range Dst, stdr::contiguous_range Src>
        requires std::same_as<std::complex<T>, stdr::range_value_t<Dst>>
                 && std::same_as<std::complex<T>, stdr::range_value_t<Src>>
    void ifft(Dst& dst, const Src& src) const {
        if (src.size() != fft_size_)
            throw std::runtime_error("Source size not equal to fft size");
        if (dst.size() != fft_size_)
            throw std::runtime_error("Destination size not equal to fft size");

        auto dst_ptr = reinterpret_cast<T*>(stdr::data(dst));
        auto src_ptr = reinterpret_cast<const T*>(stdr::data(src));
        if (src_ptr == dst_ptr)
            return (this->*ileave_inplace_r_ptr_)(dst_ptr);
        return (this->*ileave_external_r_ptr_)(dst_ptr, src_ptr);
    }

private:
    using inplace_impl_t  = auto (fft_plan::*)(T*) const -> void;
    using external_impl_t = auto (fft_plan::*)(T*, const T*) const -> void;
    using permute_idxs_t  = std::conditional_t<bit_reversed, decltype([] {}), std::vector<u32>>;
    using permuter_var_t  = decltype([] {
        using namespace detail_;
        if constexpr (bit_reversed) {
            return identity_permuter;
        } else {
            return []<uZ... Ps>(uZ_seq<Ps...>) {
                if constexpr (Opts.pt == fft_permutation::normal) {
                    return std::variant<br_permuter_sequential<powi(2, Ps), false>...>{};
                } else if constexpr (Opts.pt == fft_permutation::shifted) {
                    return std::variant<br_permuter_sequential<powi(2, Ps), true>...>{};
                }
            }(make_uZ_seq<log2i(max_perm_width) + 1>{});
        }
    }());

    uZ                                   fft_size_{};
    tw_t                                 tw_{};
    [[no_unique_address]] permute_idxs_t idxs_{};
    [[no_unique_address]] permuter_var_t permuter_{};
    inplace_impl_t                       ileave_inplace_ptr_;
    inplace_impl_t                       ileave_inplace_r_ptr_;
    external_impl_t                      ileave_external_ptr_;
    external_impl_t                      ileave_external_r_ptr_;

    void identity_tform(T* dst) const {};

    template<uZ Width, uZ Align>
    void tform_inplace_ileave(T* dst) const;
    template<uZ Width, uZ Align>
    void rtform_inplace_ileave(T* dst) const;
    template<uZ Width, uZ PermWidth, uZ Align>
    void coherent_tform_inplace_ileave(T* dst) const;
    template<uZ Width, uZ PermWidth, uZ Align>
    void coherent_rtform_inplace_ileave(T* dst) const;
    template<uZ Width, uZ NodeSize>
    void single_load_tform_inplace_ileave(T* dst) const;
    template<uZ Width, uZ NodeSize>
    void single_load_rtform_inplace_ileave(T* dst) const;

    template<uZ Width, uZ Align>
    void tform_external_ileave(T* dst, const T* src) const;
    template<uZ Width, uZ Align>
    void rtform_external_ileave(T* dst, const T* src) const;
    template<uZ Width, uZ PermWidth, uZ Align>
    void coherent_tform_external_ileave(T* dst, const T* src) const;
    template<uZ Width, uZ PermWidth, uZ Align>
    void coherent_rtform_external_ileave(T* dst, const T* src) const;
    template<uZ Width, uZ NodeSize>
    void single_load_tform_external_ileave(T* dst, const T* src) const;
    template<uZ Width, uZ NodeSize>
    void single_load_rtform_external_ileave(T* dst, const T* src) const;

    template<uZ Width, uZ DstPck, uZ SrcPck, uZ Align>
    void tform_inplace(T* dst, meta::ce_of<bool> auto reverse) const {
        using impl_t             = detail_::transform<Opts.node_size, T, Width, Opts.coherent_size, 0>;
        constexpr auto dst_pck   = cxpack<DstPck, T>{};
        constexpr auto src_pck   = cxpack<SrcPck, T>{};
        constexpr auto align     = detail_::align_param<Align, true>{};
        constexpr auto conj_tw   = std::bool_constant<reverse>{};
        constexpr auto PermWidth = std::min(Width, max_perm_width);

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = 1};

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        if constexpr (!reverse) {
            impl_t::perform_tf(dst_pck,
                               src_pck,
                               half_tw,
                               lowk,
                               align,
                               dst_data,
                               detail_::inplace_src,
                               fft_size_,
                               tw_data,
                               permuter);
        } else {
            impl_t::perform_tf_rev(dst_pck,
                                   src_pck,
                                   half_tw,
                                   lowk,
                                   align,
                                   dst_data,
                                   detail_::inplace_src,
                                   fft_size_,
                                   tw_data,
                                   permuter);
        }
    }

    template<uZ Width, uZ PermWidth, uZ DstPck, uZ SrcPck, uZ Align>
    void coherent_tform_inplace(T* dst, meta::ce_of<bool> auto reverse) const {
        using impl_t           = detail_::sequential_subtransform<Opts.node_size, T, Width>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto align   = detail_::align_param<Align, true>{};
        constexpr auto conj_tw = std::bool_constant<reverse>{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        if constexpr (reverse)
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, detail_::inplace_src);
        impl_t::perform_seq(dst_pck,
                            src_pck,
                            align,
                            lowk,
                            half_tw,
                            reverse,
                            conj_tw,
                            fft_size_,
                            dst_data,
                            detail_::inplace_src,
                            tw_data);
        if constexpr (!reverse)
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, detail_::inplace_src);
    }

    template<uZ Width, uZ NodeSize, uZ DstPck, uZ SrcPck>
    void single_load_tform_inplace(T* dst, meta::ce_of<bool> auto reverse) const {
        using impl_t             = detail_::sequential_subtransform<NodeSize, T, Width>;
        constexpr auto dst_pck   = cxpack<DstPck, T>{};
        constexpr auto src_pck   = cxpack<SrcPck, T>{};
        constexpr auto conj_tw   = std::bool_constant<reverse>{};
        constexpr auto PermWidth = detail_::powi(2, detail_::log2i(Width * NodeSize) / 2);

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};
        if constexpr (reverse)
            permuter.sequential_permute(src_pck, src_pck, dst_data, detail_::inplace_src);
        impl_t::single_load(dst_pck, src_pck, lowk, half_tw, conj_tw, reverse, dst, dst, tw_data);
        if constexpr (!reverse)
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, detail_::inplace_src);
    }

    template<uZ Width, uZ DstPck, uZ SrcPck, uZ Align>
    void tform_external(T* dst, const T* src, meta::ce_of<bool> auto reverse) const {
        using impl_t             = detail_::transform<Opts.node_size, T, Width, Opts.coherent_size, 0>;
        constexpr auto dst_pck   = cxpack<DstPck, T>{};
        constexpr auto src_pck   = cxpack<SrcPck, T>{};
        constexpr auto align     = detail_::align_param<Align, true>{};
        constexpr auto conj_tw   = std::bool_constant<reverse>{};
        constexpr auto PermWidth = std::min(Width, max_perm_width);

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = 1};
        auto src_data = detail_::sequential_data_info<const T>{.data_ptr = src, .stride = 1};

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        if constexpr (!reverse) {
            impl_t::perform_tf(dst_pck,
                               src_pck,
                               half_tw,
                               lowk,
                               align,
                               dst_data,
                               detail_::inplace_src,
                               fft_size_,
                               tw_data,
                               permuter);
        } else {
            impl_t::perform_tf_rev(dst_pck,
                                   src_pck,
                                   half_tw,
                                   lowk,
                                   align,
                                   dst_data,
                                   detail_::inplace_src,
                                   fft_size_,
                                   tw_data,
                                   permuter);
        }
    }

    template<uZ Width, uZ PermWidth, uZ DstPck, uZ SrcPck, uZ Align>
    void coherent_tform_external(T* dst, const T* src, meta::ce_of<bool> auto reverse) const {
        using impl_t           = detail_::sequential_subtransform<Opts.node_size, T, Width>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto align   = detail_::align_param<Align, true>{};
        constexpr auto conj_tw = std::bool_constant<reverse>{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};
        auto src_data = detail_::sequential_data_info<const T>{.data_ptr = src, .stride = Width};

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        if constexpr (reverse) {
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, src_data);
            impl_t::perform_seq(dst_pck,
                                src_pck,
                                align,
                                lowk,
                                half_tw,
                                reverse,
                                conj_tw,
                                fft_size_,
                                dst_data,
                                detail_::inplace_src,
                                tw_data);
        } else {
            impl_t::perform_seq(dst_pck,
                                src_pck,
                                align,
                                lowk,
                                half_tw,
                                reverse,
                                conj_tw,
                                fft_size_,
                                dst_data,
                                src_data,
                                tw_data);
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, detail_::inplace_src);
        }
    }

    template<uZ Width, uZ NodeSize, uZ DstPck, uZ SrcPck>
    void single_load_tform_external(T* dst, const T* src, meta::ce_of<bool> auto reverse) const {
        using impl_t             = detail_::sequential_subtransform<NodeSize, T, Width>;
        constexpr auto dst_pck   = cxpack<DstPck, T>{};
        constexpr auto src_pck   = cxpack<SrcPck, T>{};
        constexpr auto conj_tw   = std::bool_constant<reverse>{};
        constexpr auto PermWidth = detail_::powi(2, detail_::log2i(Width * NodeSize) / 2);

        auto permuter = [&] {
            if constexpr (bit_reversed) {
                return permuter_;
            } else {
                constexpr auto width_idx = detail_::log2i(PermWidth);

                auto perm    = get<width_idx>(permuter_);
                perm.idx_ptr = idxs_.data();
                return perm;
            }
        }();
        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = reverse ? &*tw_.end() : tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};
        auto src_data = detail_::sequential_data_info<const T>{.data_ptr = src, .stride = Width};
        if constexpr (reverse)
            permuter.sequential_permute(src_pck, src_pck, dst_data, src_data);
        impl_t::single_load(dst_pck,
                            src_pck,
                            lowk,
                            half_tw,
                            conj_tw,
                            reverse,
                            dst,
                            reverse ? dst : src,
                            tw_data);
        if constexpr (!reverse)
            permuter.sequential_permute(dst_pck, dst_pck, dst_data, detail_::inplace_src);
    }
};
}    // namespace pcx
