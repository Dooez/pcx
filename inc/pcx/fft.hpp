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
    uZ node_size     = 8;
    uZ simd_width    = 0;
};

/**
 * @brief FFT plan for performing fft over sequential data.
 */
template<floating_point T, fft_options Opts = {}>
class fft_plan {
    static constexpr auto lowk         = std::true_type{};
    static constexpr auto half_tw      = std::true_type{};
    static constexpr auto bit_reversed = std::bool_constant<Opts.pt == fft_permutation::bit_reversed>{};
    static constexpr auto sequential   = std::true_type{};
    static constexpr auto coherent_size =
        uZ_ce<Opts.coherent_size != 0 ? Opts.coherent_size : 8192 / sizeof(T)>{};
    static constexpr auto width = uZ_ce<Opts.simd_width != 0 ? Opts.simd_width : simd::max_width<T>>{};

    template<uZ AlignNode>
    using align_param = detail_::align_param<AlignNode, true>;

public:
    fft_plan(uZ fft_size)
    : fft_size_(fft_size) {
        if (fft_size == 1) {
            ileave_impl_ptr_ = &fft_plan::identity_tform;
            return;
        }
        if (fft_size > coherent_size) {
            using impl_t = detail_::transform<Opts.node_size, T, width, coherent_size, 0>;
            impl_t::insert_tw(tw_, fft_size, lowk, half_tw, sequential);
            // ileave_impl_ptr_ = &fft_plan::tform_inplace<width, 1, 1>;
            ileave_impl_ptr_ = &fft_plan::tform_inplace_ileave<width>;
            return;
        }
        constexpr auto max_single_load = Opts.node_size * width;
        if (fft_size > max_single_load) {
            using impl_t = detail_::sequential_subtransform<Opts.node_size, T, width>;

            auto align_node  = impl_t::get_align_node(fft_size);
            auto check_align = [&](auto p) {
                constexpr auto align_node_size = detail_::powi(2, p);
                if (align_node_size != align_node)
                    return false;
                constexpr auto align   = align_param<align_node_size>{};
                auto           tw_data = detail_::tw_data_t<T, true>{};
                impl_t::insert_tw(tw_, align, lowk, fft_size, tw_data, half_tw);
                // ileave_impl_ptr_ = &fft_plan::coherent_tform_inplace<width, 1, 1, align_node_size>;
                ileave_impl_ptr_ = &fft_plan::coherent_tform_inplace_ileave<width, align_node_size>;
                return true;
            };
            [&]<uZ... Is>(uZ_seq<Is...>) {
                (void)(check_align(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
            return;
        }
        if (fft_size == max_single_load) {
            using impl_t = detail_::sequential_subtransform<Opts.node_size, T, width>;
            auto tw_data = detail_::tw_data_t<T, true>{};
            impl_t::insert_single_load_tw(tw_, tw_data, lowk, half_tw);
            // ileave_impl_ptr_ = &fft_plan::coherent_tform_inplace<width, 1, 1, align_node_size>;
            ileave_impl_ptr_ = &fft_plan::single_load_tform_inplace_ileave<width, Opts.node_size>;
            return;
        };
        auto narrow = [&]<uZ... Is>(uZ_seq<Is...>) {
            auto check_narrow_tf = [&](auto p) {
                constexpr auto l_node_size = Opts.node_size;
                constexpr auto l_width     = width / detail_::powi(2, p + 1);
                constexpr auto single_load = uZ_ce<l_node_size * l_width>{};

                uZ lns       = l_node_size;
                uZ w         = l_width;
                using impl_t = detail_::sequential_subtransform<l_node_size, T, l_width>;
                if (fft_size == single_load) {
                    auto tw_data = detail_::tw_data_t<T, true>{};
                    impl_t::insert_single_load_tw(tw_, tw_data, lowk, half_tw);
                    // ileave_impl_ptr_ = &fft_plan::single_load_tform_inplace<width, l_node_size, 1, 1>;
                    ileave_impl_ptr_ = &fft_plan::single_load_tform_inplace_ileave<l_width, l_node_size>;
                    return true;
                }
                return false;
            };
            return (check_narrow_tf(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(width)>{});
        if (narrow)
            return;
        [&]<uZ... Is>(uZ_seq<Is...>) {
            auto check_small = [&](auto p) {
                constexpr auto l_node_size = Opts.node_size / detail_::powi(2, p + 1);
                constexpr auto l_width     = 1;

                uZ lns = l_node_size;
                uZ w   = l_width;

                if (fft_size == l_node_size) {
                    using impl_t = detail_::sequential_subtransform<l_node_size, T, l_width>;
                    auto tw_data = detail_::tw_data_t<T, true>{};
                    impl_t::insert_single_load_tw(tw_, tw_data, lowk, half_tw);
                    // ileave_impl_ptr_ = &fft_plan::single_load_tform_inplace<width, l_node_size, 1, 1>;
                    ileave_impl_ptr_ = &fft_plan::single_load_tform_inplace_ileave<l_width, l_node_size>;
                    return true;
                }
                return false;
            };
            (void)(check_small(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(Opts.node_size) - 1>{});
    };

    void fft(std::vector<std::complex<T>>& data) {
        if (data.size() != fft_size_)
            throw std::runtime_error("Data size not equal to fft size");

        auto raw_ptr = reinterpret_cast<T*>(data.data());
        (this->*ileave_impl_ptr_)(raw_ptr);
    }

private:
    using member_impl_t  = auto (fft_plan::*)(T*) -> void;
    using permute_idxs_t = std::conditional_t<bit_reversed, decltype([] {}), std::vector<u32>>;

    uZ                                   fft_size_{};
    std::vector<T>                       tw_{};
    [[no_unique_address]] permute_idxs_t idxs_{};
    member_impl_t                        ileave_impl_ptr_;

    void identity_tform(T* dst) {};
    template<uZ Width>
    void tform_inplace_ileave(T* dst);
    template<uZ Width, uZ Align>
    void coherent_tform_inplace_ileave(T* dst);
    template<uZ Width, uZ NodeSize>
    void single_load_tform_inplace_ileave(T* dst);

    template<uZ Width, uZ DstPck, uZ SrcPck>
    void tform_inplace(T* dst) {
        using impl_t           = detail_::transform<Opts.node_size, T, Width, Opts.coherent_size, 0>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto reverse = std::false_type{};
        constexpr auto conj_tw = std::false_type{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = 1};
        impl_t::perform(dst_pck,
                        src_pck,
                        half_tw,
                        lowk,
                        dst_data,
                        detail_::inplace_src,
                        fft_size_,
                        tw_data,
                        detail_::identity_permuter);
    }

    template<uZ Width, uZ DstPck, uZ SrcPck, uZ Align>
    void coherent_tform_inplace(T* dst) {
        using impl_t           = detail_::sequential_subtransform<Opts.node_size, T, Width>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto align   = detail_::align_param<Align, true>{};
        constexpr auto reverse = std::false_type{};
        constexpr auto conj_tw = std::false_type{};

        auto tw_data  = detail_::tw_data_t<T, false>{.tw_ptr = tw_.data()};
        auto dst_data = detail_::sequential_data_info<T>{.data_ptr = dst, .stride = Width};
        impl_t::perform_impl(dst_pck,
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
    }

    template<uZ Width, uZ NodeSize, uZ DstPck, uZ SrcPck>
    void single_load_tform_inplace(T* dst) {
        using impl_t           = detail_::sequential_subtransform<NodeSize, T, Width>;
        constexpr auto dst_pck = cxpack<DstPck, T>{};
        constexpr auto src_pck = cxpack<SrcPck, T>{};
        constexpr auto reverse = std::false_type{};
        constexpr auto conj_tw = std::false_type{};

        auto tw_data = detail_::tw_data_t<T, false>{.tw_ptr = tw_.data()};
        impl_t::single_load(dst_pck, src_pck, lowk, half_tw, conj_tw, reverse, dst, dst, tw_data);
    }
};


}    // namespace pcx
