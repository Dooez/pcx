#include "pcx/par_fft.hpp"
namespace pcx {
namespace {
constexpr auto forward     = std::false_type{};
constexpr auto reverse     = std::true_type{};
constexpr auto normal_opts = fft_options{.pt = fft_permutation::normal};
constexpr auto bitrev_opts = fft_options{.pt = fft_permutation::bit_reversed};
constexpr auto shiftd_opts = fft_options{.pt = fft_permutation::shifted};
}    // namespace


template<floating_point T, fft_options Opts>
par_fft_plan<T, Opts>::par_fft_plan(uZ fft_size)
: fft_size_(fft_size)
, permuter_(permuter_t::insert_indexes(idxs_, fft_size, coherent_size)) {
    // constexpr auto lowk = val_ce<true>{};
    if (fft_size > coherent_size / lane_size) {
        using impl_t = detail_::transform<Opts.node_size, T, width, coherent_size, lane_size>;
        impl_t::insert_tw_tf(tw_, fft_size, lowk, half_tw, not_sequential);
        auto align_node = impl_t::get_align_node_tf_par(fft_size);
        auto test_align = [&]<uZ I>(uZ_ce<I>) {
            constexpr auto l_align_node = detail_::powi(2, I);
            if (l_align_node != align_node)
                return false;
            inplace_ptr_    = &par_fft_plan::inplace<1, 1, l_align_node, false>;
            inplace_r_ptr_  = &par_fft_plan::inplace<1, 1, l_align_node, true>;
            external_ptr_   = &par_fft_plan::external<1, 1, l_align_node, false>;
            external_r_ptr_ = &par_fft_plan::external<1, 1, l_align_node, true>;
            return true;
        };
        [&]<uZ... Is>(uZ_seq<Is...>) {
            (void)(test_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
        return;
    }
    auto tw_data = detail_::tw_data_t<T, true>{};
    auto k_cnt   = fft_size / 2;
    if (fft_size > Opts.node_size) {
        using impl_t    = detail_::subtransform<Opts.node_size, T, width>;
        auto align_node = impl_t::get_align_node_subtf(fft_size);
        auto test_align = [&]<uZ I>(uZ_ce<I>) {
            constexpr auto l_align_node = detail_::powi(2, I);
            if (l_align_node != align_node)
                return false;
            constexpr auto align = detail_::align_param<l_align_node, true>{};
            impl_t::insert_tw_subtf(tw_, align, lowk, k_cnt, tw_data);
            inplace_ptr_    = &par_fft_plan::inplace_coh<l_align_node, 1, 1, false>;
            inplace_r_ptr_  = &par_fft_plan::inplace_coh<l_align_node, 1, 1, true>;
            external_ptr_   = &par_fft_plan::external_coh<l_align_node, 1, 1, false>;
            external_r_ptr_ = &par_fft_plan::external_coh<l_align_node, 1, 1, true>;
            return true;
        };
        [&]<uZ... Is>(uZ_seq<Is...>) {
            (void)(test_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
        return;
    }
    auto test_single_node = [&]<uZ I>(uZ_ce<I>) {
        constexpr auto l_node_size = detail_::powi(2, I + 1);
        if (fft_size != l_node_size)
            return false;
        using impl_t = detail_::fft_iteration_t<T, width>;
        impl_t::insert_tw_iteration(uZ_ce<l_node_size>{}, tw_, tw_data, k_cnt, lowk);
        inplace_ptr_    = &par_fft_plan::inplace_single_node<l_node_size, 1, 1, false>;
        inplace_r_ptr_  = &par_fft_plan::inplace_single_node<l_node_size, 1, 1, true>;
        external_ptr_   = &par_fft_plan::external_single_node<l_node_size, 1, 1, false>;
        external_r_ptr_ = &par_fft_plan::external_single_node<l_node_size, 1, 1, true>;
        return true;
    };
    [&]<uZ... Is>(uZ_seq<Is...>) {
        (void)(test_single_node(uZ_ce<Is>{}) || ...);
    }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
}


template<floating_point T, fft_options Opts>
template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
PCX_AINLINE void par_fft_plan<T, Opts>::impl(detail_::data_info_for<T> auto       dst_data,
                                             detail_::data_info_for<const T> auto src_data,
                                             uZ                                   data_size) const {
    using impl_t           = detail_::transform<Opts.node_size, T, width, coherent_size, lane_size>;
    constexpr auto dst_pck = cxpack<DstPck, T>{};
    constexpr auto src_pck = cxpack<SrcPck, T>{};
    constexpr auto align   = detail_::align_param<Align, true>{};

    auto tw       = detail_::tw_data_t<T, false>{Reverse ? &*tw_.end() : tw_.data()};
    auto permuter = permuter_;
    if constexpr (!bit_reversed)
        permuter.idx_ptr = idxs_.data();
    if constexpr (!Reverse) {
        impl_t::perform_tf(dst_pck,
                           src_pck,
                           half_tw,
                           lowk,
                           align,
                           dst_data,
                           src_data,
                           fft_size_,
                           tw,
                           permuter,
                           data_size);
    } else {
        impl_t::perform_tf_rev(dst_pck,
                               src_pck,
                               half_tw,
                               lowk,
                               align,
                               dst_data,
                               src_data,
                               fft_size_,
                               tw,
                               permuter,
                               data_size);
    }
}
template<floating_point T, fft_options Opts>
template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
void par_fft_plan<T, Opts>::inplace(T* dst, uZ stride, uZ data_size) const {
    auto dst_data = detail_::data_info<T, true>{.data_ptr = dst, .stride = stride, .k_stride = stride};
    auto src_data = detail_::inplace_src;
    impl<DstPck, SrcPck, Align, Reverse>(dst_data, src_data, data_size);
}
template<floating_point T, fft_options Opts>
template<uZ DstPck, uZ SrcPck, uZ Align, bool Reverse>
void par_fft_plan<T, Opts>::external(T* dst, uZ dst_stride, const T* src, uZ src_stride, uZ data_size) const {
    auto dst_data =
        detail_::data_info<T, true>{.data_ptr = dst, .stride = dst_stride, .k_stride = dst_stride};
    auto src_data =
        detail_::data_info<const T, true>{.data_ptr = src, .stride = src_stride, .k_stride = src_stride};
    impl<DstPck, SrcPck, Align, Reverse>(dst_data, src_data, data_size);
}

template<floating_point T, fft_options Opts>
template<bool SingleNode, uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
PCX_AINLINE void par_fft_plan<T, Opts>::coh_impl(detail_::data_info_for<T> auto       dst_data,
                                                 detail_::data_info_for<const T> auto src_data,
                                                 uZ                                   data_size) const {
    constexpr auto dst_pck = cxpack<DstPck, T>{};
    constexpr auto src_pck = cxpack<SrcPck, T>{};
    constexpr auto align   = detail_::align_param<Align, true>{};
    constexpr auto reverse = std::bool_constant<Reverse>{};

    auto tw       = detail_::tw_data_t<T, false>{Reverse ? &*tw_.end() : tw_.data()};
    auto permuter = permuter_;
    if constexpr (!bit_reversed)
        permuter.idx_ptr = idxs_.data();

    constexpr auto bucket_size     = coherent_size;
    const auto     batch_align_seq = [=] {
        constexpr auto min_align = std::min(DstPck, SrcPck);
        constexpr auto pbegin    = detail_::log2i(min_align);
        constexpr auto pend      = detail_::log2i(lane_size);
        return []<uZ... Ps>(uZ_seq<Ps...>) {
            return uZ_seq<detail_::powi(2, pend - 1 - Ps)...>{};
        }(make_uZ_seq<pend - pbegin>{});
    }();
    constexpr auto bucket_tfsize = uZ_ce<bucket_size / lane_size>{};
    constexpr auto batch_tfsize  = uZ_ce<1>{};
    constexpr auto conj_tw       = reverse;

    auto tform = [&] {
        auto fft_size = fft_size_;
        if constexpr (!SingleNode) {
            return [=] PCX_LAINLINE(auto width, auto batch_size, auto dst, auto src) {
                using subtf_t      = detail_::subtransform<Opts.node_size, T, width>;
                auto batch_cnt     = fft_size;
                auto final_k_count = fft_size / 2;
                auto l_tw          = tw;
                subtf_t::perform_subtf(dst_pck,
                                       src_pck,
                                       align,
                                       lowk,
                                       reverse,
                                       conj_tw,
                                       batch_cnt,
                                       batch_size,
                                       dst,
                                       src,
                                       final_k_count,
                                       l_tw);
                auto{permuter}
                    .small_permute(width, batch_size, reverse, dst_pck, dst_pck, dst, detail_::inplace_src);
            };
        } else {
            return [=] PCX_LAINLINE(auto width, auto batch_size, auto dst, auto src) {
                using subtf_t  = detail_::fft_iteration_t<T, width>;
                auto batch_cnt = fft_size;
                auto k_count   = 1UZ;
                auto l_tw      = tw;
                subtf_t::fft_iteration(uZ_ce<Align>{},
                                       dst_pck,
                                       src_pck,
                                       lowk,
                                       reverse,
                                       conj_tw,
                                       batch_cnt,
                                       batch_size,
                                       dst,
                                       src,
                                       k_count,
                                       l_tw);
                auto{permuter}
                    .small_permute(width, batch_size, reverse, dst_pck, dst_pck, dst, detail_::inplace_src);
            };
        }
    }();

    auto batch_size = bucket_tfsize / fft_size_ * lane_size;
    while (true) {
        if (data_size < batch_size) {
            if (batch_size <= lane_size)
                break;
            batch_size /= 2;
            continue;
        }
        tform(width, batch_size, dst_data, src_data);
        data_size -= batch_size;
        dst_data = dst_data.offset_contents(batch_size);
        src_data = src_data.offset_contents(batch_size);
    }
    [&]<uZ... Batch> PCX_LAINLINE(uZ_seq<Batch...>) {
        auto small_tform = [&](auto small_batch) {
            if (data_size >= small_batch) {
                constexpr auto lwidth = uZ_ce<std::min(width.value, small_batch.value)>{};
                tform(lwidth, small_batch, dst_data, src_data);
                data_size -= small_batch;
                dst_data = dst_data.offset_contents(small_batch);
                src_data = src_data.offset_contents(small_batch);
            }
            return data_size != 0;
        };
        (void)(small_tform(uZ_ce<Batch>{}) && ...);
    }(batch_align_seq);
}
template<floating_point T, fft_options Opts>
template<uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
void par_fft_plan<T, Opts>::inplace_coh(T* dst, uZ stride, uZ data_size) const {
    auto dst_data = detail_::data_info<T, true>{.data_ptr = dst, .stride = stride, .k_stride = stride};
    auto src_data = detail_::inplace_src;
    coh_impl<false, Align, DstPck, SrcPck, Reverse>(dst_data, src_data, data_size);
};
template<floating_point T, fft_options Opts>
template<uZ Align, uZ DstPck, uZ SrcPck, bool Reverse>
void par_fft_plan<T, Opts>::external_coh(T* dst, uZ dst_stride, const T* src, uZ src_stride, uZ data_size)
    const {
    auto dst_data =
        detail_::data_info<T, true>{.data_ptr = dst, .stride = dst_stride, .k_stride = dst_stride};
    auto src_data =
        detail_::data_info<const T, true>{.data_ptr = src, .stride = src_stride, .k_stride = src_stride};
    coh_impl<false, Align, DstPck, SrcPck, Reverse>(dst_data, src_data, data_size);
};
template<floating_point T, fft_options Opts>
template<uZ NodeSize, uZ DstPck, uZ SrcPck, bool Reverse>
void par_fft_plan<T, Opts>::inplace_single_node(T* dst, uZ stride, uZ data_size) const {
    auto dst_data = detail_::data_info<T, true>{.data_ptr = dst, .stride = stride, .k_stride = stride};
    auto src_data = detail_::inplace_src;
    coh_impl<true, NodeSize, DstPck, SrcPck, Reverse>(dst_data, src_data, data_size);
};
template<floating_point T, fft_options Opts>
template<uZ NodeSize, uZ DstPck, uZ SrcPck, bool Reverse>
void par_fft_plan<T, Opts>::external_single_node(T*       dst,
                                                 uZ       dst_stride,
                                                 const T* src,
                                                 uZ       src_stride,
                                                 uZ       data_size) const {
    auto dst_data =
        detail_::data_info<T, true>{.data_ptr = dst, .stride = dst_stride, .k_stride = dst_stride};
    auto src_data =
        detail_::data_info<const T, true>{.data_ptr = src, .stride = src_stride, .k_stride = src_stride};
    coh_impl<true, NodeSize, DstPck, SrcPck, Reverse>(dst_data, src_data, data_size);
};

template class par_fft_plan<f32, bitrev_opts>;
template class par_fft_plan<f32, normal_opts>;
template class par_fft_plan<f32, shiftd_opts>;

template class par_fft_plan<f64, bitrev_opts>;
template class par_fft_plan<f64, normal_opts>;
template class par_fft_plan<f64, shiftd_opts>;

}    // namespace pcx
