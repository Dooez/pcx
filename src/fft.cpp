#include "pcx/fft.hpp"
namespace pcx {
namespace {
constexpr auto forward     = std::false_type{};
constexpr auto reverse     = std::true_type{};
constexpr auto normal_opts = fft_options{.pt = fft_permutation::normal};
constexpr auto bitrev_opts = fft_options{.pt = fft_permutation::bit_reversed};
constexpr auto shiftd_opts = fft_options{.pt = fft_permutation::shifted};
}    // namespace

template<floating_point T, fft_options Opts>
fft_plan<T, Opts>::fft_plan(uZ fft_size)
: fft_size_(fft_size) {
    if (fft_size == 1) {
        ileave_inplace_ptr_ = &fft_plan::identity_tform;
        return;
    }
    auto tw_data = detail_::tw_data_t<T, true>{};
    if (fft_size > coherent_size) {
        using impl_t = detail_::transform<Opts.node_size, T, width, coherent_size, 0>;
        impl_t::insert_tw_tf(tw_, fft_size, lowk, half_tw, sequential);
        permuter_        = permuter_t<width>::insert_indexes(idxs_, fft_size);
        auto align_node  = impl_t::get_align_node_tf_seq(fft_size);
        auto check_align = [&](auto p_align) {
            constexpr auto l_align_node = detail_::powi(2, p_align);
            if (l_align_node != align_node)
                return false;
            ileave_inplace_ptr_    = &fft_plan::tform_inplace_ileave<width, l_align_node>;
            ileave_inplace_r_ptr_  = &fft_plan::rtform_inplace_ileave<width, l_align_node>;
            ileave_external_ptr_   = &fft_plan::tform_external_ileave<width, l_align_node>;
            ileave_external_r_ptr_ = &fft_plan::rtform_external_ileave<width, l_align_node>;
            return true;
        };
        [&]<uZ... Is>(uZ_seq<Is...>) {
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
        return;
    }
    constexpr auto max_single_load = Opts.node_size * width;
    if (fft_size > max_single_load) {
        using impl_t = detail_::sequential_subtransform<Opts.node_size, T, width>;

        auto align_node  = impl_t::get_align_node_seq(fft_size);
        auto check_align = [&](auto p_align) {
            constexpr auto align_node_size = detail_::powi(2, p_align);
            if (align_node_size != align_node)
                return false;
            constexpr auto align = align_param<align_node_size>{};
            impl_t::insert_tw_seq(tw_, align, lowk, fft_size, tw_data, half_tw);

            constexpr auto min_perm_width = detail_::powi(2, detail_::log2i(max_single_load) / 2);

            auto check_perm = [&](auto p_width) {
                constexpr auto perm_width = width / detail_::powi(2, p_width);
                if (fft_size < perm_width * perm_width)
                    return false;
                permuter_ = permuter_t<perm_width>::insert_indexes(idxs_, fft_size);
                ileave_inplace_ptr_ =
                    &fft_plan::coherent_tform_inplace_ileave<width, perm_width, align_node_size>;
                ileave_inplace_r_ptr_ =
                    &fft_plan::coherent_rtform_inplace_ileave<width, perm_width, align_node_size>;
                ileave_external_ptr_ =
                    &fft_plan::coherent_tform_external_ileave<width, perm_width, align_node_size>;
                ileave_external_r_ptr_ =
                    &fft_plan::coherent_rtform_external_ileave<width, perm_width, align_node_size>;
                return true;
            };
            [&]<uZ... Is>(uZ_seq<Is...>) {
                (void)(check_perm(uZ_ce<Is>{}) || ...);
            }(make_uZ_seq<detail_::log2i(width / min_perm_width) + 1>{});
            return true;
        };
        [&]<uZ... Is>(uZ_seq<Is...>) {
            (void)(check_align(uZ_ce<Is>{}) || ...);
        }(make_uZ_seq<detail_::log2i(Opts.node_size)>{});
        return;
    }
    if (fft_size == max_single_load) {
        using impl_t = detail_::sequential_subtransform<Opts.node_size, T, width>;
        impl_t::insert_tw_single_load(tw_, tw_data, lowk, half_tw);

        constexpr auto perm_width = detail_::powi(2, detail_::log2i(max_single_load) / 2);
        permuter_                 = permuter_t<perm_width>::insert_indexes(idxs_, fft_size);
        ileave_inplace_ptr_       = &fft_plan::single_load_tform_inplace_ileave<width, Opts.node_size>;
        ileave_inplace_r_ptr_     = &fft_plan::single_load_rtform_inplace_ileave<width, Opts.node_size>;
        ileave_external_ptr_      = &fft_plan::single_load_tform_external_ileave<width, Opts.node_size>;
        ileave_external_r_ptr_    = &fft_plan::single_load_rtform_external_ileave<width, Opts.node_size>;
        return;
    };
    auto narrow = [&]<uZ... Is>(uZ_seq<Is...>) {
        auto check_narrow_tf = [&](auto p) {
            constexpr auto l_node_size = Opts.node_size;
            constexpr auto l_width     = width / detail_::powi(2, p + 1);
            constexpr auto single_load = uZ_ce<l_node_size * l_width>{};

            if (fft_size == single_load) {
                using impl_t = detail_::sequential_subtransform<l_node_size, T, l_width>;
                impl_t::insert_tw_single_load(tw_, tw_data, lowk, half_tw);
                constexpr auto perm_width = detail_::powi(2, detail_::log2i(single_load) / 2);
                using perm_t              = permuter_t<perm_width>;
                permuter_                 = perm_t::insert_indexes(idxs_, fft_size);
                ileave_inplace_ptr_       = &fft_plan::single_load_tform_inplace_ileave<l_width, l_node_size>;
                ileave_inplace_r_ptr_  = &fft_plan::single_load_rtform_inplace_ileave<l_width, l_node_size>;
                ileave_external_ptr_   = &fft_plan::single_load_tform_external_ileave<l_width, l_node_size>;
                ileave_external_r_ptr_ = &fft_plan::single_load_rtform_external_ileave<l_width, l_node_size>;
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

            if (fft_size == l_node_size) {
                using impl_t = detail_::sequential_subtransform<l_node_size, T, l_width>;
                impl_t::insert_tw_single_load(tw_, tw_data, lowk, half_tw);
                constexpr auto perm_width = detail_::powi(2, detail_::log2i(l_node_size) / 2);
                using perm_t              = permuter_t<perm_width>;
                permuter_                 = perm_t::insert_indexes(idxs_, fft_size);
                ileave_inplace_ptr_       = &fft_plan::single_load_tform_inplace_ileave<l_width, l_node_size>;
                ileave_inplace_r_ptr_  = &fft_plan::single_load_rtform_inplace_ileave<l_width, l_node_size>;
                ileave_external_ptr_   = &fft_plan::single_load_tform_external_ileave<l_width, l_node_size>;
                ileave_external_r_ptr_ = &fft_plan::single_load_rtform_external_ileave<l_width, l_node_size>;
                return true;
            }
            return false;
        };
        (void)(check_small(uZ_ce<Is>{}) || ...);
    }(make_uZ_seq<detail_::log2i(Opts.node_size) - 1>{});
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ Align>
void fft_plan<T, Opts>::tform_inplace_ileave(T* dst) const {
    tform_inplace<Width, 1, 1, Align>(dst, forward);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_tform_inplace_ileave(T* dst) const {
    coherent_tform_inplace<Width, PermWidth, 1, 1, Align>(dst, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_tform_inplace_ileave(T* dst) const {
    single_load_tform_inplace<Width, NodeSize, 1, 1>(dst, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ Align>
void fft_plan<T, Opts>::rtform_inplace_ileave(T* dst) const {
    tform_inplace<Width, 1, 1, Align>(dst, reverse);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_rtform_inplace_ileave(T* dst) const {
    coherent_tform_inplace<Width, PermWidth, 1, 1, Align>(dst, reverse);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_rtform_inplace_ileave(T* dst) const {
    single_load_tform_inplace<Width, NodeSize, 1, 1>(dst, reverse);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ Align>
void fft_plan<T, Opts>::tform_external_ileave(T* dst, const T* src) const {
    tform_external<Width, 1, 1, Align>(dst, src, forward);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_tform_external_ileave(T* dst, const T* src) const {
    coherent_tform_external<Width, PermWidth, 1, 1, Align>(dst, src, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_tform_external_ileave(T* dst, const T* src) const {
    single_load_tform_external<Width, NodeSize, 1, 1>(dst, src, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ Align>
void fft_plan<T, Opts>::rtform_external_ileave(T* dst, const T* src) const {
    tform_external<Width, 1, 1, Align>(dst, src, reverse);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_rtform_external_ileave(T* dst, const T* src) const {
    coherent_tform_external<Width, PermWidth, 1, 1, Align>(dst, src, reverse);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_rtform_external_ileave(T* dst, const T* src) const {
    single_load_tform_external<Width, NodeSize, 1, 1>(dst, src, reverse);
}

template class fft_plan<f32, bitrev_opts>;
template class fft_plan<f32, normal_opts>;
template class fft_plan<f32, shiftd_opts>;
template class fft_plan<f64, bitrev_opts>;
template class fft_plan<f64, normal_opts>;
template class fft_plan<f64, shiftd_opts>;
}    // namespace pcx
