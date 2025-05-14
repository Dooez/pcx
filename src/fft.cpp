#include "pcx/fft.hpp"
namespace pcx {
namespace {
constexpr auto forward     = std::false_type{};
constexpr auto reverse     = std::true_type{};
constexpr auto normal_opts = fft_options{.pt = fft_permutation::normal};
constexpr auto shiftd_opts = fft_options{.pt = fft_permutation::shifted};
}    // namespace

template<floating_point T, fft_options Opts>
template<uZ Width>
void fft_plan<T, Opts>::tform_inplace_ileave(T* dst) {
    tform_inplace<Width, 1, 1>(dst, forward);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_tform_inplace_ileave(T* dst) {
    coherent_tform_inplace<Width, PermWidth, 1, 1, Align>(dst, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_tform_inplace_ileave(T* dst) {
    single_load_tform_inplace<Width, NodeSize, 1, 1>(dst, forward);
}

template<floating_point T, fft_options Opts>
template<uZ Width>
void fft_plan<T, Opts>::rtform_inplace_ileave(T* dst) {
    tform_inplace<Width, 1, 1>(dst, reverse);
}
template<floating_point T, fft_options Opts>
template<uZ Width, uZ PermWidth, uZ Align>
void fft_plan<T, Opts>::coherent_rtform_inplace_ileave(T* dst) {
    coherent_tform_inplace<Width, PermWidth, 1, 1, Align>(dst, reverse);
}

template<floating_point T, fft_options Opts>
template<uZ Width, uZ NodeSize>
void fft_plan<T, Opts>::single_load_rtform_inplace_ileave(T* dst) {
    single_load_tform_inplace<Width, NodeSize, 1, 1>(dst, reverse);
}

template class fft_plan<f32>;
template class fft_plan<f32, normal_opts>;
template class fft_plan<f32, shiftd_opts>;
template class fft_plan<f64>;
template class fft_plan<f64, normal_opts>;
template class fft_plan<f64, shiftd_opts>;
}    // namespace pcx
