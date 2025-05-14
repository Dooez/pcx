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

template void fft_plan<f32>::tform_inplace_ileave<16>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32>::coherent_tform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32>::single_load_tform_inplace_ileave<1, 2>(f32*);

template void fft_plan<f32, normal_opts>::tform_inplace_ileave<16>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32, normal_opts>::coherent_tform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32, normal_opts>::single_load_tform_inplace_ileave<1, 2>(f32*);

template void fft_plan<f32, shiftd_opts>::tform_inplace_ileave<16>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_tform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_tform_inplace_ileave<1, 2>(f32*);


template void fft_plan<f32>::rtform_inplace_ileave<16>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32>::coherent_rtform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32>::single_load_rtform_inplace_ileave<1, 2>(f32*);

template void fft_plan<f32, normal_opts>::rtform_inplace_ileave<16>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32, normal_opts>::coherent_rtform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32, normal_opts>::single_load_rtform_inplace_ileave<1, 2>(f32*);

template void fft_plan<f32, shiftd_opts>::rtform_inplace_ileave<16>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 16, 1>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 16, 2>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 16, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 16, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 8, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 8, 2>(f32*);
template void fft_plan<f32, shiftd_opts>::coherent_rtform_inplace_ileave<16, 8, 1>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<16, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<8, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<4, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<2, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<1, 8>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<1, 4>(f32*);
template void fft_plan<f32, shiftd_opts>::single_load_rtform_inplace_ileave<1, 2>(f32*);
}    // namespace pcx
