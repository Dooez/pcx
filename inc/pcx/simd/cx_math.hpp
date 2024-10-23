#ifndef PCX_SIMD_CX_MATH_HPP
#define PCX_SIMD_CX_MATH_HPP
#include "pcx/types.hpp"

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline
namespace pcx::simd {

template<cx_vec_c Lhs, cx_vec_c Rhs>
    requires compatible<Lhs, Rhs>
PCX_AINLINE auto add(Lhs lhs, Rhs rhs) {
    using vec = Lhs::vec_t;
    vec real;
    vec imag;

    if constexpr (Lhs::neg_real() == Rhs::neg_real()) {
        real = lhs.real() + rhs.real();
    } else if constexpr (Lhs::neg_real()) {
        real = rhs.real() - lhs.real();
    } else {
        real = lhs.real() - rhs.real();
    }

    if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
        imag = lhs.imag() + rhs.imag();
    } else if constexpr (Lhs::neg_imag()) {
        imag = rhs.imag() - lhs.imag();
    } else {
        imag = lhs.imag() - rhs.imag();
    }

    constexpr bool new_nreal = Lhs::neg_real() && Rhs::neg_real();
    constexpr bool new_nimag = Lhs::neg_real() && Rhs::neg_imag();

    using new_cx_vec = cx_vec<typename vec::value_type, new_nreal, new_nimag, Lhs::size(), Lhs::pack_size()>;
    return new_cx_vec{.m_real = real, .m_imag = imag};
}

template<cx_vec_c Lt, cx_vec_c Rt>
    requires compatible<Lt, Rt>
PCX_AINLINE auto sub(Lt lhs, Rt rhs);

template<tight_cx_vec Lt, tight_cx_vec Rt>
    requires compatible<Lt, Rt>
PCX_AINLINE auto mul(Lt lhs, Rt rhs);

template<cx_vec_c Lt, cx_vec_c Rt>
    requires(compatible<Lt, Rt> && Lt::size() == Lt::pack_size)
PCX_AINLINE auto div(Lt lhs, Rt rhs);

template<tight_cx_vec... Lhs, tight_cx_vec... Rhs>
    requires(compatible<Lhs, Rhs> && ...)
PCX_AINLINE auto mul_tuples(std::tuple<Lhs...> lhs, std::tuple<Rhs...> rhs);

template<tight_cx_vec... Lhs, tight_cx_vec... Rhs>
    requires(compatible<Lhs, Rhs> && ...)
PCX_AINLINE auto div_tuples(std::tuple<Lhs...> lhs, std::tuple<Rhs...> rhs);

}    // namespace pcx::simd

#undef PCX_AINLINE
#endif
