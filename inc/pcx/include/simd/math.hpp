#ifndef PCX_SIMD_MATH_HPP
#define PCX_SIMD_MATH_HPP

#include "pcx/include/simd/traits.hpp"

namespace pcx::simd {

template<typename T, uZ Width>
PCX_AINLINE auto fmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fmadd(a.value, b.value, c.value);
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fnmadd(a.value, b.value, c.value);
}
template<typename T, uZ Width>
PCX_AINLINE auto fmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fmsub(a.value, b.value, c.value);
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fnmsub(a.value, b.value, c.value);
}

inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::add(lhs.value, rhs.value);
    }
    template<typename T, uZ Width, uZ PackSize>
    PCX_AINLINE auto operator()(cx_vec<T, false, false, Width, PackSize> lhs,
                                cx_vec<T, false, false, Width, PackSize> rhs) const {
        using traits = detail_::vec_traits<T, Width>;
        auto real    = traits::add(lhs.m_real.value, rhs.m_real.value);
        auto imag    = traits::add(lhs.m_imag.value, rhs.m_imag.value);
        return cx_vec<T, false, false, Width, PackSize>{.m_real = real, .m_imag = imag};
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;
        vec real;
        vec imag;

        if constexpr (Lhs::neg_real() == Rhs::neg_real()) {
            real = traits::add(lhs.real().value, rhs.real().value);
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real().value, lhs.real().value);
        } else {
            real = traits::sub(lhs.real().value, rhs.real().value);
        }

        if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
            imag = traits::add(lhs.imag().value, rhs.imag().value);
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag().value, lhs.imag().value);
        } else {
            imag = traits::sub(lhs.imag().value, rhs.imag().value);
        }

        constexpr bool neg_real = Lhs::neg_real() && Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
} add;

inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::sub(lhs.value, rhs.value);
    }
    template<typename T, uZ Width, uZ PackSize>
    PCX_AINLINE auto operator()(cx_vec<T, false, false, Width, PackSize> lhs,
                                cx_vec<T, false, false, Width, PackSize> rhs) const {
        using traits = detail_::vec_traits<T, Width>;
        auto real    = traits::sub(lhs.m_real.value, rhs.m_real.value);
        auto imag    = traits::sub(lhs.m_imag.value, rhs.m_imag.value);
        return cx_vec<T, false, false, Width, PackSize>{.m_real = real, .m_imag = imag};
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;
        vec real;
        vec imag;

        if constexpr (Lhs::neg_real() != Rhs::neg_real()) {
            real = traits::add(lhs.real().value, rhs.real().value);
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real().value, lhs.real().value);
        } else {
            real = traits::sub(lhs.real().value, rhs.real().value);
        }

        if constexpr (Lhs::neg_imag() != Rhs::neg_imag()) {
            imag = traits::add(lhs.imag().value, rhs.imag().value);
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag().value, lhs.imag().value);
        } else {
            imag = traits::sub(lhs.imag().value, rhs.imag().value);
        }

        constexpr bool neg_real = Lhs::neg_real() && !Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && !Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
} sub;

namespace detail_ {
template<uZ>
struct mul_stage;
template<>
struct mul_stage<0> {
    template<iZ Lrot, iZ Rrot>
    PCX_AINLINE auto operator()(imag_unit_t<Lrot>, imag_unit_t<Rrot>) {
        return imag_unit_t<(Lrot + Rrot) % 4>{};
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>, tight_cx_vec auto Rhs) {
        return mul_by_j<Rot>(Rhs);
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(tight_cx_vec auto Lhs, imag_unit_t<Rot>) {
        return mul_by_j<Rot>(Lhs);
    }

    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;
        vec real             = traits::mul(lhs.real().value, rhs.real().value);
        vec imag             = traits::mul(lhs.real().value, rhs.imag().value);

        constexpr bool neg_real = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_real() != Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return tupi::make_interim(new_cx_vec{.m_real = real, .m_imag = imag},    //
                                  lhs,
                                  rhs);
    }
};
template<>
struct mul_stage<1> {
    template<tight_cx_vec Res, tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Res res0, Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;
        vec real;
        vec imag;

        constexpr bool imreim_neg_real = Lhs::neg_imag() != Rhs::neg_imag();
        constexpr bool imreim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

        if constexpr (Res::neg_real() == imreim_neg_real) {
            real = traits::fnmadd(lhs.imag().value, rhs.imag().value, res0.real().value);
        } else if constexpr (Res::neg_real()) {
            real = traits::fnmsub(lhs.imag().value, rhs.imag().value, res0.real().value);
        } else {
            real = traits::fmadd(lhs.imag().value, rhs.imag().value, res0.real().value);
        }

        if constexpr (Res::neg_imag() == imreim_neg_imag) {
            imag = traits::fmadd(lhs.imag().value, rhs.real().value, res0.imag().value);
        } else if constexpr (Res::neg_imag()) {
            imag = traits::fmsub(lhs.imag().value, rhs.real().value, res0.imag().value);
        } else {
            imag = traits::fnmadd(lhs.imag().value, rhs.real().value, res0.imag().value);
        }

        constexpr bool neg_real = Res::neg_real() && imreim_neg_real;
        constexpr bool neg_imag = Res::neg_imag() && imreim_neg_imag;

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
};
}    // namespace detail_

inline constexpr struct mul_t : pcx::tupi::compound_op_base {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::mul(lhs.value, rhs.value);
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        return tupi::apply(stage<1>, stage<0>(lhs, rhs));
    };
    template<uZ Power>
    PCX_AINLINE auto operator()(tight_cx_vec auto lhs, imag_unit_t<Power> rhs) {
        return stage<0>(lhs, rhs);
    }
    template<uZ Power>
    PCX_AINLINE auto operator()(imag_unit_t<Power> lhs, tight_cx_vec auto rhs) {
        return stage<0>(lhs, rhs);
    }
    template<uZ Powerl, iZ Powerr>
    PCX_AINLINE auto operator()(imag_unit_t<Powerl> lhs, imag_unit_t<Powerr> rhs) {
        return stage<0>(lhs, rhs);
    }

    template<uZ I>
    constexpr friend auto get_stage(const mul_t&) {
        return detail_::mul_stage<I>{};
    }
    template<uZ I>
    constexpr static detail_::mul_stage<I> stage{};
} mul;

namespace detail_ {
template<uZ>
struct div_stage;

template<>
struct div_stage<0> {
    template<iZ Lrot, iZ Rrot>
    PCX_AINLINE auto operator()(imag_unit_t<Lrot>, imag_unit_t<Rrot>) {
        return imag_unit_t<(Lrot - Rrot) % 4>{};
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>, tight_cx_vec auto Rhs) {
        return mul_by_j<-Rot>(Rhs);
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(tight_cx_vec auto Lhs, imag_unit_t<Rot>) {
        return mul_by_j<-Rot>(Lhs);
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;

        vec real = traits::mul(lhs.real().value, rhs.real().value);
        vec imag = traits::mul(lhs.real().value, rhs.imag().value);

        constexpr bool neg_real  = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag  = Lhs::neg_real() == Rhs::neg_imag();
        vec            rhs_re_sq = traits::mul(rhs.real().value, rhs.real().value);

        using new_cx_vec =
            cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
        return tupi::make_interim(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_re_sq, lhs, rhs);
    };
};
template<>
struct div_stage<1> {
    template<tight_cx_vec Res0, tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Res0 res0, typename Res0::vec_t rhs_re_sq, Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
        using vec            = Lhs::vec_t;

        vec real;
        vec imag;
        vec rhs_abs;

        constexpr bool im_reim_neg_real = Lhs::neg_imag() == Rhs::neg_imag();
        constexpr bool im_reim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

        if constexpr (Res0::neg_real() == im_reim_neg_real) {
            real = traits::fnmadd(lhs.imag().value, rhs.imag().value, res0.real().value);
        } else if constexpr (Res0::neg_real()) {
            real = traits::fnmsub(lhs.imag().value, rhs.imag().value, res0.real().value);
        } else {
            real = traits::fmadd(lhs.imag().value, rhs.imag().value, res0.real().value);
        }

        if constexpr (Res0::neg_imag() == im_reim_neg_imag) {
            imag = traits::fmadd(lhs.imag().value, rhs.real().value, res0.imag().value);
        } else if constexpr (Res0::neg_imag()) {
            imag = traits::fmsub(lhs.imag().value, rhs.real().value, res0.imag().value);
        } else {
            imag = traits::fnmadd(lhs.imag().value, rhs.real().value, res0.imag().value);
        }

        rhs_abs = traits::fmadd(rhs.imag().value, rhs.imag().value, rhs_re_sq.value);

        constexpr bool neg_real = Res0::neg_real() && im_reim_neg_real;
        constexpr bool neg_imag = Res0::neg_imag() && im_reim_neg_imag;

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return tupi::make_interim(new_cx_vec{.m_real = real, .m_imag = imag},    //
                                  rhs_abs);
    };
};
template<>
struct div_stage<2> {
    template<tight_cx_vec Res1>
    PCX_AINLINE auto operator()(Res1 cx_vec, typename Res1::vec_t rhs_abs) const {
        constexpr auto width = Res1::width();
        using traits         = detail_::vec_traits<typename Res1::real_type, width>;
        return Res1{.m_real = traits::div(cx_vec.real().value, rhs_abs.value),
                    .m_imag = traits::div(cx_vec.imag().value, rhs_abs.value)};
    };
};
}    // namespace detail_
inline constexpr struct div_t : tupi::compound_op_base {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::div(lhs.value, rhs.value);
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        return tupi::apply(stage<2>, tupi::apply(stage<1>, stage<0>(lhs, rhs)));
    };
    template<uZ Power>
    PCX_AINLINE auto operator()(tight_cx_vec auto lhs, imag_unit_t<Power> rhs) {
        return stage<0>(lhs, rhs);
    }
    template<uZ Power>
    PCX_AINLINE auto operator()(imag_unit_t<Power> lhs, tight_cx_vec auto rhs) {
        return stage<0>(lhs, rhs);
    }
    template<uZ Powerl, iZ Powerr>
    PCX_AINLINE auto operator()(imag_unit_t<Powerl> lhs, imag_unit_t<Powerr> rhs) {
        return stage<0>(lhs, rhs);
    }
    template<uZ I>
    constexpr friend auto get_stage(const div_t&) {
        return detail_::div_stage<I>{};
    }
    template<uZ I>
    constexpr static detail_::div_stage<I> stage{};
} div;

}    // namespace pcx::simd
#endif
