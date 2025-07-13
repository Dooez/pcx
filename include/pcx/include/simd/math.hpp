#ifndef PCX_SIMD_MATH_HPP
#define PCX_SIMD_MATH_HPP

#include "pcx/include/simd/traits.hpp"
#include "pcx/include/tupi.hpp"

namespace pcx::simd {

template<typename T, uZ Width>
PCX_AINLINE auto fmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
#ifdef PCX_FMA
    return detail_::vec_traits<T, Width>::fmadd(a.value, b.value, c.value);
#else
    auto ab = detail_::vec_traits<T, Width>::mul(a.value, b.value);
    return detail_::vec_traits<T, Width>::add(ab, c);
#endif
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
#ifdef PCX_FMA
    return detail_::vec_traits<T, Width>::fnmadd(a.value, b.value, c.value);
#else
    auto ab = detail_::vec_traits<T, Width>::mul(a.value, b.value);
    return detail_::vec_traits<T, Width>::sub(c, ab);

#endif
}
template<typename T, uZ Width>
PCX_AINLINE auto fmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
#ifdef PCX_FMA
    return detail_::vec_traits<T, Width>::fmsub(a.value, b.value, c.value);
#else
    auto ab = detail_::vec_traits<T, Width>::mul(a.value, b.value);
    return detail_::vec_traits<T, Width>::sub(ab, c);
#endif
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
#ifdef PCX_FMA
    return detail_::vec_traits<T, Width>::fnmsub(a.value, b.value, c.value);
#else
    auto ab  = detail_::vec_traits<T, Width>::mul(a.value, b.value);
    auto abc = detail_::vec_traits<T, Width>::add(ab, c);
    return detail_::vec_traits<T, Width>::sub(detail_::vec_traits<T, Width>::set1(0), abc);
#endif
}
template<typename T, uZ Width>
PCX_AINLINE auto sqrt(vec<T, Width> a) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::sqrt(a.value);
}

inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return vec<T, Width>{detail_::vec_traits<T, Width>::add(lhs.value, rhs.value)};
    }
    template<typename T, uZ Width, uZ PackSize>
    PCX_AINLINE auto operator()(cx_vec<T, false, false, Width, PackSize> lhs,
                                cx_vec<T, false, false, Width, PackSize> rhs) const {
        using traits = detail_::vec_traits<T, Width>;
        auto real    = traits::add(lhs.real_v(), rhs.real_v());
        auto imag    = traits::add(lhs.imag_v(), rhs.imag_v());
        return cx_vec<T, false, false, Width, PackSize>{.m_real = real, .m_imag = imag};
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;
        vec real;
        vec imag;

        if constexpr (Lhs::neg_real() == Rhs::neg_real()) {
            real = traits::add(lhs.real_v(), rhs.real_v());
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real_v(), lhs.real_v());
        } else {
            real = traits::sub(lhs.real_v(), rhs.real_v());
        }

        if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
            imag = traits::add(lhs.imag_v(), rhs.imag_v());
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag_v(), lhs.imag_v());
        } else {
            imag = traits::sub(lhs.imag_v(), rhs.imag_v());
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
        return vec<T, Width>{detail_::vec_traits<T, Width>::sub(lhs.value, rhs.value)};
    }
    template<typename T, uZ Width, uZ PackSize>
    PCX_AINLINE auto operator()(cx_vec<T, false, false, Width, PackSize> lhs,
                                cx_vec<T, false, false, Width, PackSize> rhs) const {
        using traits = detail_::vec_traits<T, Width>;
        auto real    = traits::sub(lhs.real_v(), rhs.real_v());
        auto imag    = traits::sub(lhs.imag_v(), rhs.imag_v());
        return cx_vec<T, false, false, Width, PackSize>{.m_real = real, .m_imag = imag};
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();

        using vec    = Lhs::vec_t;
        using traits = vec::traits;
        vec real;
        vec imag;

        if constexpr (Lhs::neg_real() != Rhs::neg_real()) {
            real = traits::add(lhs.real_v(), rhs.real_v());
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real_v(), lhs.real_v());
        } else {
            real = traits::sub(lhs.real_v(), rhs.real_v());
        }

        if constexpr (Lhs::neg_imag() != Rhs::neg_imag()) {
            imag = traits::add(lhs.imag_v(), rhs.imag_v());
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag_v(), lhs.imag_v());
        } else {
            imag = traits::sub(lhs.imag_v(), rhs.imag_v());
        }

        constexpr bool neg_real = Lhs::neg_real() && !Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && !Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
} sub;

namespace detail_ {
inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const {
        return tupi::make_tuple(vec<T, Width>{detail_::vec_traits<T, Width>::mul(lhs.value, rhs.value)});
    }
    template<iZ Lrot, iZ Rrot>
    PCX_AINLINE auto operator()(imag_unit_t<Lrot>, imag_unit_t<Rrot>) const {
        return tupi::make_tuple(imag_unit_t<(Lrot + Rrot) % 4>{});
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>, tight_cx_vec auto Rhs) const {
        return tupi::make_tuple(mul_by_j<Rot>(Rhs));
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(tight_cx_vec auto Lhs, imag_unit_t<Rot>) const {
        return tupi::make_tuple(mul_by_j<Rot>(Lhs));
    }

#ifdef PCX_FMA
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;
        vec real             = traits::mul(lhs.real_v(), rhs.real_v());
        vec imag             = traits::mul(lhs.real_v(), rhs.imag_v());

        constexpr bool neg_real = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_real() != Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return tupi::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag},    //
                                lhs,
                                rhs);
    }
#else
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;
        vec real0            = traits::mul(lhs.real_v(), rhs.real_v());
        vec imag0            = traits::mul(lhs.real_v(), rhs.imag_v());
        vec real1            = traits::mul(lhs.imag_v(), rhs.imag_v());
        vec imag1            = traits::mul(lhs.imag_v(), rhs.real_v());

        constexpr bool neg_real0 = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag0 = Lhs::neg_real() != Rhs::neg_imag();

        constexpr bool neg_real1 = Lhs::neg_imag() == Rhs::neg_imag();
        constexpr bool neg_imag1 = Lhs::neg_imag() != Rhs::neg_real();

        using new_cx_vec0 = cx_vec<typename vec::value_type, neg_real0, neg_imag0, width, Lhs::pack_size()>;
        using new_cx_vec1 = cx_vec<typename vec::value_type, neg_real1, neg_imag1, width, Lhs::pack_size()>;
        return tupi::make_tuple(new_cx_vec0{.m_real = real0, .m_imag = imag0},    //
                                new_cx_vec1{.m_real = real1, .m_imag = imag1});
    }
#endif
} mul_stage_0;
constexpr inline struct {
    PCX_AINLINE auto operator()(auto v) const {
        return v;
    }
#ifdef PCX_FMA
    template<tight_cx_vec Res, tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Res res0, Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;
        vec real;
        vec imag;

        constexpr bool imreim_neg_real = Lhs::neg_imag() != Rhs::neg_imag();
        constexpr bool imreim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

        if constexpr (Res::neg_real() == imreim_neg_real) {
            real = traits::fnmadd(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        } else if constexpr (Res::neg_real()) {
            real = traits::fnmsub(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        } else {
            real = traits::fmadd(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        }

        if constexpr (Res::neg_imag() == imreim_neg_imag) {
            imag = traits::fmadd(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        } else if constexpr (Res::neg_imag()) {
            imag = traits::fmsub(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        } else {
            imag = traits::fnmadd(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        }

        constexpr bool neg_real = Res::neg_real() && imreim_neg_real;
        constexpr bool neg_imag = Res::neg_imag() && imreim_neg_imag;

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
#else
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;
        vec real;
        vec imag;

        if constexpr (Lhs::neg_real() == Rhs::neg_real()) {
            real = traits::add(lhs.real_v(), rhs.real_v());
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real_v(), lhs.real_v());
        } else {
            real = traits::sub(lhs.real_v(), rhs.real_v());
        }

        if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
            imag = traits::add(lhs.imag_v(), rhs.imag_v());
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag_v(), lhs.imag_v());
        } else {
            imag = traits::sub(lhs.imag_v(), rhs.imag_v());
        }

        constexpr bool neg_real = Lhs::neg_real() && Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
#endif
} mul_stage_1;
}    // namespace detail_
inline constexpr auto mul = tupi::pass | detail_::mul_stage_0 | tupi::apply | detail_::mul_stage_1;

namespace detail_ {
inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const {
        return tupi::make_tuple(vec<T, Width>{detail_::vec_traits<T, Width>::div(lhs.value, rhs.value)});
    }
    template<iZ Lrot, iZ Rrot>
    PCX_AINLINE auto operator()(imag_unit_t<Lrot>, imag_unit_t<Rrot>) const {
        return tupi::make_tuple(imag_unit_t<(Lrot - Rrot) % 4>{});
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>, tight_cx_vec auto Rhs) const {
        return tupi::make_tuple(mul_by_j<-Rot>(Rhs));
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(tight_cx_vec auto Lhs, imag_unit_t<Rot>) const {
        return tupi::make_tuple(mul_by_j<-Rot>(Lhs));
    }
#ifdef PCX_FMA
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;

        vec real = traits::mul(lhs.real_v(), rhs.real_v());
        vec imag = traits::mul(lhs.real_v(), rhs.imag_v());

        constexpr bool neg_real = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_real() == Rhs::neg_imag();

        vec rhs_re_sq = traits::mul(rhs.real_v(), rhs.real_v());

        using new_cx_vec =
            cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
        return tupi::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_re_sq, lhs, rhs);
    };
#else
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;

        vec real0 = traits::mul(lhs.real_v(), rhs.real_v());
        vec imag0 = traits::mul(lhs.real_v(), rhs.imag_v());
        vec real1 = traits::mul(lhs.imag_v(), rhs.imag_v());
        vec imag1 = traits::mul(lhs.imag_v(), rhs.real_v());

        vec rhs_re_sq = traits::mul(rhs.real_v(), rhs.real_v());
        vec rhs_im_sq = traits::mul(rhs.imag_v(), rhs.imag_v());

        constexpr bool neg_real0 = Lhs::neg_real() != Rhs::neg_real();
        constexpr bool neg_imag0 = Lhs::neg_real() != Rhs::neg_imag();

        constexpr bool neg_real1 = Lhs::neg_imag() == Rhs::neg_imag();
        constexpr bool neg_imag1 = Lhs::neg_imag() != Rhs::neg_real();

        using new_cx_vec0 = cx_vec<typename vec::value_type, neg_real0, neg_imag0, width, Lhs::pack_size()>;
        using new_cx_vec1 = cx_vec<typename vec::value_type, neg_real1, neg_imag1, width, Lhs::pack_size()>;
        return tupi::make_tuple(new_cx_vec0{.m_real = real0, .m_imag = imag0},    //
                                new_cx_vec1{.m_real = real1, .m_imag = imag1},
                                rhs_re_sq,
                                rhs_im_sq);
    }
#endif
} div_stage_0;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto v) const {
        return std::make_tuple(v);
    }
#ifdef PCX_FMA
    template<tight_cx_vec Res0, tight_cx_vec Lhs, tight_cx_vec Rhs>
    PCX_AINLINE auto operator()(Res0 res0, typename Res0::vec_t rhs_re_sq, Lhs lhs, Rhs rhs) const {
        constexpr auto width = Lhs::width();
        using vec            = Lhs::vec_t;
        using traits         = vec::traits;

        vec real;
        vec imag;
        vec rhs_abs;

        constexpr bool im_reim_neg_real = Lhs::neg_imag() == Rhs::neg_imag();
        constexpr bool im_reim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

        if constexpr (Res0::neg_real() == im_reim_neg_real) {
            real = traits::fnmadd(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        } else if constexpr (Res0::neg_real()) {
            real = traits::fnmsub(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        } else {
            real = traits::fmadd(lhs.imag_v(), rhs.imag_v(), res0.real_v());
        }

        if constexpr (Res0::neg_imag() == im_reim_neg_imag) {
            imag = traits::fmadd(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        } else if constexpr (Res0::neg_imag()) {
            imag = traits::fmsub(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        } else {
            imag = traits::fnmadd(lhs.imag_v(), rhs.real_v(), res0.imag_v());
        }

        rhs_abs = traits::fmadd(rhs.imag_v(), rhs.imag_v(), rhs_re_sq.value);

        constexpr bool neg_real = Res0::neg_real() && im_reim_neg_real;
        constexpr bool neg_imag = Res0::neg_imag() && im_reim_neg_imag;

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return tupi::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag},    //
                                rhs_abs);
    };
#else
    template<tight_cx_vec Res00, tight_cx_vec Res01>
    PCX_AINLINE auto operator()(Res00                 r0,    //
                                Res01                 r1,
                                typename Res00::vec_t rhs_re_sq,
                                typename Res00::vec_t rhs_im_sq) const {
        constexpr auto width = Res00::width();
        using vec            = Res00::vec_t;
        using traits         = vec::traits;
        vec real;
        vec imag;

        if constexpr (Res00::neg_real() == Res01::neg_real()) {
            real = traits::add(r0.real_v(), r1.real_v());
        } else if constexpr (Res00::neg_real()) {
            real = traits::sub(r1.real_v(), r0.real_v());
        } else {
            real = traits::sub(r0.real_v(), r1.real_v());
        }

        if constexpr (Res00::neg_imag() == Res01::neg_imag()) {
            imag = traits::add(r0.imag_v(), r1.imag_v());
        } else if constexpr (Res00::neg_imag()) {
            imag = traits::sub(r1.imag_v(), r0.imag_v());
        } else {
            imag = traits::sub(r0.imag_v(), r1.imag_v());
        }
        auto rhs_abs = traits::add(rhs_re_sq + rhs_im_sq);

        constexpr bool neg_real = Res00::neg_real() && Res01::neg_real();
        constexpr bool neg_imag = Res00::neg_imag() && Res01::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Res00::pack_size()>;
        return tupi::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_abs);
    }
#endif
} div_stage_1;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto v) -> decltype(auto) {
        return v;
    }
    template<tight_cx_vec Res1>
    PCX_AINLINE auto operator()(Res1 cx_vec, typename Res1::vec_t rhs_abs) const {
        constexpr auto width = Res1::width();
        using traits         = detail_::vec_traits<typename Res1::real_type, width>;
        return Res1{.m_real = traits::div(cx_vec.real_v(), rhs_abs.value),
                    .m_imag = traits::div(cx_vec.imag_v(), rhs_abs.value)};
    };
} div_stage_2;
}    // namespace detail_
inline constexpr auto div = tupi::pass                //
                            | detail_::div_stage_0    //
                            | tupi::apply             //
                            | detail_::div_stage_1    //
                            | tupi::apply             //
                            | detail_::div_stage_2;

namespace detail_ {
inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> v) const {
        return tupi::make_tuple(vec<T, Width>{detail_::vec_traits<T, Width>::mul(v.value, v.value)});
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>) const {
        return tupi::make_tuple(imag_unit_t<0>{});
    }
    template<tight_cx_vec V>
    PCX_AINLINE auto operator()(V cxvec) const {
        using T              = V::real_type;
        constexpr auto width = V::width();

        auto real = cxvec.real();
        auto imag = cxvec.imag();
        return tupi::make_tuple(vec<T, width>{detail_::vec_traits<T, width>::mul(real.value, real.value)},
                                vec<T, width>{detail_::vec_traits<T, width>::mul(imag.value, imag.value)});
    }
} sq_abs_stage_0;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto v) {
        return v;
    }
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> re, vec<T, Width> im) const {
        return vec<T, Width>{detail_::vec_traits<T, Width>::add(re.value, im.value)};
    }
} sq_abs_stage_1;

inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> v) const {
        return tupi::make_tuple(vec<T, Width>{detail_::vec_traits<T, Width>::abs(v.value)});
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>) const {
        return tupi::make_tuple(imag_unit_t<0>{});
    }
    template<tight_cx_vec V>
    PCX_AINLINE auto operator()(V cxvec) const {
        using T              = V::real_type;
        constexpr auto width = V::width();

        auto real = cxvec.real();
        auto imag = cxvec.imag();
        return tupi::make_tuple(vec<T, width>{detail_::vec_traits<T, width>::mul(real.value, real.value)},
                                vec<T, width>{detail_::vec_traits<T, width>::mul(imag.value, imag.value)});
    }
} abs_stage_0;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto v) {
        return tupi::make_tuple(v);
    }
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> re, vec<T, Width> im) const {
        return tupi::make_tuple(vec<T, Width>{detail_::vec_traits<T, Width>::add(re.value, im.value)},
                                std::true_type{});
    }
} abs_stage_1;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto v) {
        return v;
    }
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> sq_abs, std::true_type) const {
        return vec<T, Width>{detail_::vec_traits<T, Width>::sqrt(sq_abs.value)};
    }
} abs_stage_2;
}    // namespace detail_

inline constexpr auto sq_abs = tupi::pass                   //
                               | detail_::sq_abs_stage_0    //
                               | tupi::apply                //
                               | detail_::sq_abs_stage_1;

inline constexpr auto abs = tupi::pass                //
                            | detail_::abs_stage_0    //
                            | tupi::apply             //
                            | detail_::abs_stage_1    //
                            | tupi::apply             //
                            | detail_::abs_stage_2;


constexpr struct {
    PCX_AINLINE static auto operator()(any_cx_vec auto a, any_cx_vec auto b) {
        return std::make_tuple(add(a, b), sub(a, b));
    }
} btfly{};
}    // namespace pcx::simd
#endif
