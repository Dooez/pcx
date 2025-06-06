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
        return detail_::vec_traits<T, Width>::sub(lhs.value, rhs.value);
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
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return tupi::make_tuple(detail_::vec_traits<T, Width>::mul(lhs.value, rhs.value));
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

    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
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
} mul_stage_0;
constexpr inline struct {
    PCX_AINLINE auto operator()(auto&& v) const {
        return std::forward<decltype(v)>(v);
    }
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
} mul_stage_1;
}    // namespace detail_
inline constexpr auto mul = tupi::pass | detail_::mul_stage_0 | tupi::apply | detail_::mul_stage_1;

namespace detail_ {
inline constexpr struct {
    template<iZ Lrot, iZ Rrot>
    PCX_AINLINE auto operator()(imag_unit_t<Lrot>, imag_unit_t<Rrot>) {
        return tupi::make_tuple(imag_unit_t<(Lrot - Rrot) % 4>{});
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(imag_unit_t<Rot>, tight_cx_vec auto Rhs) {
        return tupi::make_tuple(mul_by_j<-Rot>(Rhs));
    }
    template<iZ Rot>
    PCX_AINLINE auto operator()(tight_cx_vec auto Lhs, imag_unit_t<Rot>) {
        return tupi::make_tuple(mul_by_j<-Rot>(Lhs));
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
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
} div_stage_0;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto&& v) -> decltype(auto) {
        return std::forward_as_tuple(std::forward<decltype(v)>(v));
    }
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
} div_stage_1;
inline constexpr struct {
    PCX_AINLINE auto operator()(auto&& v) -> decltype(auto) {
        return std::forward<decltype(v)>(v);
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

}    // namespace pcx::simd
#endif
