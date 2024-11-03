#ifndef PCX_SIMD_MATH_HPP
#define PCX_SIMD_MATH_HPP
#include "pcx/include/tuple.hpp"
#include "pcx/include/types.hpp"

#define PCX_AINLINE   [[gnu::always_inline, clang::always_inline]] inline
#define PCX_LBDINLINE [[gnu::always_inline, clang::always_inline]]
namespace pcx::simd {

template<typename T, uZ Width>
PCX_AINLINE auto fmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fmadd(a.native, b.native, c.native);
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmadd(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fnmadd(a.native, b.native, c.native);
}
template<typename T, uZ Width>
PCX_AINLINE auto fmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fmsub(a.native, b.native, c.native);
}
template<typename T, uZ Width>
PCX_AINLINE auto fnmsub(vec<T, Width> a, vec<T, Width> b, vec<T, Width> c) -> vec<T, Width> {
    return detail_::vec_traits<T, Width>::fnmsub(a.native, b.native, c.native);
}

constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::add(lhs.native, rhs.native);
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
            real = traits::add(lhs.real().native, rhs.real().native);
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real().native, lhs.real().native);
        } else {
            real = traits::sub(lhs.real().native, rhs.real().native);
        }

        if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
            imag = traits::add(lhs.imag().native, rhs.imag().native);
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag().native, lhs.imag().native);
        } else {
            imag = traits::sub(lhs.imag().native, rhs.imag().native);
        }

        constexpr bool neg_real = Lhs::neg_real() && Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
} add;

constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::sub(lhs.native, rhs.native);
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
            real = traits::add(lhs.real().native, rhs.real().native);
        } else if constexpr (Lhs::neg_real()) {
            real = traits::sub(rhs.real().native, lhs.real().native);
        } else {
            real = traits::sub(lhs.real().native, rhs.real().native);
        }

        if constexpr (Lhs::neg_imag() != Rhs::neg_imag()) {
            imag = traits::add(lhs.imag().native, rhs.imag().native);
        } else if constexpr (Lhs::neg_imag()) {
            imag = traits::sub(rhs.imag().native, lhs.imag().native);
        } else {
            imag = traits::sub(lhs.imag().native, rhs.imag().native);
        }

        constexpr bool neg_real = Lhs::neg_real() && !Rhs::neg_real();
        constexpr bool neg_imag = Lhs::neg_imag() && !Rhs::neg_imag();

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
        return new_cx_vec{.m_real = real, .m_imag = imag};
    }
} sub;

constexpr struct mul_t : pcx::i::multi_stage_op<2> {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::mul(lhs.native, rhs.native);
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        return stage<1>(stage<0>(lhs, rhs));
    };

private:
    template<uZ I>
    struct stage_t;

public:
    template<uZ I>
    constexpr static stage_t<I> stage;
    template<>
    struct stage_t<0> {
        template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
            constexpr auto width = Lhs::width();
            using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
            using vec            = Lhs::vec_t;
            vec real             = traits::mul(lhs.real().native, rhs.real().native);
            vec imag             = traits::mul(lhs.real().native, rhs.real().native);

            constexpr bool neg_real = Lhs::neg_real() != Rhs::neg_real();
            constexpr bool neg_imag = Lhs::neg_real() != Rhs::neg_imag();

            using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
            return i::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, lhs, rhs);
        }
    };
    template<>
    struct stage_t<1> {
        template<tight_cx_vec Res, tight_cx_vec Lhs, tight_cx_vec Rhs>
        PCX_AINLINE auto operator()(i::tuple<Res, Lhs, Rhs> args) const {
            auto [res0, lhs, rhs] = args;

            constexpr auto width = Lhs::width();
            using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
            using vec            = Lhs::vec_t;
            vec real;
            vec imag;

            constexpr bool imreim_neg_real = Lhs::neg_imag() != Rhs::neg_imag();
            constexpr bool imreim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

            if constexpr (Res::neg_real() == imreim_neg_real) {
                real = traits::fnmadd(lhs.imag().native, rhs.imag().native, res0.real().native);
            } else if constexpr (Res::neg_real()) {
                real = traits::fnmsub(lhs.imag().native, rhs.imag().native, res0.real().native);
            } else {
                real = traits::fmadd(lhs.imag().native, rhs.imag().native, res0.real().native);
            }

            if constexpr (Res::neg_imag() == imreim_neg_imag) {
                imag = traits::fmadd(lhs.imag().native, rhs.real().native, res0.imag().native);
            } else if constexpr (Res::neg_imag()) {
                imag = traits::fmsub(lhs.imag().native, rhs.real().native, res0.imag().native);
            } else {
                imag = traits::fnmadd(lhs.imag().native, rhs.real().native, res0.imag().native);
            }

            constexpr bool neg_real = Res::neg_real() && imreim_neg_real;
            constexpr bool neg_imag = Res::neg_imag() && imreim_neg_imag;

            using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
            return new_cx_vec{.m_real = real, .m_imag = imag};
        }
    };
} mul;

constexpr struct div_t : i::multi_stage_op<3> {
    template<typename T, uZ Width>
    PCX_AINLINE auto operator()(vec<T, Width> lhs, vec<T, Width> rhs) const -> vec<T, Width> {
        return detail_::vec_traits<T, Width>::div(lhs.native, rhs.native);
    }
    template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        requires compatible_cx_vec<Lhs, Rhs>
    PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
        return stage<2>(stage<1>(stage<0>(lhs, rhs)));
    };

private:
    template<uZ I>
    struct stage_t;

public:
    template<uZ I>
    constexpr static stage_t<I> stage;

    template<>
    struct stage_t<0> {
        template<tight_cx_vec Lhs, tight_cx_vec Rhs>
        PCX_AINLINE auto operator()(Lhs lhs, Rhs rhs) const {
            constexpr auto width = Lhs::width();
            using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
            using vec            = Lhs::vec_t;

            vec real = traits::mul(lhs.real().native, rhs.real().native);
            vec imag = traits::mul(lhs.real().native, rhs.real().native);

            constexpr bool neg_real  = Lhs::neg_real() != Rhs::neg_real();
            constexpr bool neg_imag  = Lhs::neg_real() != Rhs::neg_imag();
            vec            rhs_re_sq = traits::mul(rhs.real().native, rhs.real().native);

            using new_cx_vec =
                cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
            return i::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_re_sq, lhs, rhs);
        };
    };
    template<>
    struct stage_t<1> {
        template<tight_cx_vec Res0, tight_cx_vec Lhs, tight_cx_vec Rhs>
        PCX_AINLINE auto operator()(i::tuple<Res0, typename Res0::vec_t, Lhs, Rhs> args) const {
            constexpr auto width = Lhs::width();
            using traits         = detail_::vec_traits<typename Lhs::real_type, width>;
            using vec            = Lhs::vec_t;

            auto [res0, rhs_re_sq, lhs, rhs] = args;
            vec real;
            vec imag;
            vec rhs_abs;

            constexpr bool im_reim_neg_real = Lhs::neg_imag() != Rhs::neg_imag();
            constexpr bool im_reim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

            if constexpr (Res0::neg_real() == im_reim_neg_real) {
                real = traits::fnmadd(lhs.imag().native, rhs.imag().native, res0.real().native);
            } else if constexpr (Res0::neg_real()) {
                real = traits::fnmsub(lhs.imag().native, rhs.imag().native, res0.real().native);
            } else {
                real = traits::fmadd(lhs.imag().native, rhs.imag().native, res0.real().native);
            }

            if constexpr (Res0::neg_imag() == im_reim_neg_imag) {
                imag = traits::fmadd(lhs.imag().native, rhs.real().native, res0.imag().native);
            } else if constexpr (Res0::neg_imag()) {
                imag = traits::fmsub(lhs.imag().native, rhs.real().native, res0.imag().native);
            } else {
                imag = traits::fnmadd(lhs.imag().native, rhs.real().native, res0.imag().native);
            }

            rhs_abs = traits::fmadd(rhs.imag().native, rhs.imag().native, rhs_re_sq.native);

            constexpr bool neg_real = Res0::neg_real() && im_reim_neg_real;
            constexpr bool neg_imag = Res0::neg_imag() && im_reim_neg_imag;

            using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, width, Lhs::pack_size()>;
            return i::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_abs);
        };
    };
    template<>
    struct stage_t<2> {
        template<tight_cx_vec Res1>
        PCX_AINLINE auto operator()(i::tuple<Res1, typename Res1::vec_t> args) const {
            constexpr auto width   = Res1::width();
            using traits           = detail_::vec_traits<typename Res1::real_type, width>;
            auto [cx_vec, rhs_abs] = args;
            return Res1{.m_real = traits::div(cx_vec.real().native, rhs_abs.native),
                        .m_imag = traits::div(cx_vec.imag().native, rhs_abs.native)};
        };
    };

} div;

}    // namespace pcx::simd
#undef PCX_AINLINE
#undef PCX_LBDINLINE
#endif
