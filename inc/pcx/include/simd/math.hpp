#ifndef PCX_SIMD_MATH_HPP
#define PCX_SIMD_MATH_HPP
#include "pcx/include/types.hpp"

#include <tuple>

#define PCX_AINLINE   [[gnu::always_inline, clang::always_inline]] inline
#define PCX_LBDINLINE [[gnu::always_inline, clang::always_inline]]
namespace pcx::simd {

template<typename T, uZ Size>
PCX_AINLINE auto add(vec<T, Size> lhs, vec<T, Size> rhs) {
    return detail_::vec_traits<T, Size>::add(lhs.value, rhs.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto sub(vec<T, Size> lhs, vec<T, Size> rhs) {
    return detail_::vec_traits<T, Size>::sub(lhs.value, rhs.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto mul(vec<T, Size> lhs, vec<T, Size> rhs) {
    return detail_::vec_traits<T, Size>::mul(lhs.value, rhs.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto div(vec<T, Size> lhs, vec<T, Size> rhs) {
    return detail_::vec_traits<T, Size>::div(lhs.value, rhs.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto fmadd(vec<T, Size> a, vec<T, Size> b, vec<T, Size> c) {
    return detail_::vec_traits<T, Size>::fmadd(a.value, b.value, c.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto fnmadd(vec<T, Size> a, vec<T, Size> b, vec<T, Size> c) {
    return detail_::vec_traits<T, Size>::fnmadd(a.value, b.value, c.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto fmsub(vec<T, Size> a, vec<T, Size> b, vec<T, Size> c) {
    return detail_::vec_traits<T, Size>::fmsub(a.value, b.value, c.value);
}
template<typename T, uZ Size>
PCX_AINLINE auto fnmsub(vec<T, Size> a, vec<T, Size> b, vec<T, Size> c) {
    return detail_::vec_traits<T, Size>::fnmsub(a.value, b.value, c.value);
}


template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    requires compatible_cx_vec<Lhs, Rhs>
PCX_AINLINE auto add(Lhs lhs, Rhs rhs) {
    using vec = Lhs::vec_t;
    vec real;
    vec imag;

    if constexpr (Lhs::neg_real() == Rhs::neg_real()) {
        real = add(lhs.real(), rhs.real());
    } else if constexpr (Lhs::neg_real()) {
        real = sub(rhs.real(), lhs.real());
    } else {
        real = sub(lhs.real(), rhs.real());
    }

    if constexpr (Lhs::neg_imag() == Rhs::neg_imag()) {
        imag = add(lhs.imag(), rhs.imag());
    } else if constexpr (Lhs::neg_imag()) {
        imag = sub(rhs.imag(), lhs.imag());
    } else {
        imag = sub(lhs.imag(), rhs.imag());
    }

    constexpr bool neg_real = Lhs::neg_real() && Rhs::neg_real();
    constexpr bool neg_imag = Lhs::neg_imag() && Rhs::neg_imag();

    using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
    return new_cx_vec{.m_real = real, .m_imag = imag};
}

template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    requires compatible_cx_vec<Lhs, Rhs>
PCX_AINLINE auto sub(Lhs lhs, Rhs rhs) {
    using vec = Lhs::vec_t;
    vec real;
    vec imag;

    if constexpr (Lhs::neg_real() != Rhs::neg_real()) {
        real = add(lhs.real(), rhs.real());
    } else if constexpr (Lhs::neg_real()) {
        real = sub(rhs.real(), lhs.real());
    } else {
        real = sub(lhs.real(), rhs.real());
    }

    if constexpr (Lhs::neg_imag() != Rhs::neg_imag()) {
        imag = add(lhs.imag(), rhs.imag());
    } else if constexpr (Lhs::neg_imag()) {
        imag = sub(rhs.imag(), lhs.imag());
    } else {
        imag = sub(lhs.imag(), rhs.imag());
    }

    constexpr bool neg_real = Lhs::neg_real() && !Rhs::neg_real();
    constexpr bool neg_imag = Lhs::neg_imag() && !Rhs::neg_imag();

    using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
    return new_cx_vec{.m_real = real, .m_imag = imag};
};

namespace detail_ {
template<tight_cx_vec Lhs, tight_cx_vec Rhs>
PCX_AINLINE auto mul_step1(Lhs lhs, Rhs rhs) {
    using vec = Lhs::vec_t;
    vec real  = mul(lhs.real(), rhs.real());
    vec imag  = mul(lhs.real(), rhs.real());

    constexpr bool neg_real = Lhs::neg_real() != Rhs::neg_real();
    constexpr bool neg_imag = Lhs::neg_real() != Rhs::neg_imag();

    using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
    return new_cx_vec{.m_real = real, .m_imag = imag};
}
template<tight_cx_vec Res, tight_cx_vec Lhs, tight_cx_vec Rhs>
PCX_AINLINE auto mul_step2(Res step1_res, Lhs lhs, Rhs rhs) {
    using vec = Lhs::vec_t;
    vec real;
    vec imag;

    constexpr bool imreim_neg_real = Lhs::neg_imag() != Rhs::neg_imag();
    constexpr bool imreim_neg_imag = Lhs::neg_imag() != Rhs::neg_real();

    if constexpr (Res::neg_real() == imreim_neg_real) {
        real = fnmadd(lhs.imag(), rhs.imag(), step1_res.real());
    } else if constexpr (Res::neg_real()) {
        real = fnmsub(lhs.imag(), rhs.imag(), step1_res.real());
    } else {
        real = fmadd(lhs.imag(), rhs.imag(), step1_res.real());
    }

    if constexpr (Res::neg_imag() == imreim_neg_imag) {
        imag = fmadd(lhs.imag(), rhs.real(), step1_res.imag());
    } else if constexpr (Res::neg_imag()) {
        imag = fmsub(lhs.imag(), rhs.real(), step1_res.imag());
    } else {
        imag = fnmadd(lhs.imag(), rhs.real(), step1_res.imag());
    }

    constexpr bool neg_real = Res::neg_real() && imreim_neg_real;
    constexpr bool neg_imag = Res::neg_imag() && imreim_neg_imag;

    using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, Lhs::width(), Lhs::pack_size()>;
    return new_cx_vec{.m_real = real, .m_imag = imag};
}
}    // namespace detail_

template<tight_cx_vec... Lhs, tight_cx_vec... Rhs>
    requires(compatible_cx_vec<Lhs, Rhs> && ...)
[[gnu::flatten]] PCX_AINLINE auto mul_tuples(std::tuple<Lhs...> lhs, std::tuple<Rhs...> rhs) {
    auto res1 = []<uZ... Is> [[gnu::flatten]] PCX_LBDINLINE(auto lhs, auto rhs, std::index_sequence<Is...>) {
        return std::make_tuple(detail_::mul_step1(std::get<Is>(lhs), std::get<Is>(rhs))...);
    }(lhs, rhs, std::index_sequence_for<Lhs...>{});

    return []<uZ... Is> [[gnu::flatten]] PCX_LBDINLINE(
               auto res1, auto lhs, auto rhs, std::index_sequence<Is...>) {
        return std::make_tuple(
            detail_::mul_step2(std::get<Is>(res1), std::get<Is>(lhs), std::get<Is>(rhs))...);
    }(res1, lhs, rhs, std::index_sequence_for<Lhs...>{});
};

template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    requires compatible_cx_vec<Lhs, Rhs>
[[gnu::flatten]] PCX_AINLINE auto mul(Lhs lhs, Rhs rhs) {
    return std::get<0>(mul_tuples(std::make_tuple(lhs), std::make_tuple(rhs)));
};

template<tight_cx_vec... Lhs, tight_cx_vec... Rhs>
    requires(compatible_cx_vec<Lhs, Rhs> && ...)
[[gnu::flatten]] PCX_AINLINE auto div_tuples(std::tuple<Lhs...> lhs, std::tuple<Rhs...> rhs) {
    constexpr auto step1 = []<typename L, typename R> PCX_LBDINLINE(L lhs, R rhs) {
        using vec = L::vec_t;
        vec real  = mul(lhs.real(), rhs.real());
        vec imag  = mul(lhs.real(), rhs.real());

        constexpr bool neg_real  = L::neg_real() != R::neg_real();
        constexpr bool neg_imag  = L::neg_real() != R::neg_imag();
        vec            rhs_re_sq = mul(rhs.real(), rhs.real());

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, L::width(), L::pack_size()>;
        return std::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_re_sq);
    };
    constexpr auto step2 = []<typename L, typename R> PCX_LBDINLINE(auto res1_tup, L lhs, R rhs) {
        using vec = L::vec_t;
        vec  real;
        vec  imag;
        auto res1      = std::get<0>(res1_tup);
        auto rhs_re_sq = std::get<1>(res1_tup);
        using Res      = decltype(res1);

        constexpr bool imreim_neg_real = L::neg_imag() != R::neg_imag();
        constexpr bool imreim_neg_imag = L::neg_imag() != R::neg_real();

        if constexpr (Res::neg_real() == imreim_neg_real) {
            real = fnmadd(lhs.imag(), rhs.imag(), res1.real());
        } else if constexpr (Res::neg_real()) {
            real = fnmsub(lhs.imag(), rhs.imag(), res1.real());
        } else {
            real = fmadd(lhs.imag(), rhs.imag(), res1.real());
        }

        if constexpr (Res::neg_imag() == imreim_neg_imag) {
            imag = fmadd(lhs.imag(), rhs.real(), res1.imag());
        } else if constexpr (Res::neg_imag()) {
            imag = fmsub(lhs.imag(), rhs.real(), res1.imag());
        } else {
            imag = fnmadd(lhs.imag(), rhs.real(), res1.imag());
        }

        auto rhs_abs = fmadd(rhs.imag(), rhs.imag, rhs_re_sq);

        constexpr bool neg_real = Res::neg_real() && imreim_neg_real;
        constexpr bool neg_imag = Res::neg_imag() && imreim_neg_imag;

        using new_cx_vec = cx_vec<typename vec::value_type, neg_real, neg_imag, L::width(), L::pack_size()>;
        return std::make_tuple(new_cx_vec{.m_real = real, .m_imag = imag}, rhs_abs);
    };
    constexpr auto step3 = [] PCX_LBDINLINE(auto res2_tup) {
        auto cx_vec    = std::get<0>(res2_tup);
        auto rhs_abs   = std::get<1>(res2_tup);
        using cx_vec_t = decltype(cx_vec);

        return cx_vec_t{.m_real = div(cx_vec.real(), rhs_abs), .m_imag = div(cx_vec.imag(), rhs_abs)};
    };
    auto res1 = [step1]<uZ... Is> PCX_LBDINLINE(auto lhs, auto rhs, std::index_sequence<Is...>) {
        return std::make_tuple(step1(std::get<Is>(lhs), std::get<Is>(rhs))...);
    }(lhs, rhs, std::index_sequence_for<Lhs...>{});

    auto res2 = [step2]<uZ... Is> PCX_LBDINLINE(auto res1, auto lhs, auto rhs, std::index_sequence<Is...>) {
        return std::make_tuple(step2(std::get<Is>(res1), std::get<Is>(lhs), std::get<Is>(rhs))...);
    }(res1, lhs, rhs, std::index_sequence_for<Lhs...>{});

    return [step3]<uZ... Is> PCX_LBDINLINE(auto res2, std::index_sequence<Is...>) {
        return std::make_tuple(step3(std::get<Is>(res2))...);
    }(res1, lhs, rhs, std::index_sequence_for<Lhs...>{});
};

template<tight_cx_vec Lhs, tight_cx_vec Rhs>
    requires compatible_cx_vec<Lhs, Rhs>
[[gnu::flatten]] PCX_AINLINE auto div(Lhs lhs, Rhs rhs) {
    return std::get<0>(div_tuples(std::make_tuple(lhs), std::make_tuple(rhs)));
};

}    // namespace pcx::simd
#undef PCX_AINLINE
#undef PCX_LBDINLINE
#endif
