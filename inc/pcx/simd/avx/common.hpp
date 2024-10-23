#ifndef PCX_SIMD_MATH_HPP
#define PCX_SIMD_MATH_HPP
#include "pcx/types.hpp"

#include <immintrin.h>

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline
namespace pcx::simd {

namespace detail_ {

#ifdef PCX_AVX512

template<>
struct default_vec_size<f32> {
    static constexpr uZ value = 16;
};
template<>
struct default_vec_size<f64> {
    static constexpr uZ value = 8;
};

template<>
struct vec_traits<f32, 16> {
    using type = __m512;

    PCX_AINLINE static auto set1(f32 value) {
        return _mm512_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm512_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, type vec) {
        _mm512_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm512_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm512_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm512_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm512_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm512_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm512_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm512_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm512_fnmsub_ps(a, b, c);
    }
};
template<>
struct vec_traits<f64, 8> {
    using type = __m512d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm512_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm512_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, type vec) {
        _mm512_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm512_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm512_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm512_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm512_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm512_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm512_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm512_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm512_fnmsub_pd(a, b, c);
    }
};

#else

template<>
struct default_vec_size<f32> {
    static constexpr uZ value = 8;
};
template<>
struct default_vec_size<f64> {
    static constexpr uZ value = 4;
};

#endif

template<>
struct vec_traits<f32, 8> {
    using type = __m256;
};
template<>
struct vec_traits<f32, 4> {
    using type = __m128;
};

template<>
struct vec_traits<f64, 4> {
    using type = __m256d;
};
template<>
struct vec_traits<f64, 2> {
    using type = __m128d;
};
}    // namespace detail_


template<uZ Size = detail_::default_vec_size<f32>::value>
PCX_AINLINE auto broadcast(f32 src) -> vec<f32, Size> {
#ifdef PCX_AVX512
    return _mm512_set1_ps(src);
#else
    return _m

#endif
}

}    // namespace pcx::simd
#undef PCX_AINLINE
#endif
