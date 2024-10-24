#ifndef PCX_SIMD_TRAITS_HPP
#define PCX_SIMD_TRAITS_HPP

#include <immintrin.h>
namespace pcx {
using uZ  = std::size_t;
using f32 = float;
using f64 = double;
}    // namespace pcx

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline
namespace pcx::simd::detail_ {

template<typename T>
struct max_vec_width;
template<typename T, uZ Width>
struct vec_traits;

#ifdef PCX_AVX512
template<>
struct max_vec_width<f32> {
    static constexpr uZ value = 16;
};
template<>
struct max_vec_width<f64> {
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
struct max_vec_width<f32> {
    static constexpr uZ value = 8;
};
template<>
struct max_vec_width<f64> {
    static constexpr uZ value = 4;
};

#endif

template<>
struct vec_traits<f32, 8> {
    using type = __m256;
    PCX_AINLINE static auto set1(f32 value) {
        return _mm256_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, type vec) {
        _mm256_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm256_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm256_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm256_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm256_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm256_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm256_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm256_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm256_fnmsub_ps(a, b, c);
    }
};
template<>
struct vec_traits<f32, 4> {
    using type = __m128;
    PCX_AINLINE static auto set1(f32 value) {
        return _mm_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, type vec) {
        _mm_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm_fnmsub_ps(a, b, c);
    }
};

template<>
struct vec_traits<f64, 4> {
    using type = __m256d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm256_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm256_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, type vec) {
        _mm256_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm256_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm256_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm256_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm256_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm256_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm256_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm256_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm256_fnmsub_pd(a, b, c);
    }
};
template<>
struct vec_traits<f64, 2> {
    using type = __m128d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, type vec) {
        _mm_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(type lhs, type rhs) {
        return _mm_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(type lhs, type rhs) {
        return _mm_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(type lhs, type rhs) {
        return _mm_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(type lhs, type rhs) {
        return _mm_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(type a, type b, type c) {
        return _mm_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(type a, type b, type c) {
        return _mm_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(type a, type b, type c) {
        return _mm_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(type a, type b, type c) {
        return _mm_fnmsub_pd(a, b, c);
    }
};
}    // namespace pcx::simd::detail_
#undef PCX_AINLINE
#endif
