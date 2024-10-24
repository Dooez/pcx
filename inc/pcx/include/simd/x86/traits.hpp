#ifndef PCX_SIMD_TRAITS_HPP
#define PCX_SIMD_TRAITS_HPP

#include <immintrin.h>
#include <utility>
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
    // clang-format off
    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void perm(type& a, type& b) {}
    };
    template<>
    struct repack<1, 16> {
        static inline auto idx0 = _mm512_setr_epi32( 0,16, 1,17, 2,18, 3,19, 4,20, 5,21, 6,22, 7,23);
        static inline auto idx1 = _mm512_setr_epi32( 8,24, 9,25,10,26,11,27,12,28,13,29,14,30,15,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 16> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1,16,17, 2, 3,18,19, 4, 5,20,21, 6, 7,22,23);
        static inline auto idx1 = _mm512_setr_epi32(8, 9,24,25,10,11,26,27,12,13,28,29,14,15,30,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<4, 16> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3,16,17,18,19, 4, 5, 6, 7,20,21,22,23);
        static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,24,25,26,27,12,13,14,15,28,29,30,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<8, 16> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,16,17,18,19,20,21,22,23);
        static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 8> {
        static inline auto idx0 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<2, 8> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<4, 8> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 8> {
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, repack<8, 16>::idx0, b);
            auto y = _mm512_permutex2var_ps(a, repack<8, 16>::idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 4> {
        static inline auto idx0 = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<2, 4> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 4> {
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, repack<4, 8>::idx0);
            b = _mm512_permutexvar_ps(b, repack<4, 8>::idx0);
        }
    };
    template<>
    struct repack<16, 4> {
        static inline auto idx0 = _mm512_setr_epi32( 0, 1, 2, 3, 8, 9,10,11,16,17,18,19,24,25,26,27);
        static inline auto idx1 = _mm512_setr_epi32( 4, 5, 6, 7,12,13,14,15,20,21,22,23,28,29,30,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 2> {
        static inline auto idx0 = _mm512_setr_epi32(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, repack<2, 4>::idx0);
            b = _mm512_permutexvar_ps(b, repack<2, 4>::idx0);
        }
    };
    template<>
    struct repack<8, 2> {
        static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 2> {
        static inline auto idx0 = _mm512_setr_epi32( 0, 1, 4, 5, 8, 9,12,13,16,17,20,21,24,25,28,29);
        static inline auto idx1 = _mm512_setr_epi32( 2, 3, 6, 7,10,11,14,15,18,19,22,23,26,27,30,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, repack<1, 2>::idx0);
            b = _mm512_permutexvar_ps(b, repack<1, 2>::idx0);
        }
    };
    template<>
    struct repack<4, 1> {
        static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 1> {
        static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 1> {
        static inline auto idx0 = _mm512_setr_epi32( 0, 2, 4, 6, 8,10,12,14,16,18,20,22,24,26,28,30);
        static inline auto idx1 = _mm512_setr_epi32( 1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29,31);
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    // clang-format on
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
    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void perm(type& a, type& b) {}
    };
    template<>
    struct repack<1, 8> {
        static inline auto idx0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        static inline auto idx1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);

        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 8> {
        static inline auto idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        static inline auto idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<4, 8> {
        static inline auto idx0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        static inline auto idx1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 4> {
        static inline auto      idx0 = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<2, 4> {
        static inline auto      idx0 = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<8, 4> {
        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, repack<4, 8>::idx0, b);
            auto y = _mm512_permutex2var_pd(a, repack<4, 8>::idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 2> {
        static inline auto      idx0 = _mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, repack<2, 4>::idx0);
            b = _mm512_permutexvar_pd(b, repack<2, 4>::idx0);
        }
    };
    template<>
    struct repack<8, 2> {
        static inline auto idx0 = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);
        static inline auto idx1 = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15);

        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, repack<1, 2>::idx0);
            b = _mm512_permutexvar_pd(b, repack<1, 2>::idx0);
        }
    };
    template<>
    struct repack<4, 1> {
        static inline auto      idx0 = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        PCX_AINLINE static void perm(type& a, type& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<8, 1> {
        static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 13, 14);
        static inline auto idx1 = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

        PCX_AINLINE static void perm(type& a, type& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
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
