#ifndef PCX_SIMD_TRAITS_HPP
#define PCX_SIMD_TRAITS_HPP

#include <concepts>
#include <immintrin.h>
#include <tuple>

namespace pcx {
using uZ  = std::size_t;
using f32 = float;
using f64 = double;
}    // namespace pcx

// NOLINTBEGIN(*portability*)

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
    using native = __m512;

    PCX_AINLINE static auto set1(f32 value) {
        return _mm512_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm512_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, native vec) {
        _mm512_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm512_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm512_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm512_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm512_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm512_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm512_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm512_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm512_fnmsub_ps(a, b, c);
    }

    // clang-format off
    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {}
    };
    template<>
    struct repack<1, 16> {
        const static inline auto idx0 = _mm512_setr_epi32( 0,16, 1,17, 2,18, 3,19, 4,20, 5,21, 6,22, 7,23);
        const static inline auto idx1 = _mm512_setr_epi32( 8,24, 9,25,10,26,11,27,12,28,13,29,14,30,15,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 16> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1,16,17, 2, 3,18,19, 4, 5,20,21, 6, 7,22,23);
        const static inline auto idx1 = _mm512_setr_epi32(8, 9,24,25,10,11,26,27,12,13,28,29,14,15,30,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<4, 16> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3,16,17,18,19, 4, 5, 6, 7,20,21,22,23);
        const static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,24,25,26,27,12,13,14,15,28,29,30,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<8, 16> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,16,17,18,19,20,21,22,23);
        const static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 8> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<2, 8> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<4, 8> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 8> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<8,16>::permute(a,b);
        }
    };
    template<>
    struct repack<1, 4> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<2, 4> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4,8>::permute(a,b);
        }
    };
    template<>
    struct repack<16, 4> {
        const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 2, 3, 8, 9,10,11,16,17,18,19,24,25,26,27);
        const static inline auto idx1 = _mm512_setr_epi32( 4, 5, 6, 7,12,13,14,15,20,21,22,23,28,29,30,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 2> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2,4>::permute(a,b);
        }
    };
    template<>
    struct repack<8, 2> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 2> {
        const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 4, 5, 8, 9,12,13,16,17,20,21,24,25,28,29);
        const static inline auto idx1 = _mm512_setr_epi32( 2, 3, 6, 7,10,11,14,15,18,19,22,23,26,27,30,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1,2>::permute(a,b);
        }
    };
    template<>
    struct repack<4, 1> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 1> {
        const static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm512_permutexvar_ps(a, idx0);
            b = _mm512_permutexvar_ps(b, idx0);
        }
    };
    template<>
    struct repack<16, 1> {
        const static inline auto idx0 = _mm512_setr_epi32( 0, 2, 4, 6, 8,10,12,14,16,18,20,22,24,26,28,30);
        const static inline auto idx1 = _mm512_setr_epi32( 1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29,31);
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_ps(a, idx0, b);
            auto y = _mm512_permutex2var_ps(a, idx1, b);
            a      = x;
            b      = y;
        }
    };

    using tup16 = std::tuple<native, native, native, native,
                             native, native, native, native,
                             native, native, native, native,
                             native, native, native, native>;

    PCX_AINLINE static auto bit_reverse(tup16 tup) {
        constexpr auto unpck1lo = [](native a, native b) { return _mm512_unpacklo_ps(a, b); };
        constexpr auto unpck1hi = [](native a, native b) { return _mm512_unpackhi_ps(a, b); };
        constexpr auto unpck2lo = [](native a, native b) {
            return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
        };
        constexpr auto unpck2hi = [](native a, native b) {
            return _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
        };
        constexpr auto unpck4lo = [](native a, native b) {
            const auto idx = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck4hi = [](native a, native b) {
            const auto idx = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck8lo = [](native a, native b) {
            const auto idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck8hi = [](native a, native b) {
            const auto idx = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            return _mm512_permutex2var_ps(a, idx, b);
        };

        auto res1 = [unpck1lo, unpck1hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return std::make_tuple(unpck1lo(std::get<Is>(tup), std::get<Is + 8>(tup))...,
                                   unpck1hi(std::get<Is>(tup), std::get<Is + 8>(tup))...);
        }(tup, std::make_index_sequence<8>{});

        auto res2 = [unpck2lo, unpck2hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return std::make_tuple(unpck2lo(std::get<Is>(tup), std::get<Is + 4>(tup))...,
                                   unpck2hi(std::get<Is>(tup), std::get<Is + 4>(tup))...);
        }(res1, std::index_sequence<0, 1, 2, 3, 8, 9, 10, 11>{});

        auto res4 = [unpck4lo, unpck4hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return std::make_tuple(unpck4lo(std::get<Is>(tup), std::get<Is + 2>(tup))...,
                                   unpck4hi(std::get<Is>(tup), std::get<Is + 2>(tup))...);
        }(res2, std::index_sequence<0, 1, 4, 5, 8, 9, 12, 13>{});

        auto res8 = [unpck8lo, unpck8hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return std::make_tuple(unpck8lo(std::get<Is>(tup), std::get<Is + 1>(tup))...,
                                   unpck8hi(std::get<Is>(tup), std::get<Is + 1>(tup))...);
        }(res4, std::index_sequence<0, 2, 4, 6, 8, 10, 12, 14>{});
        auto resort = []<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return std::make_tuple(std::get<Is>(tup)..., std::get<Is + 8>(tup)...);
        }(res8, std::index_sequence<0, 2, 1, 3, 4, 6, 5, 7>{});
        return resort;
    }
    // clang-format on
};
template<>
struct vec_traits<f64, 8> {
    using native = __m512d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm512_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm512_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, native vec) {
        _mm512_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm512_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm512_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm512_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm512_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm512_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm512_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm512_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm512_fnmsub_pd(a, b, c);
    }

    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {}
    };
    template<>
    struct repack<1, 8> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        const static inline auto idx1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);

        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 8> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        const static inline auto idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<4, 8> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        const static inline auto idx1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 4> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<2, 4> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<8, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4, 8>::permute(a, b);
        }
    };
    template<>
    struct repack<1, 2> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 4>::permute(a, b);
        }
    };
    template<>
    struct repack<8, 2> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);
        const static inline auto idx1 = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15);

        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm512_permutex2var_pd(a, idx0, b);
            auto y = _mm512_permutex2var_pd(a, idx1, b);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1, 2>::permute(a, b);
        }
    };
    template<>
    struct repack<4, 1> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm512_permutexvar_pd(a, idx0);
            b = _mm512_permutexvar_pd(b, idx0);
        }
    };
    template<>
    struct repack<8, 1> {
        const static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 13, 14);
        const static inline auto idx1 = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

        PCX_AINLINE static void permute(native& a, native& b) {
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
    using native = __m256;
    PCX_AINLINE static auto set1(f32 value) {
        return _mm256_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, native vec) {
        _mm256_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm256_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm256_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm256_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm256_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm256_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm256_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm256_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm256_fnmsub_ps(a, b, c);
    }

    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {}
    };
    template<>
    struct repack<1, 8> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4, 8>::permute(a, b);
            repack<1, 4>::permute(a, b);
        }
    };
    template<>
    struct repack<2, 8> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4, 8>::permute(a, b);
            repack<2, 4>::permute(a, b);
        }
    };
    template<>
    struct repack<4, 8> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm256_permute2f128_ps(a, b, 0b00100000);
            auto y = _mm256_permute2f128_ps(a, b, 0b00110001);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 4> {
        const static inline auto idx0 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm256_permutevar8x32_ps(a, idx0);
            b = _mm256_permutevar8x32_ps(b, idx0);
        }
    };
    template<>
    struct repack<2, 4> {
        const static inline auto idx0 = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm256_permutevar8x32_ps(a, idx0);
            b = _mm256_permutevar8x32_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm256_permute2f128_ps(a, b, 0b00100000);
            auto y = _mm256_permute2f128_ps(a, b, 0b00110001);
            a      = x;
            b      = y;
        }
    };
    template<>
    struct repack<1, 2> {
        const static inline auto idx0 = _mm256_setr_epi32(0, 2, 1, 3, 4, 6, 5, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm256_permutevar8x32_ps(a, idx0);
            b = _mm256_permutevar8x32_ps(b, idx0);
        }
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm256_permutevar8x32_ps(a, repack<2, 4>::idx0);
            b = _mm256_permutevar8x32_ps(b, repack<2, 4>::idx0);
        }
    };
    template<>
    struct repack<8, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4, 2>::permute(a, b);
            repack<8, 4>::permute(a, b);
        }
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1, 2>::permute(a, b);
        }
    };
    template<>
    struct repack<4, 1> {
        const static inline auto idx0 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        PCX_AINLINE static void  permute(native& a, native& b) {
            a = _mm256_permutevar8x32_ps(a, idx0);
            b = _mm256_permutevar8x32_ps(b, idx0);
        }
    };
    template<>
    struct repack<8, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<4, 1>::permute(a, b);
            repack<8, 4>::permute(a, b);
        }
    };
};
template<>
struct vec_traits<f32, 4> {
    using native = __m128;
    PCX_AINLINE static auto set1(f32 value) {
        return _mm_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, native vec) {
        _mm_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm_fnmsub_ps(a, b, c);
    }

    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {};
    };
    template<>
    struct repack<1, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm_unpacklo_ps(a, b);
            auto y = _mm_unpackhi_ps(a, b);
            a      = x;
            b      = y;
        };
    };
    template<>
    struct repack<2, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm_unpacklo_pd(_mm_castps_pd(a), _mm_castps_pd(b));
            auto y = _mm_unpackhi_pd(_mm_castps_pd(a), _mm_castps_pd(b));
            a      = _mm_castps_pd(x);
            b      = _mm_castps_pd(y);
        };
    };
    template<>
    struct repack<1, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            a = _mm_permute_ps(a, 0b11011000);
            b = _mm_permute_ps(b, 0b11011000);
        };
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 4>::permute(a, b);
        };
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1, 2>::permute(a, b);
        };
    };
    template<>
    struct repack<4, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 1>::permute(a, b);
            repack<4, 2>::permute(a, b);
        };
    };
};

template<>
struct vec_traits<f64, 4> {
    using native = __m256d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm256_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm256_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, native vec) {
        _mm256_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm256_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm256_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm256_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm256_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm256_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm256_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm256_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm256_fnmsub_pd(a, b, c);
    }
    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {};
    };
    template<>
    struct repack<1, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 4>::permute(a, b);
            repack<1, 2>::permute(a, b);
        };
    };
    template<>
    struct repack<2, 4> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm256_permute2f128_pd(a, b, 0b00100000);
            auto y = _mm256_permute2f128_pd(a, b, 0b00110001);
            a      = x;
            b      = y;
        };
    };
    template<>
    struct repack<1, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm256_permute4x64_pd(a, 0b11011000);
            auto y = _mm256_permute4x64_pd(b, 0b11011000);
            a      = x;
            b      = y;
        };
    };
    template<>
    struct repack<4, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 4>::permute(a, b);
        };
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1, 2>::permute(a, b);
        };
    };
    template<>
    struct repack<4, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<2, 1>::permute(a, b);
            repack<4, 2>::permute(a, b);
        };
    };
};
template<>
struct vec_traits<f64, 2> {
    using native = __m128d;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, native vec) {
        _mm_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(native lhs, native rhs) {
        return _mm_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(native lhs, native rhs) {
        return _mm_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(native lhs, native rhs) {
        return _mm_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(native lhs, native rhs) {
        return _mm_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(native a, native b, native c) {
        return _mm_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(native a, native b, native c) {
        return _mm_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(native a, native b, native c) {
        return _mm_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(native a, native b, native c) {
        return _mm_fnmsub_pd(a, b, c);
    }
    template<uZ To, uZ From>
    struct repack;
    template<uZ P>
    struct repack<P, P> {
        PCX_AINLINE static void permute(native& a, native& b) {};
    };
    template<>
    struct repack<1, 2> {
        PCX_AINLINE static void permute(native& a, native& b) {
            auto x = _mm_unpacklo_pd(a, b);
            auto y = _mm_unpackhi_pd(a, b);
            a      = x;
            b      = y;
        };
    };
    template<>
    struct repack<2, 1> {
        PCX_AINLINE static void permute(native& a, native& b) {
            repack<1, 2>::permute(a, b);
        };
    };
};
}    // namespace pcx::simd::detail_

// NOLINTEND(*portability*)

#undef PCX_AINLINE
#endif
