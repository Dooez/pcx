#pragma once
#include "pcx/include/tupi.hpp"

#include <immintrin.h>

namespace pcx {
using uZ  = std::size_t;
using f64 = double;
}    // namespace pcx

// NOLINTBEGIN(*portability*, *magic-number*)
namespace pcx::simd::detail_ {

template<>
struct vec_traits<f64, 1> {
    using impl_vec            = f64;
    static constexpr uZ width = 1;

    PCX_AINLINE static auto set1(f64 value) -> impl_vec {
        return value;
    }
    PCX_AINLINE static auto zero() -> impl_vec {
        return 0;
    }
    PCX_AINLINE static auto load(const f64* src) -> impl_vec {
        return *src;
    }
    PCX_AINLINE static void store(f64* dest, impl_vec vec) {
        *dest = vec;
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return lhs + rhs;
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return lhs - rhs;
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return lhs * rhs;
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return lhs / rhs;
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;

    constexpr static struct {
        static auto operator()(impl_vec v) -> impl_vec {
            return v;
        };
    } upsample{};

    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    struct split_interleave_t {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        }
    };
    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    static constexpr auto split_interleave = split_interleave_t<ChunkSize>{};

    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        };
    };
    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    static constexpr auto repack = repack_t<To, From>{};
};
template<>
struct vec_traits<f64, 2> {
    using impl_vec            = __m128d;
    static constexpr uZ width = 2;

    PCX_AINLINE static auto set1(f64 value) {
        return _mm_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, impl_vec vec) {
        _mm_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fnmsub_pd(a, b, c);
    }

    constexpr static struct {
        PCX_AINLINE static auto operator()(vec_traits<f64, 1>::impl_vec x) -> impl_vec {
            return set1(x);
        }
        PCX_AINLINE static auto operator()(impl_vec v) -> impl_vec {
            return v;
        };
    } upsample{};

    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    struct split_interleave_t;
    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    static constexpr auto split_interleave = split_interleave_t<ChunkSize>{};

    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        };
    };
    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    static constexpr auto repack = repack_t<To, From>{};
};
template<>
struct vec_traits<f64, 2>::repack_t<1, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm_unpacklo_pd(a, b);
        auto y = _mm_unpackhi_pd(a, b);
        return tupi::make_tuple(x, y);
    };
};
template<>
struct vec_traits<f64, 2>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    };
};
template<>
struct vec_traits<f64, 2>::split_interleave_t<1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm_castps_pd(_mm_movelh_ps(_mm_castpd_ps(a), _mm_castpd_ps(b)));
        auto y = _mm_unpackhi_pd(a, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 2>::split_interleave_t<2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};
PCX_AINLINE auto vec_traits<f64, 1>::fmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
    using proxy_traits = vec_traits<f64, 2>;

    auto x   = proxy_traits::upsample(a);
    auto y   = proxy_traits::upsample(b);
    auto z   = proxy_traits::upsample(c);
    auto res = proxy_traits::fmadd(x, y, z);
    return res[0];
}
PCX_AINLINE auto vec_traits<f64, 1>::fnmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
    using proxy_traits = vec_traits<f64, 2>;

    auto x   = proxy_traits::upsample(a);
    auto y   = proxy_traits::upsample(b);
    auto z   = proxy_traits::upsample(c);
    auto res = proxy_traits::fnmadd(x, y, z);
    return res[0];
}
PCX_AINLINE auto vec_traits<f64, 1>::fmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
    using proxy_traits = vec_traits<f64, 2>;

    auto x   = proxy_traits::upsample(a);
    auto y   = proxy_traits::upsample(b);
    auto z   = proxy_traits::upsample(c);
    auto res = proxy_traits::fmsub(x, y, z);
    return res[0];
}
PCX_AINLINE auto vec_traits<f64, 1>::fnmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
    using proxy_traits = vec_traits<f64, 2>;

    auto x   = proxy_traits::upsample(a);
    auto y   = proxy_traits::upsample(b);
    auto z   = proxy_traits::upsample(c);
    auto res = proxy_traits::fnmsub(x, y, z);
    return res[0];
}

template<>
struct vec_traits<f64, 4> {
    using impl_vec            = __m256d;
    static constexpr uZ width = 4;

    PCX_AINLINE static auto set1(f64 value) {
        return _mm256_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm256_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, impl_vec vec) {
        _mm256_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm256_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm256_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm256_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm256_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fnmsub_pd(a, b, c);
    }

    static constexpr struct {
        PCX_AINLINE static auto operator()(vec_traits<f64, 1>::impl_vec x) -> impl_vec {
            return set1(x);
        }
        PCX_AINLINE auto operator()(vec_traits<f64, 2>::impl_vec vec) const {
            auto a = _mm256_zextpd128_pd256(vec);
            return _mm256_permute4x64_pd(a, 0b01010000);
        }
        PCX_AINLINE auto operator()(impl_vec v) const -> impl_vec {
            return v;
        }
    } upsample{};

    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    struct split_interleave_t;
    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    static constexpr auto split_interleave = split_interleave_t<ChunkSize>{};

    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        };
    };
    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    static constexpr auto repack = repack_t<To, From>{};
};
template<>
struct vec_traits<f64, 4>::repack_t<2, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute2f128_pd(a, b, 0b00100000);
        auto y = _mm256_permute2f128_pd(a, b, 0b00110001);
        return tupi::make_tuple(x, y);
    };
};
template<>
struct vec_traits<f64, 4>::repack_t<1, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute4x64_pd(a, 0b11011000);
        auto y = _mm256_permute4x64_pd(b, 0b11011000);
        return tupi::make_tuple(x, y);
    };
};
template<>
struct vec_traits<f64, 4>::repack_t<1, 4>
: public decltype(tupi::pass | repack_t<2, 4>{} | tupi::apply | repack_t<1, 2>{}) {};
template<>
struct vec_traits<f64, 4>::repack_t<4, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<2, 4>{}(a, b);
    };
};
template<>
struct vec_traits<f64, 4>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    };
};
template<>
struct vec_traits<f64, 4>::repack_t<4, 1>
: public decltype(tupi::pass | repack_t<2, 1>{} | tupi::apply | repack_t<4, 2>{}) {};
template<>
struct vec_traits<f64, 4>::split_interleave_t<1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_unpacklo_pd(a, b);
        auto y = _mm256_unpackhi_pd(a, b);
        return tupi::make_tuple(x, y);
    }
};

template<>
struct vec_traits<f64, 4>::split_interleave_t<2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute2f128_pd(a, b, 0b00100000);
        auto y = _mm256_permute2f128_pd(a, b, 0b00110001);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 4>::split_interleave_t<4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};

#ifdef PCX_AVX512
template<>
struct max_vec_width<f64> {
    static constexpr uZ value = 8;
};
template<>
struct vec_traits<f64, 8> {
    using impl_vec                = __m512d;
    static constexpr uZ     width = 8;
    PCX_AINLINE static auto set1(f64 value) {
        return _mm512_set1_pd(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_pd();
    }
    PCX_AINLINE static auto load(const f64* src) {
        return _mm512_loadu_pd(src);
    }
    PCX_AINLINE static void store(f64* dest, impl_vec vec) {
        _mm512_storeu_pd(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm512_add_pd(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm512_sub_pd(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm512_mul_pd(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm512_div_pd(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fnmadd_pd(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fmsub_pd(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fnmsub_pd(a, b, c);
    }

    static constexpr struct upsample_t {
        inline static const auto idx2 = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
        inline static const auto idx4 = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);

        PCX_AINLINE static auto operator()(vec_traits<f64, 1>::impl_vec x) -> impl_vec {
            return set1(x);
        }
        PCX_AINLINE static auto operator()(vec_traits<f64, 2>::impl_vec vec) -> impl_vec {
            auto ve = _mm512_zextpd128_pd512(vec);
            return _mm512_permutexvar_pd(idx2, ve);
        }
        PCX_AINLINE static auto operator()(vec_traits<f64, 4>::impl_vec vec) -> impl_vec {
            auto ve = _mm512_zextpd256_pd512(vec);
            return _mm512_permutexvar_pd(idx4, ve);
        }
        PCX_AINLINE static auto operator()(impl_vec v) -> impl_vec {
            return v;
        }
    } upsample{};

    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    struct split_interleave_t;
    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    static constexpr auto split_interleave = split_interleave_t<ChunkSize>{};

    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        }
    };
    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    static constexpr auto repack = repack_t<To, From>{};
};
template<>
struct vec_traits<f64, 8>::repack_t<1, 8> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const static inline auto idx1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<2, 8> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const static inline auto idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<4, 8> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    const static inline auto idx1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<1, 4> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_pd(idx0, a);
        b = _mm512_permutexvar_pd(idx0, b);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<2, 4> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_pd(idx0, a);
        b = _mm512_permutexvar_pd(idx0, b);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<8, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<4, 8>{}(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<1, 2> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_pd(idx0, a);
        b = _mm512_permutexvar_pd(idx0, b);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<4, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        repack_t<2, 4>{}(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<8, 2> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);
    const static inline auto idx1 = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<4, 1> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_pd(idx0, a);
        b = _mm512_permutexvar_pd(idx0, b);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f64, 8>::repack_t<8, 1> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const static inline auto idx1 = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::split_interleave_t<1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_unpacklo_pd(a, b);
        auto y = _mm512_unpackhi_pd(a, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::split_interleave_t<2> {
    const static inline auto idx0 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const static inline auto idx1 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    PCX_AINLINE auto         operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::split_interleave_t<4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto b_low = _mm512_extractf64x4_pd(b, 0);
        auto x     = _mm512_insertf64x4(a, b_low, 0b1);
        auto y     = _mm512_shuffle_f64x2(a, b, 0b11101110);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f64, 8>::split_interleave_t<8> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};
#else
template<>
struct max_vec_width<f64> {
    static constexpr uZ value = 4;
};
#endif

}    // namespace pcx::simd::detail_
// NOLINTEND(*portability*, *magic-number*)
