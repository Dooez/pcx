#pragma once
#include "pcx/include/tuple.hpp"

#include <immintrin.h>

namespace pcx {
using uZ  = std::size_t;
using f64 = double;
}    // namespace pcx

// NOLINTBEGIN(*portability*, *magic-number*)
namespace pcx::simd::detail_ {

template<>
struct vec_traits<f64, 2> {
    using impl_vec = __m128d;
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
    template<uZ To, uZ From>
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        };
    };
    template<uZ To, uZ From>
    static constexpr auto repack = repack_t<To, From>{};

    using tup2 = tupi::tuple<impl_vec, impl_vec>;
    PCX_AINLINE static auto bit_reverse(tup2 tup) noexcept {
        return tupi::make_tuple(_mm_unpacklo_pd(tupi::get<0>(tup), tupi::get<1>(tup)),
                                _mm_unpackhi_pd(tupi::get<0>(tup), tupi::get<1>(tup)));
    }
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
struct vec_traits<f64, 4> {
    using impl_vec = __m256d;
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
    template<uZ To, uZ From>
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        };
    };
    template<uZ To, uZ From>
    static constexpr auto repack = repack_t<To, From>{};

    using tup4 = tupi::broadcast_tuple_t<impl_vec, 4>;
    PCX_AINLINE static auto bit_reverse(tup4 tup) noexcept {
        constexpr auto unpck1lo = [](impl_vec a, impl_vec b) noexcept { return _mm256_unpacklo_pd(a, b); };
        constexpr auto unpck1hi = [](impl_vec a, impl_vec b) noexcept { return _mm256_unpackhi_pd(a, b); };
        constexpr auto unpck2lo = [](impl_vec a, impl_vec b) {
            return _mm256_permute2f128_pd(a, b, 0b00100000);
        };
        constexpr auto unpck2hi = [](impl_vec a, impl_vec b) {
            return _mm256_permute2f128_pd(a, b, 0b00110001);
        };

        auto res1 = [unpck1lo, unpck1hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck1lo(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...,
                                    unpck1hi(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...);
        }(tup, std::make_index_sequence<2>{});

        auto res2 = [unpck2lo, unpck2hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck2lo(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...,
                                    unpck2hi(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...);
        }(res1, std::index_sequence<0, 2>{});
        return res2;
    }
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
struct vec_traits<f64, 4>::repack_t<1, 4> : tupi::compound_op_base {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto [x, y] = repack_t<2, 4>{}(a, b);
        return repack_t<1, 2>{}(x, y);
    };
    template<uZ I>
    PCX_AINLINE constexpr friend auto get_stage(const repack_t&) {
        return stage_t<I>{};
    }
    template<uZ I>
    struct stage_t {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const
            requires(I == 0)
        {
            auto [x, y] = repack_t<2, 4>{}(a, b);
            return tupi::make_intermediate(x, y);
        }
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const
            requires(I == 1)
        {
            return repack_t<1, 2>{}(a, b);
        }
    };
};
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
struct vec_traits<f64, 4>::repack_t<4, 1> : tupi::compound_op_base {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto [x, y] = repack_t<2, 1>{}(a, b);
        return repack_t<4, 2>{}(x, y);
    };
    template<uZ I>
    PCX_AINLINE constexpr friend auto get_stage(const repack_t&) {
        return stage_t<I>{};
    }
    template<uZ I>
    struct stage_t {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const
            requires(I == 0)
        {
            auto [x, y] = repack_t<2, 1>{}(a, b);
            return tupi::make_intermediate(x, y);
        }
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const
            requires(I == 1)
        {
            return repack_t<4, 2>{}(a, b);
        }
    };
};

#ifdef PCX_AVX512
template<>
struct max_vec_width<f64> {
    static constexpr uZ value = 8;
};
template<>
struct vec_traits<f64, 8> {
    using impl_vec = __m512d;
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

    template<uZ To, uZ From>
    struct repack_t;
    template<uZ P>
    struct repack_t<P, P> {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            return tupi::make_tuple(a, b);
        }
    };
    template<uZ To, uZ From>
    static constexpr auto repack = repack_t<To, From>{};

    using tup8 = tupi::broadcast_tuple_t<impl_vec, 8>;
    PCX_AINLINE static auto bit_reverse(tup8 tup) noexcept {
        constexpr auto unpck1lo = [](impl_vec a, impl_vec b) noexcept { return _mm512_unpacklo_pd(a, b); };
        constexpr auto unpck1hi = [](impl_vec a, impl_vec b) noexcept { return _mm512_unpackhi_pd(a, b); };
        constexpr auto unpck2lo = [](impl_vec a, impl_vec b) noexcept {
            const auto idx = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
            return _mm512_permutex2var_pd(a, idx, b);
        };
        constexpr auto unpck2hi = [](impl_vec a, impl_vec b) noexcept {
            const auto idx = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 12, 13);
            return _mm512_permutex2var_pd(a, idx, b);
        };
        constexpr auto unpck4lo = [](impl_vec a, impl_vec b) noexcept {
            const auto idx = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
            return _mm512_permutex2var_pd(a, idx, b);
        };
        constexpr auto unpck4hi = [](impl_vec a, impl_vec b) noexcept {
            const auto idx = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
            return _mm512_permutex2var_pd(a, idx, b);
        };

        auto res1 = [unpck1lo, unpck1hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck1lo(tupi::get<Is>(tup), tupi::get<Is + 4>(tup))...,
                                    unpck1hi(tupi::get<Is>(tup), tupi::get<Is + 4>(tup))...);
        }(tup, std::make_index_sequence<4>{});

        auto res2 = [unpck2lo, unpck2hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck2lo(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...,
                                    unpck2hi(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...);
        }(res1, std::index_sequence<0, 1, 4, 5>{});

        auto res4 = [unpck4lo, unpck4hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck4lo(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...,
                                    unpck4hi(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...);
        }(res2, std::index_sequence<0, 2, 4, 6>{});
        return res4;
    }
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
        auto repack_t<1, 2>{}(a, b);
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
    const static inline auto idx0 = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 13, 14);
    const static inline auto idx1 = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_pd(a, idx0, b);
        auto y = _mm512_permutex2var_pd(a, idx1, b);
        return tupi::make_tuple(x, y);
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
