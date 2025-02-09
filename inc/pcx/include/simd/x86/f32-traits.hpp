#pragma once
#include "pcx/include/tupi.hpp"

#include <immintrin.h>

namespace pcx {
using uZ  = std::size_t;
using f32 = float;
}    // namespace pcx

// NOLINTBEGIN(*portability*, *magic-number*)

namespace pcx::simd::detail_ {

template<typename T>
struct max_vec_width;
template<typename T, uZ Width>
struct vec_traits;

template<>
struct vec_traits<f32, 2> {
    using impl_vec            = std::array<f32, 2>;
    static constexpr uZ width = 2;

    PCX_AINLINE static auto set1(f32 value) -> impl_vec {
        return {value, value};
    }
    PCX_AINLINE static auto zero() -> impl_vec {
        return {0, 0};
    }
    PCX_AINLINE static auto load(const f32* src) -> impl_vec {
        return {src[0], src[1]};
    }
    PCX_AINLINE static void store(f32* dest, impl_vec vec) {
        dest[0] = vec[0];
        dest[1] = vec[1];
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return {lhs[0] + rhs[0], lhs[1] + rhs[1]};
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return {lhs[0] - rhs[0], lhs[1] - rhs[1]};
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return {lhs[0] * rhs[0], lhs[1] * rhs[1]};
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) -> impl_vec {
        return {lhs[0] / rhs[0], lhs[1] / rhs[1]};
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
        return {a[0] * b[0] + c[0], a[1] * b[1] + c[1]};
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
        return {-a[0] * b[0] + c[0], -a[1] * b[1] + c[1]};
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
        return {a[0] * b[0] - c[0], a[1] * b[1] - c[1]};
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec {
        return {-a[0] * b[0] - c[0], -a[1] * b[1] - c[1]};
    }

    constexpr static struct {
        static auto operator()(impl_vec v) -> impl_vec {
            return v;
        };
    } upsample;

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

    using tup_width = tupi::broadcast_tuple_t<impl_vec, width>;
    PCX_AINLINE static auto bit_reverse(tup_width tup) {
        auto first  = tupi::get<0>(tup);
        auto second = tupi::get<1>(tup);
        return tup_width{
            {first[0], second[0]},
            {first[1], second[1]},
        };
    }
};
template<>
struct vec_traits<f32, 2>::repack_t<1, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = impl_vec{a[0], b[0]};
        auto y = impl_vec{a[1], b[1]};
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 2>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    };
};
template<>
struct vec_traits<f32, 2>::split_interleave_t<1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = impl_vec{a[0], b[0]};
        auto y = impl_vec{a[1], b[1]};
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 2>::split_interleave_t<2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f32, 4> {
    using impl_vec                = __m128;
    static constexpr uZ     width = 4;
    PCX_AINLINE static auto set1(f32 value) {
        return _mm_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, impl_vec vec) {
        _mm_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm_fnmsub_ps(a, b, c);
    }

    static constexpr struct {
        PCX_AINLINE auto operator()(vec_traits<f32, 2>::impl_vec vec) const {
            auto a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<f64*>(vec.data())));
            return _mm_unpacklo_ps(a, a);
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

    struct sil1_helper_0 {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            auto x = _mm_shuffle_ps(a, b, 0b10001000);
            auto y = _mm_shuffle_ps(a, b, 0b11011101);
            return tupi::make_tuple(x, y);
        }
    };
    struct sil1_helper_1 {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            auto x = _mm_shuffle_ps(a, a, 0b11011000);
            auto y = _mm_shuffle_ps(b, b, 0b11011000);
            return tupi::make_tuple(x, y);
        }
    };

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

    using tup_width = tupi::broadcast_tuple_t<impl_vec, width>;
    PCX_AINLINE static auto bit_reverse(tup_width tup) noexcept {
        constexpr auto unpck1lo = [](impl_vec a, impl_vec b) noexcept { return _mm_unpacklo_ps(a, b); };
        constexpr auto unpck1hi = [](impl_vec a, impl_vec b) noexcept { return _mm_unpackhi_ps(a, b); };
        constexpr auto unpck2lo = [](impl_vec a, impl_vec b) {
            return _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(a), _mm_castps_pd(b)));
        };
        constexpr auto unpck2hi = [](impl_vec a, impl_vec b) {
            return _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(a), _mm_castps_pd(b)));
        };
        auto res1 = [unpck1lo, unpck1hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck1lo(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...,
                                    unpck1hi(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...);
        }(tup, std::make_index_sequence<2>{});

        auto res2 = [unpck2lo, unpck2hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(unpck2lo(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...,
                                    unpck2hi(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...);
        }(res1, std::index_sequence<0, 2>{});

        auto resort = []<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(tupi::get<Is>(tup)..., tupi::get<Is + 1>(tup)...);
        }(res2, std::index_sequence<0, 2>{});
        return resort;
    }
};

template<>
struct vec_traits<f32, 4>::split_interleave_t<1>
: public decltype(tupi::pass           //
                  | sil1_helper_0{}    //
                  | tupi::apply        //
                  | sil1_helper_1{}) {};
template<>
struct vec_traits<f32, 4>::split_interleave_t<2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm_unpacklo_pd(a, b);
        auto y = _mm_unpackhi_pd(a, b);
        return tupi::make_tuple(x, y);
    };
};
template<>
struct vec_traits<f32, 4>::split_interleave_t<4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    };
};
template<>
struct vec_traits<f32, 4>::repack_t<1, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm_unpacklo_ps(a, b);
        auto y = _mm_unpackhi_ps(a, b);
        return tupi::make_tuple(x, y);
    };
};
template<>
struct vec_traits<f32, 4>::repack_t<2, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm_unpacklo_pd(_mm_castps_pd(a), _mm_castps_pd(b));
        auto y = _mm_unpackhi_pd(_mm_castps_pd(a), _mm_castps_pd(b));
        return tupi::make_tuple(_mm_castpd_ps(x), _mm_castpd_ps(y));
    };
};
template<>
struct vec_traits<f32, 4>::repack_t<1, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm_permute_ps(a, 0b11011000);
        b = _mm_permute_ps(b, 0b11011000);
        return tupi::make_tuple(a, b);
    };
};
template<>
struct vec_traits<f32, 4>::repack_t<4, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<2, 4>{}(a, b);
    };
};
template<>
struct vec_traits<f32, 4>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    };
};

template<>
struct vec_traits<f32, 4>::repack_t<4, 1>
: public decltype(tupi::pass            //
                  | repack_t<2, 1>{}    //
                  | tupi::apply         //
                  | repack_t<4, 2>{}) {};

template<>
struct vec_traits<f32, 8> {
    using impl_vec            = __m256;
    static constexpr uZ width = 8;

    PCX_AINLINE static auto set1(f32 value) {
        return _mm256_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm256_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm256_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, impl_vec vec) {
        _mm256_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm256_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm256_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm256_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm256_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm256_fnmsub_ps(a, b, c);
    }

    static constexpr struct up_stage0_t {
        static inline const auto idx4 = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
        PCX_AINLINE auto         operator()(vec_traits<f32, 2>::impl_vec vec) const {
            auto a = _mm256_set1_ps(vec[0]);
            auto b = _mm_set1_ps(vec[1]);
            return tupi::make_tuple(a, b, uZ_ce<2>{});
        }
        PCX_AINLINE auto operator()(vec_traits<f32, 4>::impl_vec vec) const {
            auto x = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(vec), idx4);
            return tupi::make_tuple(x);
        }
        PCX_AINLINE auto operator()(impl_vec v) const {
            return tupi::make_tuple(v);
        }
    } up_stage0{};
    static constexpr struct {
        PCX_AINLINE auto operator()(impl_vec v) const {
            return v;
        }
        PCX_AINLINE auto operator()(auto a, auto b, uZ_ce<2>) const {
            return _mm256_insertf128_ps(a, b, 0b1);
        }
    } up_stage1{};
    static constexpr auto upsample = tupi::pass | up_stage0 | tupi::apply | up_stage1;

    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    struct split_interleave_t;
    template<uZ ChunkSize>
        requires(ChunkSize <= width)
    static constexpr auto split_interleave = split_interleave_t<ChunkSize>{};

    struct sil1_helper_0 {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            auto x = _mm256_shuffle_ps(a, b, 0b10001000);
            auto y = _mm256_shuffle_ps(a, b, 0b11011101);
            return tupi::make_tuple(x, y);
        }
    };
    struct sil1_helper_1 {
        PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
            auto x = _mm256_shuffle_ps(a, a, 0b11011000);
            auto y = _mm256_shuffle_ps(b, b, 0b11011000);
            return tupi::make_tuple(x, y);
        }
    };

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

    using tup_width = tupi::broadcast_tuple_t<impl_vec, width>;
    PCX_AINLINE static auto bit_reverse(tup_width tup) noexcept {
        constexpr auto unpck1lo = [](impl_vec a, impl_vec b) noexcept { return _mm256_unpacklo_ps(a, b); };
        constexpr auto unpck1hi = [](impl_vec a, impl_vec b) noexcept { return _mm256_unpackhi_ps(a, b); };
        constexpr auto unpck2lo = [](impl_vec a, impl_vec b) {
            return _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
        };
        constexpr auto unpck2hi = [](impl_vec a, impl_vec b) {
            return _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
        };
        constexpr auto unpck4lo = [](impl_vec a, impl_vec b) {
            return _mm256_permute2f128_ps(a, b, 0b00100000);
        };
        constexpr auto unpck4hi = [](impl_vec a, impl_vec b) {
            return _mm256_permute2f128_ps(a, b, 0b00110001);
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

        auto resort = []<uZ... Is>(auto tup, std::index_sequence<Is...>) noexcept {
            return tupi::make_tuple(tupi::get<Is>(tup)..., tupi::get<Is + 4>(tup)...);
        }(res4, std::index_sequence<0, 2, 1, 3>{});
        return resort;
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<4, 8> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute2f128_ps(a, b, 0b00100000);
        auto y = _mm256_permute2f128_ps(a, b, 0b00110001);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<1, 4> {
    const static inline auto idx0 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    PCX_AINLINE auto         operator()(impl_vec a, impl_vec b) const {
        a = _mm256_permutevar8x32_ps(a, idx0);
        b = _mm256_permutevar8x32_ps(b, idx0);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<2, 4> {
    const static inline auto idx0 = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    PCX_AINLINE auto         operator()(impl_vec a, impl_vec b) const {
        a = _mm256_permutevar8x32_ps(a, idx0);
        b = _mm256_permutevar8x32_ps(b, idx0);
        return tupi::make_tuple(a, b);
    }
};

template<>
struct vec_traits<f32, 8>::repack_t<1, 8>
: public decltype(tupi::pass                                //
                  | vec_traits<f32, 8>::repack_t<4, 8>{}    //
                  | tupi::apply                             //
                  | vec_traits<f32, 8>::repack_t<1, 4>{}) {};
template<>
struct vec_traits<f32, 8>::repack_t<2, 8>
: public decltype(tupi::pass                                //
                  | vec_traits<f32, 8>::repack_t<4, 8>{}    //
                  | tupi::apply                             //
                  | vec_traits<f32, 8>::repack_t<2, 4>{}) {};
template<>
struct vec_traits<f32, 8>::repack_t<8, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute2f128_ps(a, b, 0b00100000);
        auto y = _mm256_permute2f128_ps(a, b, 0b00110001);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<1, 2> {
    const static inline auto idx0 = _mm256_setr_epi32(0, 2, 1, 3, 4, 6, 5, 7);
    PCX_AINLINE auto         operator()(impl_vec a, impl_vec b) const {
        a = _mm256_permutevar8x32_ps(a, idx0);
        b = _mm256_permutevar8x32_ps(b, idx0);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<4, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm256_permutevar8x32_ps(a, vec_traits<f32, 8>::repack_t<2, 4>::idx0);
        b = _mm256_permutevar8x32_ps(b, vec_traits<f32, 8>::repack_t<2, 4>::idx0);
        return tupi::make_tuple(a, b);
    }
};

template<>
struct vec_traits<f32, 8>::repack_t<8, 2>
: public decltype(tupi::pass                                //
                  | vec_traits<f32, 8>::repack_t<8, 4>{}    //
                  | tupi::apply                             //
                  | vec_traits<f32, 8>::repack_t<4, 2>{}) {};
template<>
struct vec_traits<f32, 8>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1, 2>{}(a, b);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<4, 1> {
    const static inline auto idx0 = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    PCX_AINLINE auto         operator()(impl_vec a, impl_vec b) const {
        a = _mm256_permutevar8x32_ps(a, idx0);
        b = _mm256_permutevar8x32_ps(b, idx0);
        return tupi::make_tuple(a, b);
    }
};
template<>
struct vec_traits<f32, 8>::repack_t<8, 1>
: public decltype(tupi::pass                                //
                  | vec_traits<f32, 8>::repack_t<4, 1>{}    //
                  | tupi::apply                             //
                  | vec_traits<f32, 8>::repack_t<8, 4>{}) {};

template<>
struct vec_traits<f32, 8>::split_interleave_t<1>
: public decltype(tupi::pass           //
                  | sil1_helper_0{}    //
                  | tupi::apply        //
                  | sil1_helper_1{}) {};
template<>
struct vec_traits<f32, 8>::split_interleave_t<2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto ad = _mm256_castps_pd(a);
        auto bd = _mm256_castps_pd(b);
        auto x  = _mm256_castpd_ps(_mm256_unpacklo_pd(ad, bd));
        auto y  = _mm256_castpd_ps(_mm256_unpackhi_pd(ad, bd));
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 8>::split_interleave_t<4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm256_permute2f128_ps(a, b, 0b00100000);
        auto y = _mm256_permute2f128_ps(a, b, 0b00110001);
        return tupi::make_tuple(x, y);
    }
};
template<>
struct vec_traits<f32, 8>::split_interleave_t<8> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};

#ifdef PCX_AVX512
template<>
struct max_vec_width<f32> {
    static constexpr uZ value = 16;
};

template<>
struct vec_traits<f32, 16> {
    using impl_vec            = __m512;
    static constexpr uZ width = 16;

    PCX_AINLINE static auto set1(f32 value) {
        return _mm512_set1_ps(value);
    }
    PCX_AINLINE static auto zero() {
        return _mm512_setzero_ps();
    }
    PCX_AINLINE static auto load(const f32* src) {
        return _mm512_loadu_ps(src);
    }
    PCX_AINLINE static void store(f32* dest, impl_vec vec) {
        _mm512_storeu_ps(dest, vec);
    }
    PCX_AINLINE static auto add(impl_vec lhs, impl_vec rhs) {
        return _mm512_add_ps(lhs, rhs);
    }
    PCX_AINLINE static auto sub(impl_vec lhs, impl_vec rhs) {
        return _mm512_sub_ps(lhs, rhs);
    }
    PCX_AINLINE static auto mul(impl_vec lhs, impl_vec rhs) {
        return _mm512_mul_ps(lhs, rhs);
    }
    PCX_AINLINE static auto div(impl_vec lhs, impl_vec rhs) {
        return _mm512_div_ps(lhs, rhs);
    }
    PCX_AINLINE static auto fmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fnmadd_ps(a, b, c);
    }
    PCX_AINLINE static auto fmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fmsub_ps(a, b, c);
    }
    PCX_AINLINE static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) {
        return _mm512_fnmsub_ps(a, b, c);
    }

    static constexpr struct upsample_t {
        inline static const auto idx2 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        inline static const auto idx4 = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
        inline static const auto idx8 = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);

        PCX_AINLINE auto operator()(vec_traits<f32, 2>::impl_vec v) const -> impl_vec {
            auto v128 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<f64*>(v.data())));
            auto vs   = _mm512_zextps128_ps512(v128);
            return _mm512_permutexvar_ps(idx2, vs);
        }
        PCX_AINLINE auto operator()(vec_traits<f32, 4>::impl_vec v) const -> impl_vec {
            auto vs = _mm512_zextps128_ps512(v);
            return _mm512_permutexvar_ps(idx4, vs);
        }
        PCX_AINLINE auto operator()(vec_traits<f32, 8>::impl_vec v) const -> impl_vec {
            auto vs = _mm512_zextps256_ps512(v);
            return _mm512_permutexvar_ps(idx8, vs);
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
        }
    };
    template<uZ To, uZ From>
        requires(To <= width && From <= width)
    static constexpr auto repack = repack_t<To, From>{};

    using tup_width = tupi::broadcast_tuple_t<impl_vec, width>;
    PCX_AINLINE static auto bit_reverse(tup_width tup) {
        constexpr auto unpck1lo = [](impl_vec a, impl_vec b) { return _mm512_unpacklo_ps(a, b); };
        constexpr auto unpck1hi = [](impl_vec a, impl_vec b) { return _mm512_unpackhi_ps(a, b); };
        constexpr auto unpck2lo = [](impl_vec a, impl_vec b) {
            return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
        };
        constexpr auto unpck2hi = [](impl_vec a, impl_vec b) {
            return _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
        };
        constexpr auto unpck4lo = [](impl_vec a, impl_vec b) {
            const auto idx = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck4hi = [](impl_vec a, impl_vec b) {
            const auto idx = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck8lo = [](impl_vec a, impl_vec b) {
            const auto idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            return _mm512_permutex2var_ps(a, idx, b);
        };
        constexpr auto unpck8hi = [](impl_vec a, impl_vec b) {
            const auto idx = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
            return _mm512_permutex2var_ps(a, idx, b);
        };

        auto res1 = [unpck1lo, unpck1hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return tupi::make_tuple(unpck1lo(tupi::get<Is>(tup), tupi::get<Is + 8>(tup))...,
                                    unpck1hi(tupi::get<Is>(tup), tupi::get<Is + 8>(tup))...);
        }(tup, std::make_index_sequence<8>{});

        auto res2 = [unpck2lo, unpck2hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return tupi::make_tuple(unpck2lo(tupi::get<Is>(tup), tupi::get<Is + 4>(tup))...,
                                    unpck2hi(tupi::get<Is>(tup), tupi::get<Is + 4>(tup))...);
        }(res1, std::index_sequence<0, 1, 2, 3, 8, 9, 10, 11>{});

        auto res4 = [unpck4lo, unpck4hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return tupi::make_tuple(unpck4lo(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...,
                                    unpck4hi(tupi::get<Is>(tup), tupi::get<Is + 2>(tup))...);
        }(res2, std::index_sequence<0, 1, 4, 5, 8, 9, 12, 13>{});

        auto res8 = [unpck8lo, unpck8hi]<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return tupi::make_tuple(unpck8lo(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...,
                                    unpck8hi(tupi::get<Is>(tup), tupi::get<Is + 1>(tup))...);
        }(res4, std::index_sequence<0, 2, 4, 6, 8, 10, 12, 14>{});
        auto resort = []<uZ... Is>(auto tup, std::index_sequence<Is...>) {
            return tupi::make_tuple(tupi::get<Is>(tup)..., tupi::get<Is + 8>(tup)...);
        }(res8, std::index_sequence<0, 2, 1, 3, 4, 6, 5, 7>{});
        return resort;
    }
};

// clang-format off
template<>
struct vec_traits<f32, 16>::repack_t<1, 16> {
    const static inline auto idx0 = _mm512_setr_epi32( 0,16, 1,17, 2,18, 3,19, 4,20, 5,21, 6,22, 7,23);
    const static inline auto idx1 = _mm512_setr_epi32( 8,24, 9,25,10,26,11,27,12,28,13,29,14,30,15,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<2, 16> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1,16,17, 2, 3,18,19, 4, 5,20,21, 6, 7,22,23);
    const static inline auto idx1 = _mm512_setr_epi32(8, 9,24,25,10,11,26,27,12,13,28,29,14,15,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<4, 16> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3,16,17,18,19, 4, 5, 6, 7,20,21,22,23);
    const static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,24,25,26,27,12,13,14,15,28,29,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<8, 16> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,16,17,18,19,20,21,22,23);
    const static inline auto idx1 = _mm512_setr_epi32(8, 9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<1, 8> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<2, 8> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<4, 8> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<16, 8> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<8,16>{}(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<1, 4> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<2, 4> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<8, 4> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<4,8>{}(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<16, 4> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 2, 3, 8, 9,10,11,16,17,18,19,24,25,26,27);
    const static inline auto idx1 = _mm512_setr_epi32( 4, 5, 6, 7,12,13,14,15,20,21,22,23,28,29,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<1, 2> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<4, 2> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<2,4>{}(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<8, 2> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<16, 2> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 4, 5, 8, 9,12,13,16,17,20,21,24,25,28,29);
    const static inline auto idx1 = _mm512_setr_epi32( 2, 3, 6, 7,10,11,14,15,18,19,22,23,26,27,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<2, 1> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return repack_t<1,2>{}(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<4, 1> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<8, 1> {
    const static inline auto idx0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        a = _mm512_permutexvar_ps(idx0, a);
        b = _mm512_permutexvar_ps(idx0, b);
        return tupi::make_tuple(a,b);
    }
};
template<>
struct vec_traits<f32, 16>::repack_t<16, 1> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 2, 4, 6, 8,10,12,14,16,18,20,22,24,26,28,30);
    const static inline auto idx1 = _mm512_setr_epi32( 1, 3, 5, 7, 9,11,13,15,17,19,21,23,25,27,29,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::split_interleave_t<1> {
    const static inline auto idx0 = _mm512_setr_epi32( 0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30);
    const static inline auto idx1 = _mm512_setr_epi32( 1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::split_interleave_t<2> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29);
    const static inline auto idx1 = _mm512_setr_epi32( 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::split_interleave_t<4> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27);
    const static inline auto idx1 = _mm512_setr_epi32( 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::split_interleave_t<8> {
    const static inline auto idx0 = _mm512_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7,16,17,18,19,20,21,22,23);
    const static inline auto idx1 = _mm512_setr_epi32( 8, 9,10,11,12,13,14,15,24,25,26,27,28,29,30,31);
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        auto x = _mm512_permutex2var_ps(a, idx0, b);
        auto y = _mm512_permutex2var_ps(a, idx1, b);
        return tupi::make_tuple(x,y);
    }
};
template<>
struct vec_traits<f32, 16>::split_interleave_t<16> {
    PCX_AINLINE auto operator()(impl_vec a, impl_vec b) const {
        return tupi::make_tuple(a, b);
    }
};
// clang-format on
#else
template<>
struct max_vec_width<f32> {
    static constexpr uZ value = 8;
};
#endif
}    // namespace pcx::simd::detail_
// NOLINTEND(*portability*, *magic-number*)
