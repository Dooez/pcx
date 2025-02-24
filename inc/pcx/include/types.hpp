#ifndef PCX_TYPES_HPP
#define PCX_TYPES_HPP

#include <concepts>
#include <cstdint>
#include <limits>
#include <ranges>
#include <type_traits>

#if defined(__clang__)
#define PCX_AINLINE  [[clang::always_inline]] inline
#define PCX_LAINLINE [[clang::always_inline]]
#elif defined(__GNUC__) || defined(__GNUG__)
#define PCX_AINLINE  [[gnu::always_inline]] inline
#define PCX_LAINLINE [[gnu::always_inline]]
#endif


namespace pcx {
using f32 = float;
using f64 = double;

using uZ  = std::size_t;
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u8  = uint8_t;

using iZ  = std::ptrdiff_t;
using i64 = int64_t;
using i32 = int32_t;
using i16 = int16_t;
using i8  = int8_t;

namespace stdv = std::views;
namespace stdr = std::ranges;

template<uZ I>
using uZ_ce = std::integral_constant<uZ, I>;
template<uZ... Is>
using uZ_seq = std::index_sequence<Is...>;
template<uZ N>
using make_uZ_seq = std::make_index_sequence<N>;
template<auto V>
struct val_ce : std::integral_constant<decltype(V), V> {
    using value_type = decltype(V);
    using type       = val_ce<V>;

    static constexpr auto value = V;

    consteval operator value_type() const {    // NOLINT (*explicit*)
        return V;
    }
    static consteval auto operator()() {
        return V;
    }
};

template<uZ N>
concept power_of_two = N > 0 && (N & (N - 1)) == 0;

template<typename T>
concept floating_point = std::same_as<T, float> || std::same_as<T, double>;

template<typename T, uZ PackSize>
concept packed_floating_point = floating_point<T> && power_of_two<PackSize>;

template<iZ Power = 1>
struct imag_unit_t {
    friend auto conj(imag_unit_t) {
        return imag_unit_t<(-Power) % 4>{};
    }
};
template<iZ Power = 1>
inline constexpr auto imag_unit = imag_unit_t<Power>{};

namespace simd {

namespace detail_ {
template<typename T>
struct max_vec_width;

/**
 * @brief A structure with common operations over simd vectors.
 * 
 */
template<typename T, uZ Width>
struct vec_traits {
    struct impl_vec;
    static auto set1(T value) -> impl_vec;
    static auto zero() -> impl_vec;
    static auto load(const T* src);
    static void store(T* dest, impl_vec vec);
    static auto add(impl_vec lhs, impl_vec rhs) -> impl_vec;
    static auto sub(impl_vec lhs, impl_vec rhs) -> impl_vec;
    static auto mul(impl_vec lhs, impl_vec rhs) -> impl_vec;
    static auto div(impl_vec lhs, impl_vec rhs) -> impl_vec;
    static auto fmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    static auto fnmadd(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    static auto fmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;
    static auto fnmsub(impl_vec a, impl_vec b, impl_vec c) -> impl_vec;

    /**
     * @brief Extends a lower size simd vector by duplicating it's elements.
     * 
     * Example:
     * Width == 8
     * vec_traits<T, 4> v == [0 1 2 3]
     * result == [0 0 1 1 2 2 3 3]
     */
    constexpr static struct {
        static auto operator()(vec_traits<T, Width>::impl_vec v) -> impl_vec;
        // static auto operator()(vec_traits<T, Width / 2>::impl_vec v) -> impl_vec;
        // . . .
        // static auto operator()(vec_traits<T, 2>::impl_vec) -> impl_vec;
    } upsample;

    template<uZ To, uZ From>
        requires(To <= Width && From <= Width)
    struct repack_t {
        static auto operator()(impl_vec a, impl_vec b);    // -> tupi::tuple<impl_vec, impl_vec>;
    };
    template<uZ To, uZ From>
        requires(To <= Width && From <= Width)
    constexpr static auto repack = repack_t<To, From>{};

    /**
     * @brief Splits two vectors in half by chunks of `ChunkSize` and returns vectors of interleaved even and odd chunks.
     *
     * Example:
     * Width     == 8
     * ChunkSize == 2
     *  a = [a0 a1 a2 a3 a4 a5 a6 a7] 
     *  b = [b0 b1 b2 b3 b4 b5 b6 b7] 
     * 
     *  result<0> = [a0 a1 b0 b1 a4 a5 b4 b5]
     *  result<1> = [a2 a3 b2 b3 a6 a7 b6 b7]
     */
    template<uZ ChunkSize>
        requires(ChunkSize <= Width)
    struct split_interleave_t {
        static auto operator()(impl_vec a, impl_vec b);    // -> tupi::tuple<impl_vec, impl_vec>;
    };
    template<uZ ChunkSize>
        requires(ChunkSize <= Width)
    constexpr static auto split_interleave = split_interleave_t<ChunkSize>{};

    struct tup_width;    // tupi::broadcast_tuple_t<impl_vec, Width>
    static auto bit_reverse(tup_width tup) -> tup_width;
};
}    // namespace detail_
template<typename T>
constexpr uZ max_width = detail_::max_vec_width<T>::value;

/**
 * @brief Simd vector abstraction.
 */
template<typename T, uZ Width = detail_::max_vec_width<T>::value>
    requires(Width <= detail_::max_vec_width<T>::value)
struct vec {
    using value_type = T;
    using traits     = detail_::vec_traits<value_type, Width>;
    using impl_vec_t = typename traits::impl_vec;

    impl_vec_t value;

    auto impl_vec() -> impl_vec_t& {
        return value;
    }

    PCX_AINLINE vec() = default;
    PCX_AINLINE vec(impl_vec_t v)    //NOLINT(*explicit*)
    : value(v) {};
};

/*template <typename T> using reg_t = typename reg<T>::type;*/

/**
 * @brief Simd vector of packed complex numbers.
 * Multiplication by imaginary unit and complex conjugation can be lazily evaluated 
 * to be perfromed at a later stage or during other operations.
 *
 * @tparam T	    
 * @tparam NReal    Negate the real part of a complex vector.
 * @tparam NImag    Negate the imaginary part of a complex vector.
 * @tparam Width    Simd vector width.
 * @tparam PackSize Pack size inside simd vector.
 */
template<typename T, bool NReal = false, bool NImag = false, uZ Width = max_width<T>, uZ PackSize = Width>
    requires power_of_two<Width> && power_of_two<PackSize> && (Width <= max_width<T>) && (PackSize <= Width)
struct cx_vec {
    using real_type = T;
    using vec_t     = vec<T, Width>;

    vec_t m_real;
    vec_t m_imag;

    PCX_AINLINE auto real(this auto&& v) -> decltype(auto) {
        return v.m_real;
    }
    PCX_AINLINE auto imag(this auto&& v) -> decltype(auto) {
        return v.m_imag;
    }
    PCX_AINLINE auto real_v(this auto&& v) -> decltype(auto) {
        return v.m_real.value;
    }
    PCX_AINLINE auto imag_v(this auto&& v) -> decltype(auto) {
        return v.m_imag.value;
    }

    static consteval auto width() -> uZ {
        return Width;
    }
    static consteval auto pack_size() -> uZ {
        return PackSize;
    }
    static consteval auto neg_real() -> bool {
        return NReal;
    }
    static consteval auto neg_imag() -> bool {
        return NImag;
    }

    template<iZ Count = 1>
    PCX_AINLINE friend auto mul_by_j(const cx_vec& vec) {
        constexpr auto rot = (Count % 4 + 4) % 4;
        if constexpr (rot == 0) {
            return vec;
        } else if constexpr (rot == 1) {
            using new_cx_vec = cx_vec<T, !NImag, NReal, Width, PackSize>;
            return new_cx_vec{.m_real = vec.m_imag, .m_imag = vec.m_real};
        } else if constexpr (rot == 2) {
            using new_cx_vec = cx_vec<T, !NReal, !NImag, Width, PackSize>;
            return new_cx_vec{.m_real = vec.m_real, .m_imag = vec.m_imag};
        } else if constexpr (rot == 3) {
            using new_cx_vec = cx_vec<T, NImag, !NReal, Width, PackSize>;
            return new_cx_vec{.m_real = vec.m_imag, .m_imag = vec.m_real};
        }
    }

    PCX_AINLINE friend auto conj(const cx_vec& vec) {
        using new_cx_vec = cx_vec<T, NReal, !NImag, Width, PackSize>;
        return new_cx_vec{.m_real = vec.m_real, .m_imag = vec.m_imag};
    }
    template<uZ I>
        requires(I < 2)
    friend auto get(const cx_vec& vec) {
        if constexpr (I == 0) {
            return vec.real();
        } else {
            return vec.imag();
        }
    }
};
namespace detail_ {
template<typename T>
struct is_cx_vec : std::false_type {};
template<typename T, bool NReal, bool NImag, uZ Width, uZ PackSize>
struct is_cx_vec<cx_vec<T, NReal, NImag, Width, PackSize>> : std::true_type {};
}    // namespace detail_

template<typename T>
concept any_cx_vec = detail_::is_cx_vec<T>::value;

template<typename T>
concept tight_cx_vec = any_cx_vec<T> && (T::width() == T::pack_size());

template<typename T>
concept eval_cx_vec = any_cx_vec<T> && (!T::neg_real() && !T::neg_imag());

template<typename T, typename U>
concept compatible_cx_vec = (any_cx_vec<T> &&                                                 //
                             any_cx_vec<U> &&                                                 //
                             std::same_as<typename T::real_type, typename U::real_type> &&    //
                             T::width() == U::width() &&                                      //
                             T::pack_size() == U::pack_size());

}    // namespace simd
}    // namespace pcx

template<pcx::simd::any_cx_vec V>
struct std::tuple_size<V> : std::integral_constant<std::size_t, 2> {};
template<std::size_t I, pcx::simd::any_cx_vec V>
    requires(I < 2)
struct std::tuple_element<I, V> {
    using type = V::vec_t;
};

#endif
