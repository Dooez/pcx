
#ifndef PCX_TYPES_HPP
#define PCX_TYPES_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <ranges>
#include <type_traits>

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline

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

template<uZ N>
concept power_of_two = N > 0 && (N & (N - 1)) == 0;

template<typename T>
concept floating_point = std::same_as<T, float> || std::same_as<T, double>;

template<typename T, uZ PackSize>
concept packed_floating_point = floating_point<T> && power_of_two<PackSize>;

namespace simd {


namespace detail_ {
template<typename T>
struct default_vec_size;
template<typename T, uZ Width>
struct vec_traits;

}    // namespace detail_

/**
 * @brief Simd vector abstraction.
 */
template<typename T, uZ Width = detail_::default_vec_size<T>::value>
struct vec {
    using value_type = T;
    using vec_type   = typename detail_::vec_traits<T, Width>::type;

    vec_type value;
    vec(vec_type v)    //NOLINT(*explicit*)
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
 * @tparam Size	    Simd vector size.
 * @tparam PackSize Pack size inside simd vector.
 */
template<typename T, bool NReal, bool NImag, uZ Width = 8, uZ PackSize = Width>
    requires power_of_two<Width> && power_of_two<PackSize>
struct cx_vec {
    using real_type = T;
    using vec_t     = vec<T, Width>;

    vec_t m_real;
    vec_t m_imag;

    PCX_AINLINE auto real() -> vec_t& {
        return m_real;
    }
    PCX_AINLINE auto imag() -> vec_t& {
        return m_imag;
    }

    static consteval auto size() -> uZ {
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

    template<uZ Rot>
        requires(Rot < 4)
    PCX_AINLINE auto rotate() const {
        if constexpr (Rot == 0) {
            return *this;
        } else if (Rot == 1) {
            using new_cx_vec = cx_vec<T, !NImag, NReal, Width, PackSize>;
            return new_cx_vec{.m_real = m_imag, .m_imag = m_real};
        } else if (Rot == 2) {
            using new_cx_vec = cx_vec<T, !NReal, !NImag, Width, PackSize>;
            return new_cx_vec{.m_real = m_real, .m_imag = m_imag};
        } else if (Rot == 3) {
            using new_cx_vec = cx_vec<T, NImag, !NReal, Width, PackSize>;
            return new_cx_vec{.m_real = m_imag, .m_imag = m_real};
        }
    }

    PCX_AINLINE auto conj() const {
        using new_cx_vec = cx_vec<T, NReal, !NImag, Width, PackSize>;
        return new_cx_vec{.m_real = m_real, .m_imag = m_imag};
    }
};

namespace detail_ {
template<typename T>
struct is_cx_vec : std::false_type {};
template<typename T, bool NReal, bool NImag, uZ Width, uZ PackSize>
struct is_cx_vec<cx_vec<T, NReal, NImag, Width, PackSize>> : std::true_type {};
}    // namespace detail_

template<typename T>
concept cx_vec_c = detail_::is_cx_vec<T>::value;

template<typename T>
concept tight_cx_vec = cx_vec_c<T> && (T::size() == T::pack_size());

template<typename T, typename U>
concept compatible = (cx_vec_c<T> &&                                                     //
                      cx_vec_c<U> &&                                                     //
                      std::same_as<typename T::value_type, typename U::value_type> &&    //
                      T::size() == U::size() &&                                          //
                      T::pack_size() == U::pack_size());

}    // namespace simd
}    // namespace pcx
//
#undef PCX_AINLINE
#endif
