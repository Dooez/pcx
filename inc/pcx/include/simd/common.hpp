#ifndef PCX_SIMD_COMMON_HPP
#define PCX_SIMD_COMMON_HPP
#include "pcx/include/types.hpp"

#include <algorithm>

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline
// #define PCX_LBDINLINE [[gnu::always_inline, clang::always_inline]]
namespace pcx::simd {

template<uZ Width = max_width<f32>>
PCX_AINLINE auto load(const f32* src) {
    return detail_::vec_traits<f32, Width>::load(src);
}
template<uZ Width = max_width<f64>>
PCX_AINLINE auto load(const f64* src) {
    return detail_::vec_traits<f64, Width>::load(src);
}
template<typename T, uZ Width>
PCX_AINLINE auto store(T* dest, vec<T, Width> data) {
    detail_::vec_traits<T, Width>::store(dest, data.value);
}

template<uZ SrcPackSize, uZ Width = max_width<f32>>
PCX_AINLINE auto cxload(const f32* src) {
    constexpr uZ pack_size   = std::min(SrcPackSize, Width);
    constexpr uZ load_offset = std::max(SrcPackSize, Width);
    using cx_vec_t           = cx_vec<f32, false, false, Width, pack_size>;
    return cx_vec_t{
        .m_real = load<Width>(src),
        .m_imag = load<Width>(src + load_offset),    //NOLINT(*pointer*)
    };
}
template<uZ SrcPackSize, uZ Width = max_width<f64>>
PCX_AINLINE auto cxload(const f64* src) {
    constexpr uZ pack_size   = std::min(SrcPackSize, Width);
    constexpr uZ load_offset = std::max(SrcPackSize, Width);
    using cx_vec_t           = cx_vec<f64, false, false, Width, pack_size>;
    return cx_vec_t{
        .m_real = load<Width>(src),
        .m_imag = load<Width>(src + load_offset),    //NOLINT(*pointer*)
    };
}
template<uZ DestPackSize, eval_cx_vec V>
    requires(DestPackSize == V::width() || (tight_cx_vec<V> && DestPackSize > V::width()))
PCX_AINLINE auto cxstore(typename V::real_type* dest, V data) {
    constexpr uZ store_offset = std::max(DestPackSize, V::width());
    store(dest, data.real());
    store(dest + store_offset, data.imag());
}

}    // namespace pcx::simd

#undef PCX_ANILINE
#endif
