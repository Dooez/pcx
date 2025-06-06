#ifndef PCX_SIMD_COMMON_HPP
#define PCX_SIMD_COMMON_HPP

#include "pcx/include/simd/math.hpp"
#include "pcx/include/simd/traits.hpp"
#include "pcx/include/tupi.hpp"

#include <algorithm>
#include <complex>

namespace pcx::simd {
namespace detail_ {
template<uZ Width>
    requires(Width == 0 || power_of_two<Width>)
struct broadcast_t {
    template<typename T>
        requires(Width <= max_width<T>)
    PCX_AINLINE auto operator()(const T* src) const {
        constexpr auto real_width = Width == 0 ? max_width<T> : Width;
        return vec<T, real_width>{vec_traits<T, real_width>::set1(*src)};
    }
};
template<uZ Width>
    requires(Width == 0 || power_of_two<Width>)
struct load_t {
    template<typename T>
        requires(Width <= max_width<T>)
    PCX_AINLINE auto operator()(const T* src) const {
        constexpr auto real_width = Width == 0 ? max_width<T> : Width;
        return vec<T, real_width>{vec_traits<T, real_width>::load(src)};
    }
};
}    // namespace detail_
template<uZ Width = 0>
    requires(Width == 0 || power_of_two<Width>)
inline constexpr auto broadcast = detail_::broadcast_t<Width>{};

template<uZ Width = 0>
    requires(Width == 0 || power_of_two<Width>)
inline constexpr auto load = detail_::load_t<Width>{};

inline constexpr struct {
    template<typename T, uZ Width>
    PCX_AINLINE void operator()(T* dest, vec<T, Width> data) const {
        detail_::vec_traits<T, Width>::store(dest, data.value);
    }
} store{};

namespace detail_ {
template<uZ SrcPackSize, uZ Width>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
struct cxbroadcast_t {
    template<typename T>
    PCX_AINLINE auto bcast_impl(const T* src) const {
        constexpr auto real_width = Width == 0 ? max_width<T> : Width;
        using cx_vec_t            = cx_vec<T, false, false, real_width>;
        return cx_vec_t{
            .m_real = broadcast<real_width>(src),
            .m_imag = broadcast<real_width>(src + SrcPackSize),    //NOLINT(*pointer*)
        };
    }
    PCX_AINLINE auto operator()(const f32* src) const
        requires(Width <= max_width<f32>)
    {
        return bcast_impl(src);
    }
    PCX_AINLINE auto operator()(const f64* src) const
        requires(Width <= max_width<f64>)
    {
        return bcast_impl(src);
    }
    template<floating_point T>
    PCX_AINLINE auto operator()(const std::complex<T>* src) const
        requires(SrcPackSize == 1 && Width <= max_width<T>)
    {
        return (*this)(reinterpret_cast<const T*>(src));
    }
};

template<uZ SrcPackSize, uZ Width>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
struct cxload_t {
    PCX_AINLINE auto operator()(const f32* src) const
        requires(Width <= max_width<f32>)
    {
        constexpr uZ real_width  = Width == 0 ? max_width<f32> : Width;
        constexpr uZ pack_size   = std::min(SrcPackSize, real_width);
        constexpr uZ load_offset = std::max(SrcPackSize, real_width);
        using cx_vec_t           = cx_vec<f32, false, false, real_width, pack_size>;
        return cx_vec_t{
            .m_real = load<real_width>(src),
            .m_imag = load<real_width>(src + load_offset),    //NOLINT(*pointer*)
        };
    }
    PCX_AINLINE auto operator()(const f64* src) const
        requires(Width <= max_width<f64>)
    {
        constexpr uZ real_width  = Width == 0 ? max_width<f64> : Width;
        constexpr uZ pack_size   = std::min(SrcPackSize, real_width);
        constexpr uZ load_offset = std::max(SrcPackSize, real_width);
        using cx_vec_t           = cx_vec<f64, false, false, real_width, pack_size>;
        return cx_vec_t{
            .m_real = load<real_width>(src),
            .m_imag = load<real_width>(src + load_offset),    //NOLINT(*pointer*)
        };
    }
    template<floating_point T>
    PCX_AINLINE auto operator()(const std::complex<T>* src) const
        requires(SrcPackSize == 1 && Width <= max_width<T>)
    {
        return (*this)(reinterpret_cast<const T*>(src));
    }
};
template<uZ DestPackSize>
    requires power_of_two<DestPackSize>
struct cxstore_t {
    template<eval_cx_vec V>
        requires(DestPackSize == V::pack_size() || (tight_cx_vec<V> && DestPackSize > V::width()))
    PCX_AINLINE void operator()(typename V::real_type* dest, V data) const {
        constexpr uZ store_offset = std::max(DestPackSize, V::width());
        store(dest, data.real());
        store(dest + store_offset, data.imag());
    }
    template<eval_cx_vec V>
        requires(DestPackSize == 1 && V::pack_size() == 1)
    PCX_AINLINE void operator()(std::complex<typename V::real_type>* dest, V data) const {
        auto* dest_ptr = reinterpret_cast<typename V::real_type*>(dest);
        store(dest_ptr, data.real());
        store(dest_ptr + 1, data.imag());
    }
};
}    // namespace detail_

template<uZ SrcPackSize, uZ Width = 0>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
inline constexpr auto cxbroadcast = tupi::pass | detail_::cxbroadcast_t<SrcPackSize, Width>{};

template<uZ SrcPackSize, uZ Width = 0>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
inline constexpr auto cxload = tupi::pass | detail_::cxload_t<SrcPackSize, Width>{};

template<uZ DestPackSize>
    requires power_of_two<DestPackSize>
inline constexpr auto cxstore = tupi::pass | detail_::cxstore_t<DestPackSize>{};

// clang-format off
template<uZ PackTo>
    requires power_of_two<PackTo>
static constexpr auto repack =   
    tupi::pass             
    | []<eval_cx_vec V>(V vec)
        requires(PackTo <= V::width())
      {
        using real_type       = typename V::real_type;
        using repacked_vec_t  = cx_vec<real_type, false, false, V::width(), PackTo>;
        using traits          = detail_::vec_traits<real_type, V::width()>;
        constexpr auto repack = traits::template repack<PackTo, V::pack_size()>;
        return tupi::make_tuple(tupi::make_tuple(repack, vec.real_v(), vec.imag_v()), meta::types<repacked_vec_t>{});
      }
    | tupi::pipeline(tupi::apply | tupi::invoke, tupi::pass)
    | tupi::apply
    | []<typename cx_vec>(auto tup, meta::types<cx_vec>){
        return cx_vec{.m_real = tupi::get<0>(tup), .m_imag = tupi::get<1>(tup)};
      }
    ;
// clang-format on
namespace detail_ {
struct evaluate_t {
    template<tight_cx_vec V>
        requires(!eval_cx_vec<V>)
    PCX_AINLINE auto operator()(V vec) const {
        using real_t    = V::real_type;
        using eval_vec  = cx_vec<real_t, false, false, V::width()>;
        using vec_t     = V::vec_t;
        using traits    = detail_::vec_traits<real_t, V::width()>;
        const auto zero = vec_t(traits::zero());
        if constexpr (V::neg_real() && V::neg_imag()) {
            return eval_vec{sub(zero, vec.real()), sub(zero, vec.imag())};
        } else if constexpr (V::neg_real()) {
            return eval_vec{sub(zero, vec.real()), vec.imag()};
        } else if constexpr (V::neg_imag()) {
            return eval_vec{vec.real(), sub(zero, vec.imag())};
        } else {
            return vec;
        }
    }
    template<eval_cx_vec V>
    PCX_AINLINE auto operator()(V vec) const {
        return vec;
    }
};
template<bool Conj>
struct maybe_conj_t {
    PCX_AINLINE static auto operator()(any_cx_vec auto vec) {
        if constexpr (Conj) {
            return conj(vec);
        } else {
            return vec;
        }
    }
    template<iZ Power>
    PCX_AINLINE static auto operator()(imag_unit_t<Power> v) {
        if constexpr (Conj) {
            return conj(v);
        } else {
            return v;
        }
    }
};
}    // namespace detail_
inline constexpr auto evaluate = tupi::pass | detail_::evaluate_t{};
template<bool Conj>
inline constexpr auto maybe_conj = tupi::pass | detail_::maybe_conj_t<Conj>{};


}    // namespace pcx::simd
#endif
