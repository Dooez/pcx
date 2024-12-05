#ifndef PCX_SIMD_COMMON_HPP
#define PCX_SIMD_COMMON_HPP

#include "pcx/include/simd/math.hpp"
#include "pcx/include/simd/traits.hpp"

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
        return vec<T, real_width>{vec_traits<f32, real_width>::set1(*src)};
    }
};
template<uZ Width>
    requires(Width == 0 || power_of_two<Width>)
struct load_t {
    template<typename T>
        requires(Width <= max_width<T>)
    PCX_AINLINE auto operator()(const T* src) const {
        constexpr auto real_width = Width == 0 ? max_width<T> : Width;
        return vec<T, real_width>{vec_traits<f32, real_width>::load(src)};
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
    PCX_AINLINE auto operator()(const f32* src) const
        requires(Width <= max_width<f32>)
    {
        constexpr auto real_width = Width == 0 ? max_width<f32> : Width;
        using cx_vec_t            = cx_vec<f32, false, false, real_width>;
        return cx_vec_t{
            .m_real = broadcast<real_width>(src),
            .m_imag = broadcast<real_width>(src + SrcPackSize),    //NOLINT(*pointer*)
        };
    }
    PCX_AINLINE auto operator()(const f64* src) const
        requires(Width <= max_width<f64>)
    {
        constexpr auto real_width = Width == 0 ? max_width<f64> : Width;
        using cx_vec_t            = cx_vec<f64, false, false, real_width>;
        return cx_vec_t{
            .m_real = broadcast<real_width>(src),
            .m_imag = broadcast<real_width>(src + SrcPackSize),    //NOLINT(*pointer*)
        };
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
};

template<uZ PackTo>
    requires power_of_two<PackTo>
struct repack_t : tupi::compound_op_base {
    template<eval_cx_vec V>
        requires(PackTo <= V::width())
    PCX_AINLINE auto operator()(V vec) const {
        using real_type       = typename V::real_type;
        using repacked_vec_t  = cx_vec<real_type, false, false, V::width(), PackTo>;
        using traits          = detail_::vec_traits<real_type, V::width()>;
        constexpr auto repack = traits::template repack<PackTo, V::pack_size()>;

        auto [re, im] = repack(vec.real_v(), vec.imag_v());
        return repacked_vec_t{.m_real = re, .m_imag = im};
    }
    template<uZ I>
    PCX_AINLINE constexpr friend auto get_stage(const repack_t&) {
        return stage_t<I>{};
    }

private:
    template<typename T, uZ Width, uZ PackFrom, typename IR>
    struct interim_wrapper {
        IR result;
    };
    template<typename T, uZ Width, uZ PackFrom, typename IR>
    static constexpr auto wrap_interim(IR res) {
        return tupi::make_interim(interim_wrapper<T, Width, PackFrom, IR>(res));
    }
    template<uZ I>
    struct stage_t {
        template<eval_cx_vec V>
            requires(I == 0 && PackTo <= V::width())
        PCX_AINLINE auto operator()(V v) const {
            constexpr auto width     = V::width();
            constexpr auto pack_from = V::pack_size();
            using real_type          = typename V::real_type;
            using traits             = detail_::vec_traits<real_type, width>;
            using repacked_vec_t     = cx_vec<real_type, false, false, width, PackTo>;

            constexpr auto repack = traits::template repack<PackTo, pack_from>;
            if constexpr (tupi::compound_op<decltype(repack)>) {
                auto stage = get_stage<I>(repack);
                if constexpr (tupi::final_result<decltype(stage(v.real_v(), v.imag_v()))>) {
                    auto [re, im] = stage(v.real_v(), v.imag_v());
                    return repacked_vec_t{.m_real = re, .m_imag = im};
                } else {
                    return wrap_interim<real_type, width, pack_from>(stage(v.real_v(), v.imag_v()));
                }
            } else {
                auto [re, im] = repack(v.real_v(), v.imag_v());
                return repacked_vec_t{.m_real = re, .m_imag = im};
            }
        }
        template<typename T, uZ Width, uZ PackFrom, typename IR>
            requires(I > 0)
        PCX_AINLINE auto operator()(interim_wrapper<T, Width, PackFrom, IR> wrapper) const {
            using traits          = detail_::vec_traits<T, Width>;
            constexpr auto repack = traits::template repack<PackTo, PackFrom>;
            using repacked_vec_t  = cx_vec<T, false, false, Width, PackTo>;
            auto stage            = tupi::apply | get_stage<I>(repack);
            if constexpr (tupi::final_result<decltype(stage(wrapper.result))>) {
                auto [re, im] = stage(wrapper.result);
                return repacked_vec_t{.m_real = re, .m_imag = im};
            } else {
                return wrap_interim<T, Width, PackFrom>(stage(wrapper.result));
            }
        }
    };
};
}    // namespace detail_

template<uZ SrcPackSize, uZ Width = 0>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
inline constexpr auto cxbroadcast = detail_::cxbroadcast_t<SrcPackSize, Width>{};

template<uZ SrcPackSize, uZ Width = 0>
    requires power_of_two<SrcPackSize> && (Width == 0 || power_of_two<Width>)
inline constexpr auto cxload = detail_::cxload_t<SrcPackSize, Width>{};

template<uZ DestPackSize>
    requires power_of_two<DestPackSize>
inline constexpr auto cxstore = detail_::cxstore_t<DestPackSize>{};

template<uZ PackSize>
    requires power_of_two<PackSize>
inline constexpr auto repack = detail_::repack_t<PackSize>{};

inline constexpr struct {
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
} evaluate;

}    // namespace pcx::simd
#endif
