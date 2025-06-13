#pragma once
#include "pcx/include/meta.hpp"
#include "pcx/include/tupi.hpp"
#include "pcx/include/types.hpp"

#include <complex>
#include <numbers>

namespace pcx::detail_ {

constexpr auto log2i(u64 num) -> uZ {
    u64 order = 0;
    for (u8 shift = 32; shift > 0; shift /= 2) {
        if (num >> shift > 0) {
            order += num >> shift > 0 ? shift : 0;
            num >>= shift;
        }
    }
    return order;
}
constexpr auto powi(u64 num, u64 pow) -> u64 {    // NOLINT(*recursion*)
    auto res = (pow % 2) == 1 ? num : 1UL;
    if (pow > 1) {
        auto half_pow = powi(num, pow / 2UL);
        res *= half_pow * half_pow;
    }
    return res;
}

constexpr auto reverse_bit_order(u64 num, u64 depth) -> u64 {
    if (depth == 0)
        return 0;
    //NOLINTBEGIN(*magic-numbers*)
    num = num >> 32U | num << 32U;
    num = (num & 0xFFFF0000FFFF0000U) >> 16U | (num & 0x0000FFFF0000FFFFU) << 16U;
    num = (num & 0xFF00FF00FF00FF00U) >> 8U | (num & 0x00FF00FF00FF00FFU) << 8U;
    num = (num & 0xF0F0F0F0F0F0F0F0U) >> 4U | (num & 0x0F0F0F0F0F0F0F0FU) << 4U;
    num = (num & 0xCCCCCCCCCCCCCCCCU) >> 2U | (num & 0x3333333333333333U) << 2U;
    num = (num & 0xAAAAAAAAAAAAAAAAU) >> 1U | (num & 0x5555555555555555U) << 1U;
    //NOLINTEND(*magic-numbers*)
    return num >> (64 - depth);
}

template<typename T>
struct data_info_base {};

template<floating_point T, bool Contiguous, typename C = void>
struct data_info : public data_info_base<T> {
    using data_ptr_t    = std::conditional_t<Contiguous, T*, C*>;
    using data_offset_t = std::conditional_t<Contiguous, decltype([] {}), uZ>;
    using k_offset_t    = std::conditional_t<Contiguous, decltype([] {}), uZ>;
    using k_stride_t    = std::conditional_t<Contiguous, uZ, decltype([] {})>;

    data_ptr_t                          data_ptr;
    uZ                                  stride = 1;
    [[no_unique_address]] k_stride_t    k_stride;
    [[no_unique_address]] k_offset_t    k_offset{};
    [[no_unique_address]] data_offset_t data_offset{};

    static constexpr auto sequential() -> std::false_type {
        return {};
    }
    static constexpr auto contiguous() -> std::bool_constant<Contiguous> {
        return {};
    }
    static constexpr auto empty() -> std::false_type {
        return {};
    }

    constexpr auto mul_stride(uZ n) const -> data_info {
        auto new_info = *this;
        new_info.stride *= n;
        return new_info;
    }
    constexpr auto div_stride(uZ n) const -> data_info {
        auto new_info = *this;
        new_info.stride /= n;
        return new_info;
    }
    constexpr auto reset_stride() const -> data_info {
        auto new_info = *this;
        if constexpr (Contiguous) {
            new_info.stride = k_stride;
        } else {
            new_info.stride = 1;
        }
        return new_info;
    }

    PCX_AINLINE constexpr auto offset_k(uZ n) const -> data_info {
        auto new_info = *this;
        if constexpr (Contiguous) {
            new_info.data_ptr += k_stride * n * 2;
        } else {
            new_info.k_offset += n;
        }
        return new_info;
    }
    PCX_AINLINE constexpr auto offset_contents(uZ n) const -> data_info {
        auto new_info = *this;
        if constexpr (Contiguous) {
            new_info.data_ptr += n * 2;
        } else {
            new_info.data_offset += n;
        }
        return new_info;
    }
    PCX_AINLINE constexpr auto get_batch_base(uZ i) const -> T* {
        if constexpr (Contiguous) {
            return data_ptr + i * stride * 2;
        } else {
            auto ptr = reinterpret_cast<T*>((*data_ptr)[i * stride + k_offset].data());
            return ptr + data_offset * 2;
        }
    };
};
template<floating_point T>
struct sequential_data_info : public data_info_base<T> {
    T* data_ptr;
    uZ stride = 1;

    static constexpr auto sequential() -> std::true_type {
        return {};
    }
    static constexpr auto contiguous() -> std::true_type {
        return {};
    }
    static constexpr auto empty() -> std::false_type {
        return {};
    }

    PCX_AINLINE constexpr auto mul_stride(uZ n) const -> sequential_data_info {
        auto new_info = *this;
        new_info.stride *= n;
        return new_info;
    }
    PCX_AINLINE constexpr auto div_stride(uZ n) const -> sequential_data_info {
        auto new_info = *this;
        new_info.stride /= n;
        return new_info;
    }
    PCX_AINLINE constexpr auto reset_stride() const -> sequential_data_info {
        auto new_info   = *this;
        new_info.stride = 1;
        return new_info;
    }
    PCX_AINLINE constexpr auto offset_k(uZ n) const -> sequential_data_info {
        return {{}, data_ptr + n * 2, stride};
    }
    PCX_AINLINE constexpr auto offset_contents(uZ n) const -> sequential_data_info {
        return {{}, data_ptr + n * 2, stride};
    }
    PCX_AINLINE constexpr auto get_batch_base(uZ i) const -> T* {
        return data_ptr + i * stride * 2;
    };
};
struct empty_data_info {
    static constexpr auto sequential() -> std::false_type {
        return {};
    }
    static constexpr auto contiguous() -> std::false_type {
        return {};
    }
    static constexpr auto empty() -> std::true_type {
        return {};
    }
    static constexpr auto mul_stride(uZ /*n*/) -> empty_data_info {
        return {};
    }
    static constexpr auto div_stride(uZ /*n*/) -> empty_data_info {
        return {};
    }
    static constexpr auto offset_k(uZ /*n*/) -> empty_data_info {
        return {};
    };
    static constexpr auto offset_contents(uZ /*n*/) -> empty_data_info {
        return {};
    };
};

template<typename T, typename U>
concept data_info_for = floating_point<U>
                        && (std::derived_from<T, data_info_base<U>>                            //
                            || std::derived_from<T, data_info_base<std::remove_const_t<U>>>    //
                            || std::same_as<T, empty_data_info>);

inline constexpr auto inplace_src = empty_data_info{};

template<typename T = f64>
inline auto wnk(uZ n, uZ k) -> std::complex<T> {
    if (n == 0)
        throw std::runtime_error("N should be non-zero\n");

    while ((k > 0) && (k % 2 == 0)) {
        k /= 2;
        n /= 2;
    }
    if (k == 0)
        return {1, 0};
    if (n == k * 2)
        return {-1, 0};
    if (n == k * 4)
        return {0, -1};
    if (k > n / 4) {
        auto v = wnk<T>(n, k - n / 4);
        return {v.imag(), -v.real()};
    }
    constexpr auto pi = std::numbers::pi;
    return static_cast<std::complex<T>>(
        std::exp(std::complex<f64>(0, -2 * pi * static_cast<f64>(k) / static_cast<f64>(n))));
}
/**
 * @brief Returnes twiddle factor with k bit-reversed
 */
template<typename T = f64>
inline auto wnk_br(uZ n, uZ k) -> std::complex<T> {
    k = reverse_bit_order(k, log2i(n) - 1);
    return wnk<T>(n, k);
}
constexpr auto next_pow_2(u64 v) {
    v--;
    v |= v >> 1U;
    v |= v >> 2U;
    v |= v >> 4U;
    v |= v >> 8U;
    v |= v >> 16U;
    v |= v >> 32U;
    v++;
    return v;
}


}    // namespace pcx::detail_
