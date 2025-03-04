#pragma once
#include "pcx/include/fft_impl.hpp"

#include <print>
#include <vector>

namespace pcx::testing {
#ifdef FULL_FFT_TEST
inline constexpr auto f32_widths = uZ_seq<4, 8, 16>{};
inline constexpr auto f64_widths = uZ_seq<2, 4, 8>{};
inline constexpr auto half_tw    = meta::val_seq<false, true>{};
#else
inline constexpr auto f32_widths = uZ_seq<16>{};
inline constexpr auto f64_widths = uZ_seq<8>{};
inline constexpr auto half_tw    = meta::val_seq<true>{};
#endif
inline constexpr auto low_k    = meta::val_seq<true>{};
inline constexpr auto local_tw = meta::val_seq<true>{};

template<typename T, uZ NodeSize>
bool test_fft(uZ fft_size);
}    // namespace pcx::testing

template<typename T>
struct std::formatter<std::complex<T>, char> {
    std::formatter<T, char> m_fmt;
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        return m_fmt.parse(ctx);
    }

    template<class FmtContext>
    FmtContext::iterator format(std::complex<T> cxv, FmtContext& ctx) const {
        *ctx.out() = "(";
        auto out   = m_fmt.format(cxv.real(), ctx);
        *(out++)   = ",";
        *out       = " ";
        out        = m_fmt.format(cxv.imag(), ctx);
        *(out++)   = ")";
        return out;
    }
};
static auto get_type_name(pcx::meta::types<pcx::f32>) {
    return std::string_view("f32");
}
static auto get_type_name(pcx::meta::types<pcx::f64>) {
    return std::string_view("f64");
}
template<typename T>
struct std::formatter<pcx::meta::types<T>> {
    template<class ParseContext>
    constexpr auto parse(ParseContext& ctx) -> ParseContext::iterator {
        auto it = ctx.begin();
        if (it == ctx.end())
            return it;
        if (it != ctx.end() && *it != '}')
            throw std::format_error("Invalid format args for pcx::meta::types<T>.");

        return it;
    }

    template<class FmtContext>
    FmtContext::iterator format(pcx::meta::types<T> mt, FmtContext& ctx) const {
        return std::ranges::copy(get_type_name(mt), ctx.out()).out;
    }
};

namespace pcx::testing {
template<typename fX>
void naive_fft(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width);

template<typename fX, uZ Width, uZ NodeSize, bool LowK, bool LocalTw, bool HalfTw>
bool test_prototype(uZ fft_size) {
    constexpr auto half_tw = std::bool_constant<HalfTw>{};
    const uZ       freq_n  = 0;

    auto datavec = [=]() {
        auto vec = std::vector<std::complex<fX>>(fft_size);
        for (auto [i, v]: stdv::enumerate(vec)) {
            v = std::exp(std::complex<fX>(0, 1)                 //
                         * static_cast<fX>(2)                   //
                         * static_cast<fX>(std::numbers::pi)    //
                         * static_cast<fX>(i)                   //
                         * static_cast<fX>(freq_n)              //
                         / static_cast<fX>(fft_size));
        }
        return vec;
    }();
    auto datavec2 = datavec;
    naive_fft(datavec, NodeSize, Width);

    auto* data_ptr = reinterpret_cast<fX*>(datavec2.data());
    auto  tw       = [=] {
        using tw_t = detail_::tw_data_t<fX, LocalTw>;
        if constexpr (LocalTw) {
            return tw_t{1, 0};
        } else {
            // return tw_t{tw_ptr};
        }
    }();

    using fimpl = pcx::detail_::transform<NodeSize, fX, Width>;
    fimpl::template perform<1, 1>(fft_size, data_ptr, tw, half_tw);

    auto subtform_error = stdr::any_of(stdv::zip(datavec, datavec2),    //
                                       [](auto v) { return std::get<0>(v) != std::get<1>(v); });
    if (subtform_error) {
        std::println("[Error] {}×{}, width {}, node size {}{}{}.",
                     pcx::meta::types<fX>{},
                     fft_size,
                     Width,
                     NodeSize,
                     LowK ? ", low k" : "",
                     LocalTw ? ", local tw" : "");
        for (auto [i, naive, pcx]: stdv::zip(stdv::iota(0U), datavec, datavec2) | stdv::take(999999)) {
            // if (naive != pcx)
            std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                         i,
                         (naive),
                         (pcx),
                         (naive - pcx));
            if (std::abs(naive - pcx) > 1) {
                // // std::println("{:>3}| naive:{: >6.2f}, pcx:{: >6.2f}, diff:{}",    //
                // //              i,
                // //              (naive),
                // //              (pcx),
                // //              (naive - pcx));
                // std::println("Over 1 found.");
                // break;
            }
            // }
        }
        return false;
    }
    std::println("[Success] {}×{}, width {}, node size {}{}{}.",
                 pcx::meta::types<fX>{},
                 fft_size,
                 Width,
                 NodeSize,
                 LowK ? ", low k" : "",
                 LocalTw ? ", local tw" : "");
    return true;
}

template<typename fX, uZ NodeSize, uZ... VecWidth, bool... low_k, bool... local_tw, bool... half_tw>
bool run_tests(uZ_seq<VecWidth...>,
               meta::val_seq<low_k...>,
               meta::val_seq<local_tw...>,
               meta::val_seq<half_tw...>,
               uZ fft_size) {
    auto lk_passed = [=]<bool LowK>(val_ce<LowK>) {
        auto ltw_passed = [=]<bool LocalTw>(val_ce<LocalTw>) {
            auto htw_passed = [=]<bool HalfTw>(val_ce<HalfTw>) {
                return ((fft_size <= NodeSize * VecWidth
                         || test_prototype<fX, VecWidth, NodeSize, LowK, LocalTw, HalfTw>(fft_size))
                        && ...);
            };
            return (htw_passed(val_ce<half_tw>{}) && ...);
        };
        return (ltw_passed(val_ce<local_tw>{}) && ...);
    };
    return (lk_passed(val_ce<low_k>{}) && ...);
};

}    // namespace pcx::testing
