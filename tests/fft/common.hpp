#pragma once
#include "pcx/include/fft/fft_impl.hpp"

#include <cmath>
#include <print>
#include <vector>

namespace pcx::testing {
enum class permute_t {
    bit_reversed = 0,
    normal       = 1,
    shifted      = 2,
};
template<typename T>
class chk_t {
    std::array<std::vector<std::complex<T>>, 3> data{};

public:
    auto operator()(permute_t type) -> std::vector<std::complex<T>>& {
        return data[static_cast<uZ>(type)];
    }
    auto operator()(permute_t type) const -> const std::vector<std::complex<T>>& {
        return data[static_cast<uZ>(type)];
    }
};
inline constexpr auto half_tw    = meta::val_seq<true>{};
inline constexpr auto low_k      = meta::val_seq<true>{};
inline constexpr auto node_sizes = uZ_seq<PCX_TESTING_NODE_SIZES>{};
inline constexpr auto perm_types = meta::val_seq<permute_t::normal>{};

inline constexpr auto local_tw = meta::val_seq<false>{};


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
template<typename fX>
void naive_reverse(std::vector<std::complex<fX>>& data, uZ node_size, uZ vec_width);

inline auto check_nan(const std::vector<std::complex<f32>>& vec) {
    return std::ranges::any_of(vec, [](auto v) { return std::isnan(v.real()) || std::isnan(v.imag()); });
};
inline auto check_nan(const std::vector<std::complex<f64>>& vec) {
    return std::ranges::any_of(vec, [](auto v) { return std::isnan(v.real()) || std::isnan(v.imag()); });
};
template<typename fX>
bool check_correctness(const std::vector<std::complex<fX>>& naive,
                       const std::vector<std::complex<fX>>& pcx,
                       uZ                                   width,
                       uZ                                   node_size,
                       bool                                 lowk,
                       bool                                 local_tw,
                       bool                                 half_tw);

template<typename fX, typename R>
bool par_check_correctness(std::complex<fX> val,
                           const R&         pcx,
                           uZ               fft_size,
                           uZ               fft_id,
                           uZ               width,
                           uZ               node_size,
                           bool             local_tw);

void bit_reverse(auto& r) {
    auto size = r.size();
    for (auto i: stdv::iota(0U, size)) {
        auto br = detail_::reverse_bit_order(i, detail_::log2i(size));
        if (br > i)
            std::swap(r[i], r[br]);
    }
}
void shifted_bit_reverse(auto& r) {
    auto size = r.size();
    if (size < 4)
        return;
    auto rbo_sh = [=](auto k) {
        auto br = detail_::reverse_bit_order(k, detail_::log2i(size));
        return (br + size / 2) % size;
    };
    for (uZ i: stdv::iota(0U, size)) {
        uZ br0 = rbo_sh(i);
        uZ br1 = rbo_sh(br0);
        uZ br2 = rbo_sh(br1);
        if (std::min({i, br0, br1, br2}) == i) {
            auto vi = r[i];
            auto v0 = r[br0];
            auto v1 = r[br1];
            auto v2 = r[br2];
            r[i]    = v2;
            r[br0]  = vi;
            r[br1]  = v0;
            r[br2]  = v1;
        }
    }
}


}    // namespace pcx::testing
