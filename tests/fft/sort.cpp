#include "pcx/include/fft_impl.hpp"

#include <array>
#include <cstdint>
#include <iterator>
#include <print>
#include <ranges>
#include <vector>

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

constexpr auto reverse_bit_order(u64 num, u64 max) -> u64 {
    auto depth = log2i(max);
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

namespace stdv = std::views;
namespace stdr = std::ranges;

template<uZ Width>
inline bool test_sort(auto data) {
    auto size = data.size();
    if (size < Width * Width)
        return false;

    using impl = pcx::detail_::br_sort_inplace<Width, true>;
    auto idx   = std::vector<uZ>{};
    impl::insert_idxs(idx, size);
    auto cnt      = impl::swap_count(size);
    f32* data_ptr = reinterpret_cast<f32*>(data.data());
    impl::perform(pcx::cxpack<1, f32>{}, size, data_ptr, cnt, idx.data());
    bool falty = false;
    for (auto [i, v]: stdv::enumerate(data)) {
        if (i != static_cast<uZ>(v.real())) {
            std::println("Error sorting size {} width {}. i: {}, v: {}", size, Width, i, v);
            falty = true;
        }
    }
    if (falty)
        return false;

    std::println("Success sorting size {} width {}.", size, Width);
    return true;
}


int main() {
    constexpr auto w_seq = pcx::uZ_seq<2, 4, 8, 16>{};

    uZ size = 2;
    while (size < 4096 * 1024) {
        auto idx = stdr::to<std::vector<uZ>>(stdv::iota(0U, size));
        for (auto& v: idx) {
            v = reverse_bit_order(v, size);
        }
        auto cxbr = stdr::to<std::vector<std::complex<f32>>>(idx);
        if (![&]<uZ... Width>(pcx::uZ_seq<Width...>) {
                return (((size < Width * Width) || test_sort<Width>(cxbr)) && ...);
            }(w_seq))
            return -1;
        size *= 2;
    }
    return 0;
}
