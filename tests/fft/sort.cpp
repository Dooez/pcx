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
auto shuffle_swap(std::array<std::array<uZ, Width>, Width> data) {
    auto res = std::array<std::array<uZ, Width>, Width>{};
    for (uZ i: stdv::iota(0U, Width)) {
        for (uZ k: stdv::iota(0U, Width)) {
            auto brk    = reverse_bit_order(k, Width);
            auto bri    = reverse_bit_order(i, Width);
            res[bri][k] = data[brk][i];
        }
    }
    return res;
}
template<uZ Width>
void do_swaps(auto& r) {
    auto size   = r.size();
    auto stride = size / Width;
    for (uZ i_slice: stdv::iota(0U, stride / Width)) {
        auto slice = std::array<std::array<uZ, Width>, Width>{};
        for (uZ i: stdv::iota(0U, Width)) {
            for (uZ k: stdv::iota(0U, Width)) {
                slice[i][k] = r[i_slice * Width + i * stride + k];
            }
        }
        auto sw = shuffle_swap(slice);
        for (uZ i: stdv::iota(0U, Width)) {
            for (uZ k: stdv::iota(0U, Width)) {
                r[i_slice * Width + i * stride + k] = sw[i][k];
            }
        }
    }
}

template<uZ Width>
auto get_close_swaps(uZ size, uZ coherent_size) {
    auto swaps = std::vector<uZ>{};
    for (uZ i: stdv::iota(0U, size)) {
        auto ibr       = reverse_bit_order(i, size);
        auto czone_i   = i / coherent_size;
        auto czone_ibr = ibr / coherent_size;
        // if (i >= coherent_size && ibr > i && ibr >= coherent_size) {
        if (ibr > i && czone_ibr == czone_i) {
            swaps.push_back(i);
            swaps.push_back(ibr);
        }
    }
    return swaps;
}
template<uZ Width>
auto get_far_swaps(uZ size, uZ coherent_size) {
    auto swaps = std::vector<uZ>{};
    for (uZ i: stdv::iota(0U, size)) {
        auto ibr       = reverse_bit_order(i, size);
        auto czone_i   = i / coherent_size;
        auto czone_ibr = ibr / coherent_size;
        // if (i >= coherent_size && ibr > i && ibr >= coherent_size) {
        if (czone_ibr > czone_i) {
            if (ibr > size) {
                std::println("error");
            }
            swaps.push_back(i);
            swaps.push_back(ibr);
        }
    }
    return swaps;
}
template<uZ Width, typename T>
void simd_swaps(T* data, uZ stride) {
    constexpr uZ pack = 1;
    using namespace pcx;
    auto mk_data_ptr = [=](uZ offset) {
        return [=]<uZ... Is>(uZ_seq<Is...>) {
            return tupi::make_tuple(data + offset * 2 + Is * stride * 2 ...);
        }(make_uZ_seq<Width>{});
    };
    for (uZ i: stdv::iota(0U, stride) | stdv::stride(Width)) {
        auto data_ptr = mk_data_ptr(i);
        auto data     = tupi::group_invoke(simd::cxload<pack, Width> | simd::repack<Width>, data_ptr);
        auto br_data  = simd::br_permute<T>(data);
        auto rdata    = tupi::group_invoke(simd::repack<1>, br_data);
        tupi::group_invoke(simd::cxstore<pack>, data_ptr, rdata);
    }
}


int main() {
    constexpr uZ width    = 16;
    constexpr uZ coh_size = 2048;

    uZ   size = width * 16 * 2;
    auto idx  = stdr::to<std::vector<uZ>>(stdv::iota(0U, size));
    auto br   = idx;
    for (auto& v: br) {
        v = reverse_bit_order(v, size);
    }

    auto cxidx   = stdr::to<std::vector<std::complex<f32>>>(stdv::iota(0U, size));
    auto cxbr    = stdr::to<std::vector<std::complex<f32>>>(br);
    auto cxbr_br = cxbr;
    simd_swaps<width>(reinterpret_cast<f32*>(cxbr_br.data()), size / width);

    bool equal = true;
    for (auto [i, br, br2]: stdv::zip(stdv::iota(0U), cxbr, cxbr_br)) {
        std::println("{:>4}: {:>4} -> {:3}", i, (br), (br2.real()));
        if (br2.real() != i)
            equal = false;
    }
    std::println("Equal: {}", equal);

    return 0;
}
