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

int main() {
    uZ   size = 2048 * 32;
    auto idx  = stdr::to<std::vector<uZ>>(stdv::iota(0U, size));
    auto br   = idx;
    for (auto& v: br) {
        v = reverse_bit_order(v, size);
    }
    constexpr uZ width    = 16;
    constexpr uZ coh_size = 2048;
    auto         br2      = br;
    do_swaps<width>(br2);
    auto br3         = br2;
    auto stride      = size / width;
    auto close_swaps = get_close_swaps<width>(stride / width, coh_size / width);
    auto far_swaps   = get_far_swaps<width>(stride / width, coh_size / width);

    std::println("n vecs:{}", size / width);
    std::println("n coh vecs:{}", coh_size / width);
    std::print("close swaps: ");
    for (auto v: close_swaps) {
        std::print("{} ", v);
    }
    std::println();
    std::print("far swaps: ");
    for (auto v: far_swaps) {
        std::print("{} ", v);
    }
    std::println();


    // for (uZ i_str: stdv::iota(0U, width)) {
    //     for (uZ i_coh: stdv::iota(0U, std::max(stride / coh_size, 1UZ))) {
    //         auto x = 0;
    //         for (auto [l, r]: stdv::adjacent<2>(close_swaps) | stdv::stride(2)) {
    //             // std::println("{} <-> {}", l, r);
    //             for (uZ i: stdv::iota(0U, width)) {
    //                 using std::swap;
    //                 swap(br3[i_str * stride + i_coh * coh_size + l * width + i],
    //                      br3[i_str * stride + i_coh * coh_size + r * width + i]);
    //             }
    //         }
    //     }
    // }
    // auto br4 = br3;
    // for (uZ i_str: stdv::iota(0U, width)) {
    //     for (auto [l, r]: stdv::adjacent<2>(far_swaps) | stdv::stride(2)) {
    //         // std::println("{} <-> {}", l, r);
    //         for (uZ i: stdv::iota(0U, width)) {
    //             using std::swap;
    //             swap(br4[i_str * stride + l * width + i],    //
    //                  br4[i_str * stride + r * width + i]);
    //         }
    //     }
    // }

    auto smidx = stdr::to<std::vector<uZ>>(stdv::iota(0U, stride / width));
    auto sm_br = smidx;
    for (auto& v: sm_br) {
        v = reverse_bit_order(v, stride / width);
    }

    auto sm_stride = stride / width;
    auto sm_coh    = coh_size / width;

    auto sm_br2 = sm_br;
    // for (uZ i_str: stdv::iota(0U, width)) {
    // for (uZ i_coh: stdv::iota(0U, std::max(sm_stride / sm_coh, 1UZ))) {
    // for (uZ i_coh: stdv::iota(0U, 1U)) {
    for (auto [l, r]: stdv::adjacent<2>(close_swaps) | stdv::stride(2)) {
        // std::println("{} <-> {}", l, r);
        using std::swap;
        swap(sm_br2[l], sm_br2[r]);
    }
    // }
    // }
    auto sm_br3 = sm_br2;
    for (auto [l, r]: stdv::adjacent<2>(far_swaps) | stdv::stride(2)) {
        // std::println("{} <-> {}", l, r);
        using std::swap;
        swap(sm_br3[l], sm_br3[r]);
    }


    // for (auto [i, br, br2, br3, br4]: stdv::zip(idx, br, br2, br3, br4)) {
    //     // std::println("{:>4}: {:>4} -> {:>4} -> {}", i, br, br2, br3);
    //     if (i != br4) {
    //         std::println("{:>4}: {:>4} -> {:>4} -> {:4} -> {}",
    //                      i / width,    //
    //                      br / width,
    //                      br2 / width,
    //                      br3 / width,
    //                      br4 / width);
    //         break;
    //     }
    // }
    for (auto [i, br, br2, br3]: stdv::zip(smidx, sm_br, sm_br2, sm_br3)) {
        std::println("{:>4}: {:>4} -> {:>4} -> {}", i, br, br2, br3);
        // if (i != br3) {
        //     std::println("{:>4}: {:>4} -> {:>4} -> {:4}", i, br, br2, br3);
        //     // break;
        // }
    }
    if (stdr::equal(smidx, sm_br3)) {
        std::println("Equal");
    } else {
        std::println("Not equal");
    }

    return 0;
}
