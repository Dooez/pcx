#include "pcx/fft.hpp"

#include <print>

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
namespace stdv = std::views;
namespace stdr = std::ranges;

int main() {
    size_t fft_size = 256;

    auto fft = pcx::fft_plan<float>(fft_size);
    auto dat = std::vector<std::complex<float>>(fft_size);
    for (auto [i, v]: stdv::enumerate(dat)) {
        v = std::exp(std::complex<float>{0, static_cast<float>(i)} * 2.F * std::numbers::pi_v<float> / 2.F);
    }
    fft.fft(dat);
    for (auto v: dat) {
        std::print("{:.2f} ", abs(v));
    }

    return 0;
}
