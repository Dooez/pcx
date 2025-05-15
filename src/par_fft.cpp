#include "pcx/fft.hpp"
namespace pcx {
namespace {
constexpr auto forward     = std::false_type{};
constexpr auto reverse     = std::true_type{};
constexpr auto normal_opts = fft_options{.pt = fft_permutation::normal};
constexpr auto shiftd_opts = fft_options{.pt = fft_permutation::shifted};
}    // namespace

}    // namespace pcx
