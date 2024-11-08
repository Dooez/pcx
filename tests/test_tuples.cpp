#include "pcx/include/tuple.hpp"

#include <print>
using uZ  = std::size_t;
using u64 = std::uint64_t;
template<uZ I>
using uZ_constant = std::integral_constant<uZ, I>;
constexpr struct s_t : pcx::i::multi_stage_op_base<2> {
    [[gnu::always_inline]] auto operator()(int i) const {
        return i * 4 + 100;
    };

    template<uZ I>
    struct stage_t;

    template<uZ I>
    static constexpr stage_t<I> stage{};
} staged;

template<>
struct s_t::stage_t<0> {
    [[gnu::always_inline]] auto operator()(int i) const {
        return i * 2;
    };
};
template<>
struct s_t::stage_t<1> {
    [[gnu::always_inline]] auto operator()(int i) const {
        return i + 1;
    };
};

constexpr struct s_nr_t : pcx::i::multi_stage_op_base<2> {
    [[gnu::always_inline]] auto operator()(int i) const {
        std::print("{}", i * 4 + 100);
    };

    template<uZ I>
    struct stage_t;

    template<uZ I>
    static constexpr stage_t<I> stage{};
} staged_noret;

template<>
struct s_nr_t::stage_t<0> {
    [[gnu::always_inline]] auto operator()(int i) const {
        return i * 3;
    };
};
template<>
struct s_nr_t::stage_t<1> {
    [[gnu::always_inline]] void operator()(int i) const {
        std::print("{}\n", i + 1);
    };
};

auto foo(std::tuple<int, int, int> x) {
    return pcx::i::group_invoke(staged, x);
}

int main() {
    using namespace pcx;

    auto [x0, x1, x2, x3] = i::group_invoke(staged, i::make_tuple(0, 1, 2, 3));
    i::group_invoke(staged_noret, i::make_tuple(0, 1, 2, 3));

    std::print("{} {} {} {}\n", x0, x1, x2, x3);
    return 0;
}
