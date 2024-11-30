#include "pcx/include/tuple.hpp"

#include <print>
using uZ  = std::size_t;
using u64 = std::uint64_t;
template<uZ I>
using uZ_constant = std::integral_constant<uZ, I>;
constexpr struct s_t : pcx::tupi::multistage_op_base {
    [[gnu::always_inline]] auto operator()(int i) const {
        return i * 4 + 100;
    };

    template<uZ I>
    struct stage_t {
        template<typename Arg>
        auto operator()(Arg&& arg) {
            return std::forward<Arg>(arg);
        }
    };

    template<uZ I>
    static constexpr stage_t<I> stage{};
} staged;

template<>
struct s_t::stage_t<0> {
    [[gnu::always_inline]] auto operator()(int i) const {
        return pcx::tupi::intermediate_result(pcx::tupi::make_tuple(i * 2));
    };
};
template<>
struct s_t::stage_t<1> {
    [[gnu::always_inline]] auto operator()(auto i) const {
        return std::get<0>(i) + 1000;
    };
};

constexpr struct s_nr_t : pcx::tupi::multistage_op_base {
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
        return pcx::tupi::intermediate_result(pcx::tupi::make_tuple(i * 3));
    };
};
template<>
struct s_nr_t::stage_t<1> {
    [[gnu::always_inline]] void operator()(pcx::tupi::tuple<int> i) const {
        std::print("{}\n", pcx::tupi::get<0>(i) + 1);
    };
};

auto foo(std::tuple<int, int, int> x) {
    return pcx::tupi::group_invoke(staged, x);
}

int main() {
    using namespace pcx;

    static_assert(tupi::final_group_result<void>);

    auto [x0, x1, x2, x3] = tupi::group_invoke(staged, tupi::make_tuple(0, 1, 2, 3));
    tupi::group_invoke(staged_noret, tupi::make_tuple(0, 1, 2, 3));

    std::print("{} {} {} {}\n", x0, x1, x2, x3);
    return 0;
}
