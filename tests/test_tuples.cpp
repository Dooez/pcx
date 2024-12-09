#include "pcx/include/tuple.hpp"

#include <complex>
#include <print>
using uZ  = std::size_t;
using u64 = std::uint64_t;
// template<uZ I>
// using uZ_constant = std::integral_constant<uZ, I>;
// constexpr struct s_t : pcx::tupi::compound_op_base {
//     [[gnu::always_inline]] auto operator()(int i) const {
//         return i * 4 + 100;
//     };
//     [[gnu::always_inline]] auto operator()(std::complex<float> i) const {
//         return i * 4.F + 100.F;
//     };
//
//     template<uZ I>
//     struct stage_t {
//         // template<typename Arg>
//         // auto operator()(Arg&& arg) {
//         //     std::print("Arg stage: {}\n", I);
//         //     return std::forward<Arg>(arg);
//         // }
//         auto operator()(float i) const
//             requires(I == 0)
//         {
//             std::print("float stage: {}\n", I);
//             return i * std::sqrt(2.F);
//         }
//         auto operator()(int i) const
//             requires(I == 0)
//         {
//             std::print("int stage: {}\n", I);
//             return pcx::tupi::make_interim(i * 2);
//         };
//         auto operator()(int i) const
//             requires(I == 1)
//         {
//             std::print("int stage: {}\n", I);
//             return i + 1000;
//         };
//
//         auto operator()(std::complex<float> i) const
//             requires(I == 0)
//         {
//             std::print("cx float stage: {}\n", I);
//             return pcx::tupi::make_interim(i * std::exp(std::complex(0.F, std::numbers::pi_v<float> / 4.F)));
//         }
//         auto operator()(std::complex<float> cx) const
//             requires(I == 1)
//         {
//             std::print("cx float stage: {}\n", I);
//             return pcx::tupi::make_interim(cx * 100.f);
//         }
//         auto operator()(std::complex<float> cx) const
//             requires(I == 2)
//         {
//             std::print("cx float stage: {}\n", I);
//             return pcx::tupi::make_interim(cx * 100.f);
//         }
//         auto operator()(std::complex<float> cx) const
//             requires(I == 3)
//         {
//             std::print("cx float stage: {}\n", I);
//             return cx;
//         }
//     };
//
//     template<uZ I>
//     static constexpr stage_t<I> stage{};
//     template<uZ I>
//     constexpr friend auto get_stage(const s_t&) {
//         return stage_t<I>{};
//     }
// } staged;
//
// constexpr struct s_nr_t : pcx::tupi::compound_op_base {
//     [[gnu::always_inline]] auto operator()(int i) const {
//         std::print("{}", i * 4 + 100);
//     };
//
//     template<uZ I>
//     struct stage_t;
//
//     template<uZ I>
//     static constexpr stage_t<I> stage{};
//
//     template<uZ I>
//     constexpr friend auto get_stage(const s_nr_t&) {
//         return stage_t<I>{};
//     }
// } staged_noret;
//
// template<>
// struct s_nr_t::stage_t<0> {
//     [[gnu::always_inline]] auto operator()(int i) const {
//         std::print("Stage 0, v: {}.\n", i);
//         return pcx::tupi::make_interim(i * 3);
//     };
// };
// template<>
// struct s_nr_t::stage_t<1> {
//     [[gnu::always_inline]] void operator()(int i) const {
//         std::print("{}\n", i + 1);
//     };
// };

// auto foo(std::tuple<int, int, int> x) {
//     return pcx::tupi::group_invoke(staged, x);
// }

int main() {
    using namespace pcx;

    namespace td = pcx::tupi::detail_;


    auto s0 = [](auto x) {
        std::print("Stage 0. x: {}.\n", x);
        return x + 1;
    };
    auto s1 = [](auto x) {
        std::print("Stage 1. x: {}.\n", x);
        return x + 1;
    };
    auto s2 = [](auto x) {
        std::print("Stage 2. x: {}.\n", x);
        return x + 1;
    };
    auto split = [](auto x) {
        std::print("Splitting value. {}.\n", x);
        return tupi::make_tuple(x, x * 2);
    };
    auto join = [](auto x, auto x2) {
        std::print("Joining value. {} {}.\n", x, x2);
        return x + x2;
    };

    // clang-format off
    auto proc = tupi::pass
                | s0 
                | s1 
                | s2 
                | split 
                | tupi::apply 
                | join;

    auto proc2 = tupi::pass 
                | [&](auto x){return tupi::make_tuple(tupi::pass | s0 | s1, x);}
                | tupi::apply
                | tupi::invoke
                | s2;

    // clang-format on
    auto comb = [](auto x, auto y) {
        std::print("Combining 2 values. {}+{}.\n", x, y);
        return x + y;
    };
    auto post = [](auto x) {
        std::print("Post combine. {}.\n", x);
        return x + 1;
    };
    auto pipelined_f = tupi::distribute                  //
                       | tupi::pipeline(proc2, proc2)    //
                       | tupi::apply                     //
                       | comb                            //
                       | post                            //
        ;

    std::print("Start\n");
    auto res = pipelined_f(10, 200);
    // auto r = proc2(10);
    // auto r = tupi::group_invoke(pipelined_f, tupi::make_tuple(10, 100), tupi::make_tuple(1, 2));
    // auto r = tupi::group_invoke(tupi::pass | s0 | s1, tupi::make_tuple(10));
    // std::print("{}.\n", res);
    std::print("End\n");

    // static_assert(tupi::final_group_result<void>);
    // auto [x0, x1, x2, x3, cx0] =
    //     tupi::group_invoke(staged, tupi::make_tuple(0, 1, 2, 3, std::complex<float>(1, 0)));
    // tupi::group_invoke(staged_noret, tupi::make_tuple(0, 1, 2, 3));
    //
    // std::print("{} {} {} {}\n", x0, x1, x2, x3);
    return 0;
}
