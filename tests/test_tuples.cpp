#include "pcx/include/tuple.hpp"

#include <complex>
#include <print>
using uZ  = std::size_t;
using u64 = std::uint64_t;
int main() {
    using namespace pcx;

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
                | join
                ;

    auto proc2 = tupi::pass 
                    | [&](auto x){return tupi::make_tuple(tupi::pass | s0 | s1, x);}
                    | tupi::apply
                    | tupi::invoke
                    | s2
                    ;

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

    auto gri   = tupi::detail_::group_invoke_t{};
    auto grp_f = tupi::detail_::group_invoke_t{}(proc2);
    auto prnt  = gri([](auto x) { std::print("prnt: {}\n", x); });

    std::print("Start\n");
    // auto res = pipelined_f(10, 200);
    // proc2(10);
    // auto res = grp_f(tupi::make_tuple(200, 10));
    // auto r = proc2(10);
    // auto r = tupi::group_invoke(pipelined_f, tupi::make_tuple(10, 100), tupi::make_tuple(1, 2));
    // gri(proc2, tupi::make_tuple(10, 200));
    auto t = tupi::make_tuple(10, 200);
    prnt(tupi::make_tuple(10, 200));
    // grp_f(tupi::make_tuple(10, 200));
    // std::print("prnt: {}\n", get<0>(t));
    // std::print("prnt: {}\n", get<1>(t));
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
