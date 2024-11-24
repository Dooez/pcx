#include "pcx/include/simd/common.hpp"
#include "pcx/include/simd/math.hpp"

#include <format>
#include <print>
using namespace pcx;

auto operator+(simd::any_cx_vec auto lhs, simd::any_cx_vec auto rhs) {
    return simd::add(lhs, rhs);
}
auto operator-(simd::any_cx_vec auto lhs, simd::any_cx_vec auto rhs) {
    return simd::sub(lhs, rhs);
}
auto operator*(simd::any_cx_vec auto lhs, simd::any_cx_vec auto rhs) {
    return simd::mul(lhs, rhs);
}
auto operator/(simd::any_cx_vec auto lhs, simd::any_cx_vec auto rhs) {
    return simd::div(lhs, rhs);
}
template<uZ I, typename T>
auto mul_by_j(std::complex<T> v) {
    if constexpr (I % 4 == 0) {
        return v;
    } else if constexpr (I % 4 == 1) {
        return v * std::complex<T>(0, 1);
    } else if constexpr (I % 4 == 2) {
        return -v;
    } else if constexpr (I % 4 == 3) {
        return v * std::complex<T>(0, -1);
    }
}


template<typename F, typename Fd>
class foo_op {
    F  m_foo;
    Fd m_descr;

public:
    constexpr foo_op(const F& mod, const Fd& fmt)
    : m_foo(mod)
    , m_descr(fmt) {};

    auto operator()(auto... args) const {
        return m_foo(args...);
    }

    constexpr auto operator|(auto op) const {
        auto x = [op, foo = m_foo](auto... args) { return op(foo(args...)); };
        auto y = [op, descr = m_descr](auto... text_args) { return op.descr(descr(text_args...)); };
        return foo_op<decltype(x), decltype(y)>(x, y);
    }

    auto descr(auto... text_args) const {
        return m_descr(text_args...);
    }
};

constexpr auto base_unary_ops = tupi::make_tuple(
    foo_op([](auto v) { return conj(v); }, [](auto descr) { return std::format("conj( {} )", descr); }),
    foo_op([](auto v) { return mul_by_j<1>(v); }, [](auto descr) { return std::format("1j•{}", descr); }),
    foo_op([](auto v) { return mul_by_j<2>(v); }, [](auto descr) { return std::format("2j•{}", descr); }),
    foo_op([](auto v) { return mul_by_j<3>(v); }, [](auto descr) { return std::format("3j•{}", descr); }));

constexpr auto base_binary_ops = tupi::make_tuple(    //
    foo_op([](auto lhs, auto rhs) { return lhs + rhs; },
           [](auto ltxt, auto rtxt) { return std::format("( {} + {} )", ltxt, rtxt); }),
    foo_op([](auto lhs, auto rhs) { return lhs - rhs; },
           [](auto ltxt, auto rtxt) { return std::format("( {} - {} )", ltxt, rtxt); }),
    foo_op([](auto lhs, auto rhs) { return lhs * rhs; },
           [](auto ltxt, auto rtxt) { return std::format("( {} * {} )", ltxt, rtxt); }),
    foo_op([](auto lhs, auto rhs) { return lhs / rhs; },
           [](auto ltxt, auto rtxt) { return std::format("( {} / {} )", ltxt, rtxt); }));

template<typename T>
auto cxbroad(std::complex<T> a) {
    auto v = simd::cxbroadcast<1>(reinterpret_cast<T*>(&a));
    return simd::repack<decltype(v)::width()>(v);
}
auto cxval(simd::any_cx_vec auto v) {
    auto tmp = std::array<std::complex<typename decltype(v)::real_type>, decltype(v)::width()>();

    simd::cxstore<1>(reinterpret_cast<f32*>(tmp.data()), simd::repack<1>(simd::evaluate(v)));
    return tmp[0];
}

template<uZ N>
constexpr auto apply_unary_mods(auto op) {
    if constexpr (N == 0) {
        return tupi::make_tuple(op);
    } else {
        constexpr auto mod_seq   = std::make_index_sequence<tupi::tuple_size_v<decltype(base_unary_ops)>>{};
        constexpr auto moded_ops = []<uZ... Is>(auto op, std::index_sequence<Is...>) {
            return tupi::make_tuple((op | tupi::get<Is>(base_unary_ops))...);
        }(op, mod_seq);
        constexpr auto n_ops = tupi::tuple_size_v<decltype(moded_ops)>;
        return [=]<uZ... Is>(std::index_sequence<Is...>) {
            return tupi::tuple_cat(moded_ops, apply_unary_mods<N - 1>(tupi::get<Is>(moded_ops))...);
        }(std::make_index_sequence<n_ops>{});
    }
}

constexpr auto unary_ops = apply_unary_mods<2>(foo_op([](auto x) { return x; },    //
                                                      [](auto x) { return std::format("{}", x); }));
constexpr auto n_unary   = tupi::tuple_size_v<decltype(unary_ops)>;

constexpr auto binary_ops = [] {
    return []<uZ... Is>(std::index_sequence<Is...>) {
        return tupi::tuple_cat(apply_unary_mods<1>(tupi::get<Is>(base_binary_ops))...);
    }(std::make_index_sequence<tupi::tuple_size_v<decltype(base_binary_ops)>>{});
}();
constexpr auto n_binary = tupi::tuple_size_v<decltype(binary_ops)>;

uZ cmp_count = 0;
template<typename T>
auto compare_std_simd(std::complex<T> l, std::complex<T> r, auto l_op, auto r_op, auto op) {
    auto std_res  = op(l_op(l), r_op(r));
    auto simd_res = cxval(op(l_op(cxbroad(l)), r_op(cxbroad(r))));
    auto descr    = op.descr(l_op.descr("z1"), r_op.descr("z2"));
    if (std_res != simd_res) {
        auto diff = std_res - simd_res;
        std::print("{}. Not equal. {}. Diff: {}%.\n",
                   cmp_count++,
                   descr,
                   std::abs(diff) / std::abs(std_res) * 100);
        if (abs(diff) > abs(std_res) * 0.001)
            return false;
        return true;
    }
    cmp_count++;
    return true;
}

auto check_all_ops(auto l, auto r) {
    return [=]<uZ... I>(std::index_sequence<I...>) {
        return ([=]<uZ... Il>(auto op, std::index_sequence<Il...>) {
            return ([=]<uZ... Ir>(auto op, auto l_op, std::index_sequence<Ir...>) {
                return (compare_std_simd(l, r, l_op, tupi::get<Ir>(unary_ops), op) && ...);
            }(op, tupi::get<Il>(unary_ops), std::make_index_sequence<n_unary>{})
                    && ...);
        }(tupi::get<I>(binary_ops), std::make_index_sequence<n_unary>{})
                && ...);
    }(std::make_index_sequence<n_binary>{});
}

int main() {
    check_all_ops(std::complex<f32>(-1, 1), std::complex<f32>(-200, 123));
    return 0;
}
