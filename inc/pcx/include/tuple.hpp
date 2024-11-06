#include "pcx/include/types.hpp"

#include <tuple>

#define PCX_AINLINE [[gnu::always_inline, clang::always_inline]] inline

namespace pcx::h {
namespace detail_ {
template<typename... Ts>
class ituple_impl {};
template<typename T, typename... Ts>
class ituple_impl<T, Ts...> : public ituple_impl<Ts...> {
    template<uZ I, typename... Us>
    friend struct getter;

    using base = ituple_impl<Ts...>;

public:
    template<typename U, typename... Args>
        requires(!std::same_as<std::remove_cvref_t<U>, ituple_impl>)
    PCX_AINLINE explicit ituple_impl(U&& val, Args&&... args)
    : base(static_cast<Args&&>(args)...)
    , v(static_cast<U&&>(val)){};

    PCX_AINLINE ituple_impl()                   = default;
    PCX_AINLINE ituple_impl(ituple_impl&&)      = default;
    PCX_AINLINE ituple_impl(const ituple_impl&) = default;

    PCX_AINLINE ituple_impl& operator=(ituple_impl&&)      = default;
    PCX_AINLINE ituple_impl& operator=(const ituple_impl&) = default;
    PCX_AINLINE ~ituple_impl()                             = default;

private:
    T v;
};

template<uZ I, typename... Ts>
struct ituple_base {
    using type = void;
};
template<uZ I, typename T, typename... Ts>
struct ituple_base<I, T, Ts...> {
    using type = std::conditional_t<I == 0, ituple_impl<T, Ts...>, typename ituple_base<I - 1, Ts...>::type>;
};

template<typename T, typename U>
struct reference_like {
    using c_t  = std::conditional_t<std::is_const_v<U>, const T, T>;
    using type = std::conditional_t<std::is_rvalue_reference_v<U>,
                                    c_t&&,
                                    std::conditional_t<std::is_lvalue_reference_v<U>,    //
                                                       c_t&,
                                                       c_t>>;
};
template<typename T, typename U>
using reference_like_t = reference_like<T, U>::type;

template<uZ I, typename... Ts>
struct getter {
    PCX_AINLINE static auto get(auto&& tup) {
        using base_t   = ituple_base<I, Ts...>::type;
        using base_ref = reference_like_t<base_t, decltype(tup)>;
        auto base      = static_cast<base_ref>(tup);
        using elem_ref = reference_like_t<decltype(base.v), base_ref>;
        return static_cast<elem_ref>(base.v);
    }
};

}    // namespace detail_


template<typename... Ts>
class ituple : public detail_::ituple_impl<Ts...> {
    using base = detail_::ituple_impl<Ts...>;

public:
    template<typename... Args>
        requires(!std::same_as<std::remove_cvref_t<Args>, ituple> && ...)
    PCX_AINLINE explicit ituple(Args&&... args)
    : base(static_cast<Args&&>(args)...){};

    PCX_AINLINE ituple()              = default;
    PCX_AINLINE ituple(ituple&&)      = default;
    PCX_AINLINE ituple(const ituple&) = default;

    PCX_AINLINE ituple& operator=(ituple&&)      = default;
    PCX_AINLINE ituple& operator=(const ituple&) = default;
    PCX_AINLINE ~ituple()                        = default;

    template<uZ I>
        requires(I < sizeof...(Ts))
    PCX_AINLINE friend auto get(const ituple& tup) {
        return detail_::getter<I, Ts...>::get(tup);
    }
    template<uZ I>
        requires(I < sizeof...(Ts))
    PCX_AINLINE friend auto get(ituple& tup) {
        return detail_::getter<I, Ts...>::get(tup);
    }
    template<uZ I>
        requires(I < sizeof...(Ts))
    PCX_AINLINE friend auto get(ituple&& tup) {
        return detail_::getter<I, Ts...>::get(std::move(tup));
    }

private:
};

template<typename... Args>
PCX_AINLINE auto make_ituple(Args&&... args) {
    using ituple_t = ituple<std::remove_cvref_t<Args>...>;
    return ituple_t(static_cast<Args&&>(args)...);
}

namespace detail_ {
template<uZ, uZ, typename...>
struct index_into_types_h;
template<uZ K, uZ I, typename T, typename... Ts>
struct index_into_types_h<K, I, T, Ts...> {
    using type = std::conditional_t<K == I, T, typename index_into_types_h<K + 1, I, Ts...>::type>;
};
template<uZ K, typename T>
struct index_into_types_h<K, K, T> {
    using type = T;
};
template<uZ I, typename... Ts>
    requires(I < sizeof...(Ts))
struct index_into_types {
    using type = index_into_types_h<0, I, Ts...>::type;
};

};    // namespace detail_
}    // namespace pcx::h

template<typename... Ts>
struct std::tuple_size<pcx::h::ituple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};
template<std::size_t I, typename... Ts>
struct std::tuple_element<I, pcx::h::ituple<Ts...>> {
    using type = pcx::h::detail_::index_into_types<I, Ts...>::type;
};

namespace pcx::i {
template<typename... Ts>
using tuple = std::tuple<Ts...>;
template<typename... Args>
PCX_AINLINE auto make_tuple(Args&&... args) {
    return std::make_tuple(std::forward<Args>(args)...);
}
template<typename... Ts>
PCX_AINLINE auto tuple_cat(Ts&&... tuples) {
    return std::tuple_cat(std::forward<Ts>(tuples)...);
}
template<uZ I, typename T>
PCX_AINLINE auto get(T&& tuple) {
    return std::get<I>(std::forward<T>(tuple));
}

using std::tuple_element;
using std::tuple_element_t;
using std::tuple_size;
using std::tuple_size_v;

template<uZ N>
struct multi_stage_op {
    static constexpr uZ stage_count = N;
};

namespace detail_ {
template<typename>
struct is_tuple : std::false_type {};
template<typename... Ts>
struct is_tuple<tuple<Ts...>> : std::true_type {};
template<typename T>
inline constexpr auto is_tuple_v = is_tuple<T>::value;

template<typename...>
struct are_same_size_tuples : std::false_type {};
template<typename T>
    requires(is_tuple_v<T>)
struct are_same_size_tuples<T> : std::true_type {};
template<typename T, typename... Ts>
    requires(is_tuple_v<T> && (is_tuple_v<Ts> && ...))
struct are_same_size_tuples<T, Ts...> {
    static constexpr bool value = ((tuple_size_v<T> == tuple_size_v<Ts>) && ...);
};
template<typename... T>
inline constexpr auto are_same_size_tuples_v = are_same_size_tuples<T...>::value;

template<uZ I, typename F, typename... Args>
struct is_group_invokable_h {
    static constexpr bool value =
        std::invocable<F, tuple_element_t<I, Args>...> && is_group_invokable_h<I - 1, F, Args...>::value;
};
template<typename F, typename... Args>
struct is_group_invokable_h<0, F, Args...> {
    static constexpr bool value = std::invocable<F, tuple_element_t<0, Args>...>;
};
template<typename F, typename... Args>
struct is_group_invokable {
    static constexpr bool value = is_group_invokable_h<(..., tuple_size_v<Args>)-1, F, Args...>::value;
};

}    // namespace detail_

template<typename T>
concept any_tuple = detail_::is_tuple_v<T>;
template<typename... Ts>
concept same_size_tuples = (any_tuple<Ts> && ...) && detail_::are_same_size_tuples_v<Ts...>;
template<typename F, typename... Args>
concept group_invokable = same_size_tuples<Args...> && detail_::is_group_invokable<F, Args...>::value;

template<typename F, typename... Args>
    requires group_invokable<F, Args...>
constexpr auto group_invoke(F&& f, Args&&... args);


}    // namespace pcx::i

#undef PCX_AINLINE
