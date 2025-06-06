#pragma once
#include "pcx/include/types.hpp"

#include <utility>
namespace pcx::meta {

template<typename... T>
struct types;

namespace detail_ {
template<typename T>
struct is_types : std::false_type {};
template<typename... Ts>
struct is_types<types<Ts...>> : std::true_type {};
template<uZ K, uZ I, typename... Ts>
    requires(K <= I && I <= sizeof...(Ts))
struct get_types_h;
template<uZ K, uZ I, typename T, typename... Ts>
struct get_types_h<K, I, T, Ts...> {
    using type = std::conditional_t<K == I, T, typename get_types_h<K + 1, I, Ts...>::type>;
};
template<uZ K>
struct get_types_h<K, K> {
    using type = void;
};
}    // namespace detail_

template<typename T>
concept any_types = detail_::is_types<T>::value;
namespace detail_ {
template<uZ I, any_types Ts>
struct get_type;
template<uZ I, typename... Ts>
struct get_type<I, types<Ts...>> {
    using type = get_types_h<0, I, Ts...>::type;
};
}    // namespace detail_
template<uZ I, any_types Ts>
using get_type_t = detail_::get_type<I, Ts>::type;


template<typename T>
struct t_id : public std::type_identity<T> {};

struct is_same {
    template<typename T, typename U>
    static constexpr auto operator()(t_id<T>, t_id<U>) -> bool {
        return std::is_same_v<T, U>;
    }
};


template<typename... Ts>
struct types {
    static constexpr auto reverse();
    template<typename Cmp = is_same>
    static constexpr bool unique(Cmp cmp = {});
    template<typename Pred>
    static constexpr auto filter(Pred pred = {});

    template<typename T, typename Cmp = is_same>
    static constexpr auto includes(t_id<T>, Cmp cmp = {}) -> bool {
        return (cmp(t_id<T>{}, t_id<Ts>{}) || ...);
    }
    template<uZ I>
        requires(I < sizeof...(Ts))
    static constexpr auto get(uZ_ce<I>) {
        return t_id<get_type_t<I, types>>{};
    }
    template<typename... Us>
    static constexpr auto append(types<Us...>) {
        return types<Ts..., Us...>{};
    };
    template<typename... Us>
    static constexpr auto prepend(types<Us...>) {
        return types<Us..., Ts...>{};
    };
};

}    // namespace pcx::meta
