#pragma once
#include "pcx/include/types.hpp"

#include <utility>
namespace pcx::meta {

namespace detail_ {
template<auto...>
struct are_equal : std::false_type {};
template<auto V, auto... Vs>
    requires(std::equality_comparable_with<decltype(V), decltype(Vs)> && ...)
struct are_equal<V, Vs...> {
    static constexpr bool value = ((V == Vs) && ...) && are_equal<Vs...>::value;
};
template<auto V>
struct are_equal<V> : std::true_type {};

template<auto...>
struct are_unique : std::true_type {};
template<auto V, auto... Vs>
struct are_unique<V, Vs...> {
    static constexpr bool value = (!are_equal<V, Vs>::value && ...) && are_unique<Vs...>::value;
};
}    // namespace detail_

template<auto... Vs>
concept equal_values = detail_::are_equal<Vs...>::value;
template<auto... Vs>
concept unique_values = detail_::are_unique<Vs...>::value;

namespace detail_ {
template<auto... Vs>
struct generic_value_sequence {};
template<auto... Vs>
struct value_sequence {
    using type = generic_value_sequence<Vs...>;
};
template<std::integral T, T I>
struct value_sequence<I> {
    using type = std::integral_constant<T, I>;
};
template<typename T, T I, typename... Ts, Ts... Is>
    requires std::integral<T> && (std::same_as<T, Ts> && ...)
struct value_sequence<I, Is...> {
    using type = std::integer_sequence<T, I, Is...>;
};
}    // namespace detail_

template<auto... Vs>
using value_sequence = detail_::value_sequence<Vs...>::type;

template<auto... Vs>
using val_seq = value_sequence<Vs...>;

namespace detail_ {
template<typename>
struct is_value_seq : std::false_type {};
template<auto... Vs>
struct is_value_seq<value_sequence<Vs...>> : std::true_type {};

template<typename>
struct is_value_seq_of_unique;
template<auto... Vs>
struct is_value_seq_of_unique<value_sequence<Vs...>> {
    static constexpr bool value = unique_values<Vs...>;
};

template<typename, auto...>
struct expand_value_seq;
template<auto... Vs1, auto... Vs2>
struct expand_value_seq<value_sequence<Vs1...>, Vs2...> {
    using type = value_sequence<Vs1..., Vs2...>;
};
template<typename, typename>
struct concat_value_seq;
template<auto... Vs1, auto... Vs2>
struct concat_value_seq<value_sequence<Vs1...>, value_sequence<Vs2...>> {
    using type = value_sequence<Vs1..., Vs2...>;
};
template<typename S>
struct reverse_value_seq {
    using type = S;
};
template<auto V, auto... Vs>
struct reverse_value_seq<value_sequence<V, Vs...>> {
    using type = typename expand_value_seq<typename reverse_value_seq<value_sequence<Vs...>>::type, V>::type;
};

template<typename>
struct is_index_seq : std::false_type {};
template<uZ... Is>
struct is_index_seq<std::index_sequence<Is...>> : std::true_type {};

template<typename>
struct value_to_index_seq;
template<uZ... Is>
struct value_to_index_seq<value_sequence<Is...>> {
    using type = std::index_sequence<Is...>;
};
template<typename S>
struct index_to_value_seq;
template<uZ... Is>
struct index_to_value_seq<std::index_sequence<Is...>> {
    using type = value_sequence<Is...>;
};

}    // namespace detail_
template<typename T>
concept any_value_sequence = detail_::is_value_seq<T>::value;

template<typename T>
concept any_index_sequence = detail_::is_index_seq<T>::value;

template<any_value_sequence S>
using value_to_index_sequence = typename detail_::value_to_index_seq<S>::type;

template<any_index_sequence S>
using index_to_value_sequence = typename detail_::index_to_value_seq<S>::type;

template<typename T>
concept value_sequence_of_unique = detail_::is_value_seq_of_unique<T>::value;

template<any_value_sequence S, auto... Vs>
using expand_value_sequence = typename detail_::expand_value_seq<S, Vs...>::type;

template<any_value_sequence S1, any_value_sequence S2>
using concat_value_sequences = typename detail_::concat_value_seq<S1, S2>::type;

template<any_value_sequence S>
using reverse_value_sequence = typename detail_::reverse_value_seq<S>::type;

template<any_index_sequence S>
using reverse_index_sequence =
    value_to_index_sequence<typename detail_::reverse_value_seq<index_to_value_sequence<S>>::type>;

template<typename... T>
struct types {};
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

}    // namespace pcx::meta
