#pragma once
#include "pcx/include/meta.hpp"

#include <tuple>
#include <type_traits>

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

// Tuple interface
namespace pcx::tupi {

template<typename... Ts>
using tuple = std::tuple<Ts...>;

using std::tuple_element;
using std::tuple_element_t;
using std::tuple_size;
using std::tuple_size_v;

namespace detail_ {
template<typename>
struct is_tuple : std::false_type {};
template<typename... Ts>
struct is_tuple<tuple<Ts...>> : std::true_type {};
template<typename T>
inline constexpr auto is_tuple_v = is_tuple<T>::value;
}    // namespace detail_
template<typename T>
concept any_tuple = detail_::is_tuple_v<T>;
template<typename T>
concept tuple_like = requires(T v) {
    { tuple_size_v<T> } -> std::common_with<uZ>;
    {
        []<uZ... Is>(std::index_sequence<Is...>, auto&& v) {
            ((void)get<Is>(v), ...);
        }(std::make_index_sequence<tuple_size_v<T>>{}, v)
    };
};
template<typename T>
    requires tuple_like<std::remove_cvref_t<T>>
using index_sequence_for_tuple = std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>;

template<typename... Args>
PCX_AINLINE constexpr auto make_tuple(Args&&... args) {
    return std::make_tuple(std::forward<Args>(args)...);
}
template<typename... Ts>
PCX_AINLINE constexpr auto tuple_cat(Ts&&... tuples) {
    return std::tuple_cat(std::forward<Ts>(tuples)...);
}
namespace detail_ {
template<uZ I>
struct get_t {
    template<typename T>
    PCX_AINLINE constexpr static auto operator()(T&& v) {
        return std::get<I>(std::forward<T>(v));
    }
};
}    // namespace detail_
template<uZ I>
inline constexpr auto get = detail_::get_t<I>{};

template<uZ TupleSize, typename T>
PCX_AINLINE constexpr auto make_broadcast_tuple(T&& v) {
    return []<uZ... Is> PCX_LAINLINE(auto&& v, std::index_sequence<Is...>) {
        return make_tuple((void(Is), v)...);
    }(static_cast<T&&>(v), std::make_index_sequence<TupleSize>{});
}
template<typename T>
PCX_AINLINE auto make_flat_tuple(T&& tuple) {
    if constexpr (any_tuple<std::remove_cvref_t<T>>) {
        return []<uZ... Is, typename U> PCX_LAINLINE(std::index_sequence<Is...>, U&& tuple) {
            return tuple_cat(make_flat_tuple(get<Is>(std::forward<U>(tuple)))...);
        }(index_sequence_for_tuple<T>{}, std::forward<T>(tuple));
    } else {
        return make_tuple(tuple);
    }
}
template<typename... Ts>
PCX_AINLINE auto make_flat_tuple(Ts&&... args) {
    return tuple_cat(make_flat_tuple(std::forward<Ts>(args))...);
}
namespace detail_ {
template<typename T>
concept tuple_like = requires(T v) {
    { tuple_size_v<T> } -> std::common_with<uZ>;
    {
        []<uZ... Is>(std::index_sequence<Is...>, auto&& v) {
            ((void)get<Is>(v), ...);
        }(std::make_index_sequence<tuple_size_v<T>>{}, v)
    };
};

template<typename F>
struct to_apply;
struct apply_t {
    template<typename F, typename Tup>
        requires tuple_like<std::remove_cvref_t<Tup>>
    static auto operator()(F&& f, Tup&& arg) -> decltype(auto) {
        return []<uZ... Is>(F&& f, Tup&& arg, std::index_sequence<Is...>) -> decltype(auto) {
            return f(get<Is>(std::forward<Tup>(arg))...);
        }(std::forward<F>(f), std::forward<Tup>(arg), index_sequence_for_tuple<Tup>{});
    };
    template<typename F>
    constexpr friend auto operator|(F&& f, const apply_t&) {
        return to_apply<F>{std::forward<F>(f)};
    }
    template<typename F>
    constexpr auto operator|(F&& f) const {
        return [f = std::forward<F>(f)]<typename Tup>(Tup&& arg) -> decltype(auto)
                   requires tuple_like<std::remove_cvref_t<Tup>>
        { return apply_t{}(f, std::forward<Tup>(arg)); };
    };
};
template<typename F>
struct to_apply {
    template<typename AF, typename G>
    constexpr auto operator|(this AF&& apply_f, G&& g) {
        return std::forward_like<AF>(apply_f.op) |
                   [g = std::forward<G>(g)]<typename Tup>(Tup&& arg) -> decltype(auto)
                   requires tuple_like<std::remove_cvref_t<Tup>>
        { return apply_t{}(g, std::forward<Tup>(arg)); };
    };
    F op;
};
}    // namespace detail_
inline constexpr auto apply = detail_::apply_t{};

namespace detail_ {
template<typename T, uZ I>
using broadcast_type_t = T;
template<typename T, meta::any_index_sequence Is>
struct broadcast_tuple_impl;
template<typename T, uZ... Is>
struct broadcast_tuple_impl<T, std::index_sequence<Is...>> {
    using type = tuple<broadcast_type_t<T, Is>...>;
};
template<typename T, uZ TupleSize>
struct broadcast_tuple {
    using type = broadcast_tuple_impl<T, std::make_index_sequence<TupleSize>>::type;
};
}    // namespace detail_
template<typename T, uZ TupleSize>
using broadcast_tuple_t = detail_::broadcast_tuple<T, TupleSize>::type;

namespace detail_ {
template<uZ, typename>
struct get_element;
template<uZ I, typename... Ts>
struct get_element<I, tuple<Ts...>> {
    using type = tuple_element_t<I, tuple<Ts...>>;
};
template<uZ I, typename... Ts>
struct get_element<I, tuple<Ts...>&> {
    using type = tuple_element_t<I, tuple<Ts...>>&;
};
template<uZ I, typename... Ts>
struct get_element<I, tuple<Ts...>&&> {
    using type = tuple_element_t<I, tuple<Ts...>>&&;
};
template<uZ I, typename... Ts>
struct get_element<I, const tuple<Ts...>&> {
    using type = tuple_element_t<I, const tuple<Ts...>>&;
};
template<uZ I, typename... Ts>
struct get_element<I, const tuple<Ts...>&&> {
    using type = tuple_element_t<I, const tuple<Ts...>>&&;
};
template<uZ I, typename T>
using get_element_t = detail_::get_element<I, T>::type;

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
struct is_group_invocable_h {
    static constexpr bool value =
        std::invocable<F, get_element_t<I, Args>...> && is_group_invocable_h<I - 1, F, Args...>::value;
};
template<typename F, typename... Args>
struct is_group_invocable_h<0, F, Args...> {
    static constexpr bool value = std::invocable<F, get_element_t<0, Args>...>;
};
template<typename F, typename... Args>
struct is_group_invocable {
    static constexpr bool value =
        is_group_invocable_h<(..., tuple_size_v<std::remove_cvref_t<Args>>)-1, F, Args...>::value;
};

template<uZ I, typename F, typename... Args>
struct has_group_invoke_result_h {
    static constexpr bool value = !std::same_as<std::invoke_result_t<F, get_element_t<I, Args>...>, void>
                                  && has_group_invoke_result_h<I - 1, F, Args...>::value;
};
template<typename F, typename... Args>
struct has_group_invoke_result_h<0, F, Args...> {
    static constexpr bool value = !std::same_as<std::invoke_result_t<F, get_element_t<0, Args>...>, void>;
};
template<typename F, typename... Args>
struct has_group_invoke_result {
    static constexpr bool value =
        has_group_invoke_result_h<(..., tuple_size_v<std::remove_cvref_t<Args>>)-1, F, Args...>::value;
};

struct interim_result_base {};
struct compound_op_base {};
}    // namespace detail_

using detail_::compound_op_base;
template<typename T>
concept compound_op = std::derived_from<T, detail_::compound_op_base>;
template<typename... Ts>
struct interim_result
: public tuple<Ts...>
, public detail_::interim_result_base {
    using tuple<Ts...>::tuple;
};
}    // namespace pcx::tupi

template<typename... Ts>
struct std::tuple_size<pcx::tupi::interim_result<Ts...>>
: std::integral_constant<std::size_t, sizeof...(Ts)> {};
template<std::size_t I, typename... Ts>
struct std::tuple_element<I, pcx::tupi::interim_result<Ts...>> {
    using type = pcx::h::detail_::index_into_types<I, Ts...>::type;
};

namespace pcx::tupi {

template<typename... Args>
PCX_AINLINE auto make_interim(Args&&... args) {
    using res_t = interim_result<std::remove_cvref_t<Args>...>;
    return res_t(std::forward<Args>(args)...);
}
template<typename T>
concept final_result = !std::derived_from<T, detail_::interim_result_base>;

namespace detail_ {
template<any_tuple T>
struct is_final_tuple;
template<typename... Ts>
struct is_final_tuple<tuple<Ts...>> {
    static constexpr bool value = (final_result<Ts> && ...);
};
template<any_tuple T>
inline constexpr auto is_final_tuple_v = is_final_tuple<T>::value;
}    // namespace detail_
template<typename T>
concept final_group_result = std::same_as<T, void> || any_tuple<T> && detail_::is_final_tuple_v<T>;

template<typename... Ts>
concept same_size_tuples = (any_tuple<Ts> && ...) && detail_::are_same_size_tuples_v<Ts...>;
template<typename F, typename... Args>
concept group_invocable =
    same_size_tuples<std::remove_cvref_t<Args>...> && detail_::is_group_invocable<F, Args...>::value;
template<typename F, typename... Args>
concept group_invocable_with_result = group_invocable<F, Args...>    //
                                      && detail_::has_group_invoke_result<F, Args...>::value;
namespace detail_ {
struct void_wrapper {};
template<meta::any_value_sequence S, uZ I, typename... Ts>
struct nonvoid_index_sequence_impl;
template<meta::any_value_sequence S, uZ I, typename T, typename... Ts>
struct nonvoid_index_sequence_impl<S, I, T, Ts...> {
    using type = nonvoid_index_sequence_impl<meta::expand_value_sequence<S, I>, I + 1, Ts...>::type;
};
template<meta::any_value_sequence S, uZ I, typename... Ts>
struct nonvoid_index_sequence_impl<S, I, void_wrapper, Ts...> {
    using type = nonvoid_index_sequence_impl<S, I + 1, Ts...>::type;
};
template<meta::any_value_sequence S, uZ I>
struct nonvoid_index_sequence_impl<S, I> {
    using type = meta::value_to_index_sequence<S>;
};
template<typename... Ts>
struct nonvoid_index_sequence {
    using type = nonvoid_index_sequence_impl<meta::value_sequence<>, 0, Ts...>::type;
};
template<typename... Ts>
using index_sequence_for_nonvoid = nonvoid_index_sequence<Ts...>::type;
}    // namespace detail_

inline constexpr struct {
    template<typename F, typename... Args>
        requires group_invocable<F, Args...>
    PCX_AINLINE static constexpr void operator()(F&& f, Args&&... args) {
        group_invoke_impl(std::forward<F>(f), std::forward<Args>(args)...);
    }
    template<typename F, typename... Args>
        requires group_invocable_with_result<F, Args...>
    PCX_AINLINE static constexpr auto operator()(F&& f, Args&&... args) {
        return group_invoke_impl(std::forward<F>(f), std::forward<Args>(args)...);
    }

private:
    template<typename F, typename... Args>
    PCX_AINLINE static auto group_invoke_impl(F&& f, Args&&... args) {
        if constexpr (compound_op<std::remove_cvref_t<F>>) {
            return invoke_stage_recursive<0>(std::forward<F>(f), std::forward<Args>(args)...);
        } else {
            constexpr auto group_size = (tuple_size_v<std::remove_cvref_t<Args>>, ...);
            return []<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>, F&& f, Args&&... args) {
                constexpr auto invoker = []<uZ I> PCX_LAINLINE(uZc<I>, F&& f, Args&&... args) {
                    if constexpr (std::same_as<decltype(f(get<I>(std::forward<Args>(args))...)), void>) {
                        f(get<I>(std::forward<Args>(args))...);
                        return detail_::void_wrapper{};
                    } else {
                        return f(get<I>(std::forward<Args>(args))...);
                    }
                };
                return make_nonvoid_tuple(invoker(uZc<Is>{},    //
                                                  std::forward<F>(f),
                                                  std::forward<Args>(args)...)...);
            }(std::make_index_sequence<group_size>{}, std::forward<F>(f), std::forward<Args>(args)...);
        }
    }

    template<uZ I, typename F, typename... Args>
    PCX_AINLINE static auto invoke_stage_recursive(F&& f, Args&&... args) {
        constexpr auto final_stage = final_group_result<    //
            decltype(invoke_stage<I>(std::forward<F>(f),    //
                                     std::forward<Args>(args)...))>;
        if constexpr (final_stage) {
            return invoke_stage<I>(std::forward<F>(f), std::forward<Args>(args)...);
        } else {
            return invoke_stage_recursive<I + 1>(
                std::forward<F>(f),
                invoke_stage<I>(std::forward<F>(f), std::forward<Args>(args)...));
        }
    }

    template<uZ I, typename F, typename... Args>
        requires(I == 0)
    PCX_AINLINE static auto invoke_stage(F&& f, Args&&... args) {
        auto&&         stage      = get_stage<I>(std::forward<F>(f));
        constexpr auto group_size = (..., tuple_size_v<std::remove_cvref_t<Args>>);
        return []<uZ... Is, typename S> PCX_LAINLINE(std::index_sequence<Is...>,    //
                                                     S&& stage,
                                                     Args&&... args) {
            constexpr auto invoker = []<uZ K> PCX_LAINLINE(uZc<K>, S&& stage, Args&&... args) {
                return stage(get<K>(std::forward<Args>(args))...);
            };
            return make_nonvoid_tuple(invoker(uZc<Is>{},    //
                                              std::forward<S>(stage),
                                              std::forward<Args>(args)...)...);
        }(std::make_index_sequence<group_size>{},
               std::forward<decltype(stage)>(stage),
               std::forward<Args>(args)...);
    }

    template<uZ I, typename F, typename Arg>
        requires(I > 0)
    PCX_AINLINE static auto invoke_stage(F&& f, Arg&& arg) {
        auto&&         stage      = get_stage<I>(std::forward<F>(f));
        constexpr auto group_size = tuple_size_v<std::remove_cvref_t<Arg>>;
        return []<uZ... Is, typename S> PCX_LAINLINE(std::index_sequence<Is...>,    //
                                                     S&&   stage,
                                                     Arg&& arg) {
            return make_nonvoid_tuple(                        //
                passthrough_invoke(std::forward<S>(stage),    //
                                   std::get<Is>(std::forward<Arg>(arg)))...);
        }(std::make_index_sequence<group_size>{},
               std::forward<decltype(stage)>(stage),
               std::forward<Arg>(arg));
    }

    template<typename... Args>
    PCX_AINLINE static auto make_nonvoid_tuple(Args&&... args) {
        constexpr auto nonvoid_indexes = detail_::index_sequence_for_nonvoid<Args...>{};
        return []<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>, auto&& arg_tuple) {
            return tupi::make_tuple(get<Is>(arg_tuple)...);
        }(nonvoid_indexes, std::forward_as_tuple(std::forward<Args>(args)...));
    };

    // Passes through the argument if it is not an intermediate result
    template<typename S, typename Arg>
        requires(final_result<std::remove_cvref_t<Arg>>)
    PCX_AINLINE static auto passthrough_invoke(S&& /*stage*/, Arg&& arg) {
        return std::forward<Arg>(arg);
    }
    template<typename S, typename Arg>
        requires(!final_result<std::remove_cvref_t<Arg>>)
    PCX_AINLINE static auto passthrough_invoke(S&& stage, Arg&& arg) {
        if constexpr (compound_op<std::remove_cvref_t<S>>) {
            return []<uZ... Is> PCX_LAINLINE(S&& stage, Arg&& arg, std::index_sequence<Is...>) {
                return invoke_stage_recursive<0>(std::forward<S>(stage), get<Is>(std::forward<Arg>(arg))...);
            }(std::forward<S>(stage), std::forward<Arg>(arg), index_sequence_for_tuple<Arg>{});
        } else {
            if constexpr (std::same_as<decltype(apply(std::forward<S>(stage),    //
                                                      std::forward<Arg>(arg))),
                                       void>) {
                apply(std::forward<S>(stage), std::forward<Arg>(arg));
                return detail_::void_wrapper{};
            } else {
                return apply(std::forward<S>(stage), std::forward<Arg>(arg));
            }
        }
    }
} group_invoke;


namespace detail_ {
template<typename F0, typename... Fs>
struct compound_functor_t : compound_op_base {
    template<typename F, typename... Ts>
    auto operator()(this F&& f, Ts&&... args) -> decltype(auto) {
        return [&]<uZ I>(this auto invoker, uZc<I>, auto&&... args) {
            auto stage = get_stage<I>(f);
            if constexpr (final_result<decltype(stage(std::forward<decltype(args)>(args)...))>) {
                return stage(std::forward<decltype(args)>(args)...);
            } else {
                return invoker(uZc<I + 1>{}, stage(std::forward<decltype(args)>(args)...));
            }
        }(uZc<0>{}, args...);
    };
    template<typename F, typename G>
        requires(!std::same_as<std::remove_cvref_t<G>, apply_t>)
    auto operator|(this F&& f, G&& g) {
        return compound_functor_t<std::remove_cvref_t<F>, std::remove_cvref_t<G>>{
            .ops{std::forward<F>(f), std::forward<G>(g)}
        };
    };
    template<uZ I, typename G>
        requires(std::same_as<std::remove_cvref_t<G>, compound_functor_t>)
    constexpr friend auto get_stage(G&& f) {
        return stage_t<I>{.fptr = &f};
    }
    using ops_t = tuple<F0, Fs...>;
    ops_t ops;

private:
    template<uZ OpIdx, uZ OpStage, typename IR>
    struct interim_wrapper : public interim_result_base {
        IR result;
    };
    template<uZ OpIdx, uZ OpStage, typename IR>
    static auto wrap_interim(IR&& result) {
        return interim_wrapper<OpIdx, OpStage, IR>{.result = std::forward<IR>(result)};
    }
    template<uZ I>
    struct stage_t {
        compound_functor_t* fptr;

        auto operator()(auto&&... args)
            requires(I == 0)
        {
            if constexpr (compound_op<tuple_element_t<0, ops_t>>) {
                auto stage = get_stage<0>(get<0>(fptr->ops));
                if constexpr (tupi::final_result<decltype(stage(args...))>) {
                    if constexpr (tuple_size_v<ops_t> == 1) {
                        return stage(args...);
                    } else {
                        return wrap_interim<1, 0>(stage(args...));
                    }
                } else {
                    return wrap_interim<0, 1>(stage(args...));
                }
            } else {
                if constexpr (tuple_size_v<ops_t> == 1) {
                    return get<0>(fptr->ops)(args...);
                } else {
                    return wrap_interim<1, 0>(get<0>(fptr->ops)(args...));
                }
            }
        };
        template<uZ OpIdx, uZ OpStage, typename IR>
        auto operator()(interim_wrapper<OpIdx, OpStage, IR> wr) {
            if constexpr (OpStage > 0 || compound_op<tuple_element_t<OpIdx, ops_t>>) {
                auto stage = get_stage<OpStage>(get<OpIdx>(fptr->ops));
                if constexpr (tupi::final_result<decltype(stage(wr.result))>) {
                    if constexpr (OpIdx == tuple_size_v<ops_t> - 1) {
                        return stage(wr.result);
                    } else {
                        return wrap_interim<OpIdx + 1, 0>(stage(wr.result));
                    }
                } else {
                    return wrap_interim<OpIdx, OpStage + 1>(stage(wr.result));
                }
            } else {
                auto op = get<OpIdx>(fptr->ops);
                if constexpr (OpIdx == tuple_size_v<ops_t> - 1) {
                    return op(wr.result);
                } else {
                    return wrap_interim<OpIdx + 1, 0>(op(wr.result));
                }
            }
        };
    };
};
template<typename... Ts>
struct distributed_t
: public tuple<Ts...>
, std::conditional_t<(final_result<Ts> && ...), decltype([] {}), interim_result_base> {
    using tuple<Ts...>::tuple;
};
}    // namespace detail_
}    // namespace pcx::tupi

template<typename... Ts>
struct std::tuple_size<pcx::tupi::detail_::distributed_t<Ts...>>
: std::integral_constant<std::size_t, sizeof...(Ts)> {};
template<std::size_t I, typename... Ts>
struct std::tuple_element<I, pcx::tupi::detail_::distributed_t<Ts...>> {
    using type = pcx::h::detail_::index_into_types<I, Ts...>::type;
};

namespace pcx::tupi {
namespace detail_ {
struct pass_t {
    template<typename... Args>
    PCX_AINLINE static auto operator()(Args&&... args) {
        return std::forward_as_tuple(std::forward<Args>(args)...);
    }
    template<typename F>
        requires(!std::same_as<std::remove_cvref_t<F>, detail_::apply_t>)
    constexpr auto operator|(F&& f) const {
        return detail_::compound_functor_t<std::remove_cvref_t<F>>{.ops{std::forward<F>(f)}};
    }
};
}    // namespace detail_
/**
 * @brief Compound functor factory. 
 * Example: `pass | [](auto&&... args){ /.../ }` will create a functor accepting `auto&&... args` 
 */
constexpr auto pass = detail_::pass_t{};

namespace detail_ {
struct distribute_t {
    template<typename... Args>
    static constexpr auto operator()(Args&&... args) {
        return detail_::distributed_t<Args...>{std::forward<Args>(args)...};
    }
    template<typename F, typename G>
        requires(!std::same_as<std::remove_cvref_t<G>, detail_::apply_t>)
    constexpr auto operator|(this F&& f, G&& g) {
        return detail_::compound_functor_t<std::remove_cvref_t<F>, std::remove_cvref_t<G>>{
            .ops{std::forward<F>(f), std::forward<G>(g)}
        };
    }
};
}    // namespace detail_

/**
 * @brief Distributes the arguments to be forwarded to pipelined operations.
 * Number of `distribute()` arguments must be equal to the number of `pipeline()` arguments.
 */
inline constexpr auto distribute = detail_::distribute_t{};

namespace detail_ {
template<typename... Fs>
struct pipelined_t : compound_op_base {
    template<typename... Ts>
    auto operator()(distributed_t<Ts...> args) {
        return [&]<uZ I>(this auto invoker, uZc<I>, auto&& args) {
            auto stage = get_stage<I>(*this);
            if constexpr (final_result<decltype(stage(std::forward<decltype(args)>(args)))>) {
                return stage(std::forward<decltype(args)>(args));
            } else {
                return invoker(uZc<I + 1>{}, stage(std::forward<decltype(args)>(args)));
            }
        }(uZc<0>{}, args);
    }
    template<typename F, typename G>
        requires(!std::same_as<std::remove_cvref_t<G>, apply_t>)
    constexpr auto operator|(this F&& f, G&& g) {
        return compound_functor_t<std::remove_cvref_t<F>, std::remove_cvref_t<G>>{
            .ops{std::forward<F>(f), std::forward<G>(g)}
        };
    }
    template<uZ I, typename F>
        requires std::same_as<std::remove_cvref_t<F>, pipelined_t>
    friend constexpr auto get_stage(F&& f) {
        return stage_t<I>{.ref = &f};
    }

    using op_t = tuple<Fs...>;
    op_t ops;

private:
    static constexpr auto op_count = sizeof...(Fs);

    template<uZ OpStage, typename IR>
        requires(!final_result<IR>)
    struct interim_wrapper : public interim_result_base {
        static constexpr auto op_stage_idx = OpStage;

        IR result;
    };
    template<uZ OpStage, typename IR>
    static auto wrap_interim(IR&& result) {
        return interim_wrapper<OpStage, IR>{.result = std::forward<IR>(result)};
    }
    template<uZ I>
    struct stage_t {
        pipelined_t* ref;
        template<typename... Ts>
            requires(sizeof...(Ts) == op_count)
        auto operator()(distributed_t<Ts...> args) /* -> distributed_t<...> */ {
            // Ts => interim_wrapper<OpStage, IR> or final_result<Ts>
            return [&]<uZ... OpIs>(std::index_sequence<OpIs...>) {
                constexpr auto invoke_stage = []<typename Arg>(auto&& f, Arg&& arg) -> decltype(auto) {
                    if constexpr (I == 0) {
                        auto stage = get_stage<0>(std::forward<decltype(f)>(f));
                        if constexpr (final_result<decltype(stage(std::forward<Arg>(arg)))>) {
                            return stage(std::forward<Arg>(arg));
                        } else {
                            return wrap_interim<1>(stage(std::forward<Arg>(arg)));
                        }
                    } else if constexpr (final_result<std::remove_cvref_t<Arg>>) {
                        return std::forward<Arg>(arg);
                    } else {
                        // Arg => interim_wrapper<OpStage, IR>
                        constexpr uZ stage_idx = Arg::op_stage_idx;

                        auto stage = get_stage<stage_idx>(std::forward<decltype(f)>(f));
                        if constexpr (final_result<decltype(stage(arg.result))>) {
                            return stage(arg.result);
                        } else {
                            return wrap_interim<stage_idx + 1>(stage(arg.result));
                        }
                    }
                };
                return distribute(invoke_stage(get<OpIs>(ref->ops), get<OpIs>(args))...);
            }(std::make_index_sequence<op_count>{});
        }
    };
};
}    // namespace detail_
/**
 * @brief Combines the passed functors. If the passed functors are compound, 
 * they will be executed inteleaved. The input must be 
 *
 * @param  f... functors to be pipelined. 
 * @return Compound functor, accepting a return value of `distribute(args...)` 
 * with `sizeof...(args) == sizeof...(f)`.
 */
template<typename... Fs>
auto pipeline(Fs&&... f) {
    return detail_::pipelined_t<std::remove_cvref_t<Fs>...>{.ops{std::forward<Fs>(f)...}};
}

}    // namespace pcx::tupi
