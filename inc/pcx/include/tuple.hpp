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

static constexpr struct {
    template<typename... Args>
    PCX_AINLINE constexpr static auto operator()(Args&&... args) {
        return tuple<std::remove_cvref_t<Args>...>{std::forward<Args>(args)...};
    }
} make_tuple{};
static constexpr struct {
    template<typename... Args>
    PCX_AINLINE constexpr static auto operator()(Args&&... args) {
        return tuple<Args&&...>{std::forward<Args>(args)...};
    }
} forward_as_tuple{};

namespace detail_ {
template<uZ I>
struct get_t {
    template<typename T>
    PCX_AINLINE constexpr static auto operator()(T&& v) -> decltype(auto) {
        return std::get<I>(std::forward<T>(v));
    }
};
}    // namespace detail_
template<uZ I>
inline constexpr auto get = detail_::get_t<I>{};

namespace detail_ {
template<typename... Tups>
struct cat_t;
template<typename... Ts, typename... Us, typename... Tups>
struct cat_t<tuple<Ts...>, tuple<Us...>, Tups...> {
    using type = cat_t<tuple<Ts..., Us...>, Tups...>::type;
};
template<typename... Ts>
struct cat_t<tuple<Ts...>> {
    using type = tuple<Ts...>;
};
template<any_tuple... Tups>
using tuple_cat_t = cat_t<Tups...>::type;
}    // namespace detail_

static constexpr struct tuple_cat_t {
    template<typename... Tups>
        requires(any_tuple<std::remove_cvref_t<Tups>> && ...)
    PCX_AINLINE constexpr static auto operator()(Tups&&... tups) {
        auto tuptup          = forward_as_tuple(std::forward<Tups>(tups)...);
        auto get_cat_element = [&]<uZ I>(uZc<I>) -> decltype(auto) {
            return [&]<uZ ITup = 0, uZ K = 0, uZ L = 0>(this auto&& it,
                                                        uZc<ITup> = {},
                                                        uZc<K>    = {},
                                                        uZc<L>    = {}) -> decltype(auto) {
                if constexpr (K == I) {
                    return get<L>(get<ITup>(tuptup));
                } else {
                    constexpr auto tup_size = tuple_size_v<std::remove_cvref_t<decltype(get<ITup>(tuptup))>>;
                    if constexpr (L < tup_size - 1) {
                        return std::forward<decltype(it)>(it)(uZc<ITup>{}, uZc<K + 1>{}, uZc<L + 1>{});
                    } else {
                        return std::forward<decltype(it)>(it)(uZc<ITup + 1>{}, uZc<K + 1>{}, uZc<0>{});
                    }
                }
            }();
        };
        constexpr auto total_size = (tuple_size_v<Tups> + ...);
        return [&]<uZ... Is>(std::index_sequence<Is...>) {
            using cat_t = detail_::tuple_cat_t<std::remove_cvref_t<Tups>...>;
            return cat_t{get_cat_element(uZc<Is>{})...};
        }(std::make_index_sequence<total_size>{});
    };
} tuple_cat{};

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
template<typename T>
concept tuple_like_cvref = tuple_like<std::remove_cvref_t<T>>;


struct interim_result_base {};
struct compound_op_base {};

}    // namespace detail_
using detail_::compound_op_base;
template<typename T>
concept compound_op = std::derived_from<T, detail_::compound_op_base>;
template<typename T>
concept compound_op_cvref = compound_op<std::remove_cvref_t<T>>;
template<typename T>
concept final_result = !std::derived_from<T, detail_::interim_result_base>;
template<typename T>
concept final_result_cvref = final_result<std::remove_cvref_t<T>>;

namespace detail_ {
template<typename F>
struct to_apply_t;
template<typename F>
struct applied_functor_t;
struct apply_t {
    template<typename F, typename Tup>
        requires tuple_like<std::remove_cvref_t<Tup>>
    static auto operator()(F&& f, Tup&& arg) -> decltype(auto) {
        return [&]<uZ... Is>(std::index_sequence<Is...>) -> decltype(auto) {
            return std::forward<F>(f)(get<Is>(std::forward<Tup>(arg))...);
        }(index_sequence_for_tuple<Tup>{});
    };
    template<typename F>
    constexpr friend auto operator|(F&& f, const apply_t&) {
        return to_apply_t<F>{std::forward<F>(f)};
    }
    template<typename F>
    constexpr auto operator|(F&& f) const {
        return applied_functor_t<F>{.op = std::forward<F>(f)};
    };
};

template<typename F>
struct applied_functor_t : public compound_op_base {
    template<typename G, typename Tup>
        requires tuple_like<std::remove_cvref_t<Tup>>
    constexpr auto operator()(this G&& g, Tup&& args) -> decltype(auto) {
        return [&]<uZ... Is>(std::index_sequence<Is...>) -> decltype(auto) {
            [&]<uZ I, typename... Args>(this auto invoker, uZc<I>, Args&&... args) -> decltype(auto) {
                using res_t = decltype(get_stage<I>(std::forward<G>(g))(std::forward<Args>(args)...));
                if constexpr (final_result_cvref<res_t>) {
                    return get_stage<I>(std::forward<G>(g))(std::forward<Args>(args)...);
                } else {
                    return invoker(uZc<I + 1>{},
                                   get_stage<I>(std::forward<G>(g))(std::forward<Args>(args)...));
                }
            }(uZc<0>{}, get<Is>(std::forward<Tup>(args))...);
        }(index_sequence_for_tuple<Tup>{});
    }
    template<uZ I, typename G>
        requires std::same_as<std::remove_cvref_t<G>, applied_functor_t>
    friend constexpr auto get_stage(G&& g) {    // NOLINT(*std-forward*)
        return stage_t<std::add_pointer_t<G>, I>{.fptr = &g};
    }
    F op;

private:
    template<typename Fptr, uZ I>
    struct stage_t {
        Fptr fptr;

        template<typename Tup>
            requires(I == 0 && tuple_like<std::remove_cvref_t<Tup>>)
        constexpr auto operator()(Tup&& args) const -> decltype(auto) {
            if constexpr (compound_op_cvref<F>) {
                using res_t = decltype(apply_t{}(get_stage<I>(fptr->op), std::forward<Tup>(args)));
                return apply_t{}(get_stage<I>(fptr->op), std::forward<Tup>(args));
            } else {
                return apply_t{}(fptr->op, std::forward<Tup>(args));
            }
        }
        constexpr auto operator()(auto&& v) const -> decltype(auto) {
            return get_stage<I>(fptr->op)(std::forward<decltype(v)>(v));
        }
    };
};

template<typename F>
struct to_apply_t {
    template<typename AF, typename G>
    constexpr auto operator|(this AF&& apply_f, G&& g) {
        return std::forward_like<AF>(apply_f.op) | applied_functor_t<G>{.op = std::forward<G>(g)};
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
}    // namespace detail_

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
//
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

template<typename F0, typename... Fs>
struct compound_functor_t : compound_op_base {
    template<typename F, typename... Ts>
    constexpr auto operator()(this F&& f, Ts&&... args) -> decltype(auto) {
        return [&]<uZ I>(this auto invoker, uZc<I>, auto&&... args) -> decltype(auto) {
            using res_t = decltype(get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args)...));
            if constexpr (final_result<res_t>) {
                return get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args)...);
            } else {
                return invoker(uZc<I + 1>{},
                               get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args)...));
            }
        }(uZc<0>{}, args...);
    };
    template<typename F, typename G>
        requires(!std::same_as<std::remove_cvref_t<G>, apply_t>)
    constexpr auto operator|(this F&& f, G&& g) {
        return compound_functor_t<std::remove_cvref_t<F>, std::remove_cvref_t<G>>{
            .ops{std::forward<F>(f), std::forward<G>(g)}
        };
    };
    template<uZ I, typename G>
        requires(std::same_as<std::remove_cvref_t<G>, compound_functor_t>)
    constexpr friend auto get_stage(G&& g) {    // NOLINT(*std-forward*)
        return stage_t<std::add_pointer_t<G>, I>{.fptr = &g};
    }
    using ops_t = tuple<F0, Fs...>;
    ops_t ops;

private:
    static constexpr auto op_count = tuple_size_v<ops_t>;
    template<uZ OpIdx, uZ OpStage, typename IR>
    struct interim_wrapper : public interim_result_base {
        IR result;
    };
    template<uZ OpIdx, uZ OpStage, typename IR>
    static auto wrap_interim(IR&& result) {
        return interim_wrapper<OpIdx, OpStage, IR>{.result = std::forward<IR>(result)};
    }
    template<typename Fptr, uZ I>
    struct stage_t {
        Fptr fptr;

        template<typename... Args>
            requires(I == 0)
        constexpr auto operator()(Args&&... args) const {
            if constexpr (compound_op_cvref<tuple_element_t<0, ops_t>>) {
                using res_t = decltype(get_stage<0>(get<0>(fptr->ops))(std::forward<Args>(args)...));
                if constexpr (tupi::final_result<res_t>) {
                    if constexpr (op_count == 1) {
                        return get_stage<0>(get<0>(fptr->ops))(std::forward<Args>(args)...);
                    } else {
                        return wrap_interim<1, 0>(
                            get_stage<0>(get<0>(fptr->ops))(std::forward<Args>(args)...));
                    }
                } else {
                    return wrap_interim<0, 1>(get_stage<0>(get<0>(fptr->ops))(std::forward<Args>(args)...));
                }
            } else {
                if constexpr (op_count == 1) {
                    return get<0>(fptr->ops)(std::forward<Args>(args)...);
                } else {
                    return wrap_interim<1, 0>(get<0>(fptr->ops)(std::forward<Args>(args)...));
                }
            }
        };
        template<uZ OpIdx, uZ OpStage, typename IR>
        constexpr auto operator()(interim_wrapper<OpIdx, OpStage, IR> wr) const {
            if constexpr (OpStage > 0 || compound_op_cvref<tuple_element_t<OpIdx, ops_t>>) {
                using res_t = decltype(get_stage<OpStage>(get<OpIdx>(fptr->ops))(wr.result));
                if constexpr (tupi::final_result_cvref<res_t>) {
                    if constexpr (OpIdx == op_count - 1) {
                        return get_stage<OpStage>(get<OpIdx>(fptr->ops))(wr.result);
                    } else {
                        return wrap_interim<OpIdx + 1, 0>(
                            get_stage<OpStage>(get<OpIdx>(fptr->ops))(wr.result));
                    }
                } else {
                    return wrap_interim<OpIdx, OpStage + 1>(
                        get_stage<OpStage>(get<OpIdx>(fptr->ops))(wr.result));
                }
            } else {
                if constexpr (OpIdx == op_count - 1) {
                    return get<OpIdx>(fptr->ops)(wr.result);
                } else {
                    return wrap_interim<OpIdx + 1, 0>(get<OpIdx>(fptr->ops)(wr.result));
                }
            }
        };
    };
};
struct invoke_t : compound_op_base {
    template<typename F, typename... Args>
        requires std::invocable<F, Args...>
    PCX_AINLINE static constexpr auto operator()(F&& f, Args&&... args) {
        return []<uZ I>(this auto invoker, uZc<I>, auto&&... args) {
            using res_t = decltype(stage_t<I>{}(std::forward<decltype(args)>(args)...));
            if constexpr (final_result<res_t>) {
                return stage_t<I>{}(std::forward<decltype(args)>(args)...);
            } else {
                return invoker(uZc<I + 1>{}, stage_t<I>{}(std::forward<decltype(args)>(args)...));
            }
        }(uZc<0>{}, std::forward<F>(f), std::forward<Args>(args)...);
    }
    template<typename F>
    constexpr auto operator|(F&& f) const {
        return compound_functor_t<invoke_t, F>{
            .ops = {invoke_t{}, std::forward<F>(f)}
        };
    }
    template<uZ I>
    friend constexpr auto get_stage(invoke_t) {
        return stage_t<I>{};
    }

private:
    template<typename Fptr, typename IR>
    struct interim_wrapper : interim_result_base {
        Fptr fptr;
        IR   result;
    };
    template<typename Fptr, typename IR>
    static constexpr auto wrap_interim(Fptr fptr, IR res) {
        return interim_wrapper<Fptr, IR>{.fptr = fptr, .result = res};
    }

    template<uZ I>
    struct stage_t {
        template<typename F, typename... Args>
        constexpr static auto operator()(F&& f, Args&&... args) {
            if constexpr (compound_op_cvref<F>) {
                using res_t = decltype(get_stage<I>(std::forward<F>(f))(std::forward<Args>(args)...));
                if constexpr (final_result<res_t>) {
                    return get_stage<I>(std::forward<F>(f))(std::forward<Args>(args)...);
                } else {
                    return wrap_interim(&f, get_stage<I>(std::forward<F>(f))(std::forward<Args>(args)...));
                }
            } else {
                return std::forward<F>(f)(std::forward<Args>(args)...);
            }
        }
        template<typename Fptr, typename IR>
        constexpr static auto operator()(interim_wrapper<Fptr, IR> wrapper) {
            using res_t = decltype(get_stage<I>(*wrapper.fptr)(wrapper.result));
            if constexpr (final_result<res_t>) {
                return get_stage<I>(*wrapper.fptr)(wrapper.result);
            } else {
                return wrap_interim(wrapper.fptr, get_stage<I>(*wrapper.fptr)(wrapper.result));
            }
        }
    };
};

}    // namespace detail_
constexpr inline auto invoke = detail_::invoke_t{};
namespace detail_ {
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
    template<typename Arg>
    PCX_AINLINE static auto operator()(Arg&& arg) {
        return std::forward<Arg>(arg);
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
 * If used as a functor will pass input argument.
 */
constexpr auto pass = detail_::pass_t{};

namespace detail_ {
struct distribute_t {
    template<typename... Args>
    static constexpr auto operator()(Args&&... args) {
        return detail_::distributed_t<std::remove_cvref_t<Args>...>{std::forward<Args>(args)...};
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
    template<typename F, typename Tup>
        requires tuple_like_cvref<Tup> && (tuple_size_v<std::remove_cvref_t<Tup>> == sizeof...(Fs))
    auto operator()(this F&& f, Tup&& args) {
        return [&]<uZ I>(this auto invoker, uZc<I>, auto&& args) {
            using res_t = decltype(get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args)));
            if constexpr (final_result<res_t>) {
                return get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args));
            } else {
                return invoker(uZc<I + 1>{},
                               get_stage<I>(std::forward<F>(f))(std::forward<decltype(args)>(args)));
            }
        }(uZc<0>{}, std::forward<Tup>(args));
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
    friend constexpr auto get_stage(F&& f) {    // NOLINT(*std-forward*)
        return stage_t<std::add_pointer_t<F>, I>{.ref = &f};
    }

    using op_t = tuple<Fs...>;
    op_t ops;

private:
    static constexpr auto op_count = sizeof...(Fs);

    template<typename Pptr, uZ I>
    struct stage_t {
        Pptr ref;

        template<typename Tup>
            requires tuple_like_cvref<Tup> && (tuple_size_v<std::remove_cvref_t<Tup>> == sizeof...(Fs))
        constexpr auto operator()(Tup&& args) /* -> distributed_t<...> */ {
            // Ts => interim_wrapper<OpStage, IR> or final_result<Ts>
            return [&]<uZ... OpIs>(std::index_sequence<OpIs...>) {
                constexpr auto invoke_stage = []<typename Arg>(auto&& f, Arg&& arg) -> decltype(auto) {
                    if constexpr (I == 0) {
                        if constexpr (compound_op_cvref<decltype(f)>) {
                            return get_stage<I>(std::forward<decltype(f)>(f))(std::forward<Arg>(arg));
                        } else {
                            return std::forward<decltype(f)>(f)(std::forward<Arg>(arg));
                        }
                    } else if constexpr (final_result_cvref<Arg>) {
                        return std::forward<Arg>(arg);
                    } else {
                        return get_stage<I>(std::forward<decltype(f)>(f))(arg);
                    }
                };
                constexpr auto final =
                    (final_result_cvref<decltype(invoke_stage(get<OpIs>(ref->ops), get<OpIs>(args)))> && ...);
                using ret_t = std::conditional_t<
                    final,
                    tuple<decltype(invoke_stage(get<OpIs>(ref->ops), get<OpIs>(args)))...>,
                    distributed_t<decltype(invoke_stage(get<OpIs>(ref->ops), get<OpIs>(args)))...>>;
                return ret_t{invoke_stage(get<OpIs>(ref->ops), get<OpIs>(args))...};
            }(std::make_index_sequence<op_count>{});
        }
    };
};

struct group_invoke_t {
    template<typename F>
    static constexpr auto operator()(F&& f) {
        // clang-format off
        return pass                                          
               | [f = std::forward<F>(f)](auto&&... args){
                   return std::forward_as_tuple(f, std::forward<decltype(args)>(args)...);
                 }
               | apply    //
               | group_invoke_t{};
        // clang-format on
    }
    template<typename F, typename... Args>
    static constexpr auto operator()(F&& f, Args&&... args) {
        return []<uZ I>(this auto invoker, uZc<I>, auto&&... args) {
            using res_t = decltype(stage_t<I>{}(std::forward<decltype(args)>(args)...));
            if constexpr (final_result<res_t>) {
                return stage_t<I>{}(std::forward<decltype(args)>(args)...);
            } else {
                return invoker(uZc<I + 1>{}, stage_t<I>{}(std::forward<decltype(args)>(args)...));
            }
        }(uZc<0>{}, std::forward<F>(f), std::forward<Args>(args)...);
    }
    template<typename F>
    constexpr auto operator|(F&& f) const {
        return compound_functor_t<group_invoke_t, F>{
            .ops = {group_invoke_t{}, std::forward<F>(f)}
        };
    }
    template<uZ I>
    constexpr friend auto get_stage(group_invoke_t) {
        return stage_t<I>{};
    }

private:
    template<typename Fptr, typename IR>
        requires(!final_result<IR>)
    struct interim_wrapper : public interim_result_base {
        Fptr fptr;
        IR   result;
    };
    template<typename T>
    struct is_interim_wrapper : public std::false_type {};
    template<typename Fptr, typename IR>
    static auto wrap_interim(Fptr fptr, IR&& result) {
        return interim_wrapper<Fptr, IR>{.fptr = fptr, .result = std::forward<IR>(result)};
    }

    template<uZ I>
    struct stage_t {
        template<typename F, typename... Tups>
            requires(I == 0 && (tuple_like<std::remove_cvref_t<Tups>> && ...))
        static constexpr auto
        operator()(F&& f, Tups&&... arg_tups) /* -> distributed_t<...> */ {    // NOLINT(*std-forward*)
            constexpr auto group_count = (..., tuple_size_v<std::remove_cvref_t<Tups>>);
            return [&]<uZ... Is>(std::index_sequence<Is...>) {
                auto invoke_group = [&]<uZ IGrp>(uZc<IGrp>) -> decltype(auto) {
                    auto invoke_stage = [&]<typename... Args>(Args&&... args) -> decltype(auto) {
                        if constexpr (compound_op_cvref<F>) {
                            return get_stage<0>(std::forward<F>(f))(std::forward<Args>(args)...);
                        } else {
                            return std::forward<F>(f)(std::forward<Args>(args)...);
                        }
                    };
                    using ret_t = decltype(invoke_stage(get<IGrp>(std::forward<Tups>(arg_tups))...));
                    if constexpr (std::same_as<ret_t, void>) {
                        invoke_stage(get<IGrp>(std::forward<Tups>(arg_tups))...);
                        return void_wrapper{};
                    } else {
                        return invoke_stage(get<IGrp>(std::forward<Tups>(arg_tups))...);
                    }
                };
                constexpr auto final = (final_result<decltype(invoke_group(uZc<Is>{}))> && ...);
                if constexpr (final) {
                    using res_t = tuple<decltype(invoke_group(uZc<Is>{}))...>;
                    return res_t{invoke_group(uZc<Is>{})...};
                } else {
                    using res_t = distributed_t<decltype(invoke_group(uZc<Is>{}))...>;
                    return wrap_interim(&f, res_t{invoke_group(uZc<Is>{})...});
                }
            }(std::make_index_sequence<group_count>{});
        };

        template<typename Fptr, typename... Ts>
        static constexpr auto operator()(interim_wrapper<Fptr, distributed_t<Ts...>> wrapper) {
            return [&]<uZ... Is>(std::index_sequence<Is...>) {
                auto invoke_stage = [&]<typename Arg>(Arg&& arg) -> decltype(auto) {
                    if constexpr (final_result_cvref<Arg>) {
                        return std::forward<Arg>(arg);
                    } else {
                        using ret_t = decltype(get_stage<I>(*wrapper.fptr)(arg));
                        if constexpr (std::same_as<ret_t, void>) {
                            get_stage<I> (*wrapper.fptr)(arg);
                            return void_wrapper{};
                        } else {
                            return get_stage<I>(*wrapper.fptr)(arg);
                        }
                    }
                };
                constexpr auto final =
                    (final_result_cvref<decltype(invoke_stage(get<Is>(wrapper.result)))> && ...);
                if constexpr (final) {
                    using res_t = tuple<decltype(invoke_stage(get<Is>(wrapper.result)))...>;
                    return res_t{invoke_stage(get<Is>(wrapper.result))...};
                } else {
                    using res_t = distributed_t<decltype(invoke_stage(get<Is>(wrapper.result)))...>;
                    return wrap_interim(wrapper.fptr, res_t{invoke_stage(get<Is>(wrapper.result))...});
                }
            }(std::make_index_sequence<sizeof...(Ts)>{});
        }
    };
};
template<typename Fptr, typename IR>
struct group_invoke_t::is_interim_wrapper<group_invoke_t::interim_wrapper<Fptr, IR>>
: public std::true_type {};
}    // namespace detail_
inline constexpr auto group_invoke = detail_::group_invoke_t{};
/**
 * @brief Combines the passed functors. If the passed functors are compound, 
 * they will be executed inteleaved. The input must be 
 *
 * @param  f... functors to be pipelined. 
 * @return Compound functor, accepting a return value of `distribute(args...)` 
 * with `sizeof...(args) == sizeof...(f)`.
 */
template<typename... Fs>
constexpr auto pipeline(Fs&&... f) {
    return detail_::pipelined_t<std::remove_cvref_t<Fs>...>{.ops{std::forward<Fs>(f)...}};
}

}    // namespace pcx::tupi
