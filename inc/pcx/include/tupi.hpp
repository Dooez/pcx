#pragma once
#include "pcx/include/meta.hpp"

#include <algorithm>
#include <tuple>
#include <type_traits>

// Tuple interface
namespace pcx::tupi {
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
}    // namespace detail_

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
concept tuple_like_cvref = tuple_like<std::remove_cvref_t<T>>;
template<typename T>
    requires tuple_like<std::remove_cvref_t<T>>
using index_sequence_for_tuple = std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<T>>>;
template<tuple_like_cvref T>
struct tuple_cvref_size {
    static constexpr auto value = tuple_size_v<std::remove_cvref_t<T>>;
};
template<tuple_like_cvref T>
inline constexpr auto tuple_cvref_size_v = tuple_cvref_size<T>::value;


namespace detail_ {
struct interim_result_base {};
struct compound_op_base {};
struct void_wrapper {};

template<typename T>
concept compound_op = std::derived_from<T, detail_::compound_op_base>;
template<typename T>
concept compound_op_cvref = compound_op<std::remove_cvref_t<T>>;
template<typename T>
concept final_result = !std::derived_from<T, detail_::interim_result_base>;
template<typename T>
concept final_result_cvref = final_result<std::remove_cvref_t<T>>;

template<typename F0, typename... Fs>
struct compound_functor_t;
struct apply_t;
struct pipe_mixin {
    template<typename G, typename F>
        requires(!std::same_as<std::remove_cvref_t<F>, apply_t>)
    constexpr auto operator|(this G&& g, F&& f) {
        return compound_functor_t<std::remove_cvref_t<G>, std::remove_cvref_t<F>>{
            .ops{std::forward<G>(g), std::forward<F>(f)}
        };
    }
};
struct call_mixin {
    template<typename G, typename... Args>
    PCX_AINLINE constexpr auto call(this G&& g, Args&&... args) -> decltype(auto) {
        return [&]<uZ I, typename... IArgs>(this auto invoker, uZc<I>, IArgs&&... args) -> decltype(auto) {
            using res_t = decltype(get_stage<I>(std::forward<G>(g))(std::forward<IArgs>(args)...));
            if constexpr (final_result_cvref<res_t>) {
                return get_stage<I>(std::forward<G>(g))(std::forward<IArgs>(args)...);
            } else {
                return invoker(uZc<I + 1>{}, get_stage<I>(std::forward<G>(g))(std::forward<IArgs>(args)...));
            }
        }(uZc<0>{}, std::forward<Args>(args)...);
    }
};

template<uZ I>
struct get_t : public pipe_mixin {
    template<tuple_like_cvref T>
        requires(I < tuple_cvref_size_v<T>)
    PCX_AINLINE constexpr static auto operator()(T&& v) -> decltype(auto) {
        // return std::get<I>(std::forward<T>(v));
        return get<I>(std::forward<T>(v));
    }
};
struct make_tuple_t : public pipe_mixin {
    template<typename... Args>
    PCX_AINLINE constexpr static auto operator()(Args&&... args) {
        return tuple<std::remove_cvref_t<Args>...>{std::forward<Args>(args)...};
    }
};
struct forward_as_tuple_t : public pipe_mixin {
    template<typename... Args>
    PCX_AINLINE constexpr static auto operator()(Args&&... args) {
        return tuple<Args&&...>{std::forward<Args>(args)...};
    }
};

template<typename... Tups>
struct cat_helper;
template<typename... Ts, typename... Us, typename... Tups>
struct cat_helper<tuple<Ts...>, tuple<Us...>, Tups...> {
    using type = cat_helper<tuple<Ts..., Us...>, Tups...>::type;
};
template<typename... Ts>
struct cat_helper<tuple<Ts...>> {
    using type = tuple<Ts...>;
};
template<any_tuple... Tups>
using tuple_cat_result_t = cat_helper<Tups...>::type;
struct tuple_cat_t : public pipe_mixin {
    template<typename... Tups>
        requires(any_tuple<std::remove_cvref_t<Tups>> && ...)
    PCX_AINLINE constexpr static auto operator()(Tups&&... tups) {
        auto tuptup          = forward_as_tuple_t{}(std::forward<Tups>(tups)...);
        auto get_cat_element = [&]<uZ I>(uZc<I>) -> decltype(auto) {
            return [&]<uZ ITup = 0, uZ K = 0, uZ L = 0>(this auto&& it,
                                                        uZc<ITup> = {},
                                                        uZc<K>    = {},
                                                        uZc<L>    = {}) -> decltype(auto) {
                if constexpr (K == I) {
                    return get<L>(get<ITup>(tuptup));
                } else {
                    constexpr auto tup_size = tuple_cvref_size_v<decltype(get<ITup>(tuptup))>;
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
            using cat_t = detail_::tuple_cat_result_t<std::remove_cvref_t<Tups>...>;
            return cat_t{get_cat_element(uZc<Is>{})...};
        }(std::make_index_sequence<total_size>{});
    }
};
template<uZ TupleSize>
struct make_broadcast_tuple_t : public pipe_mixin {
    PCX_AINLINE static constexpr auto operator()(const auto& v) {
        return [&]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
            return make_tuple_t{}((void(Is), v)...);
        }(std::make_index_sequence<TupleSize>{});
    }
};
struct make_flat_tuple_t : public pipe_mixin {
    template<typename T>
    PCX_AINLINE static constexpr auto operator()(T&& tuple) {
        if constexpr (any_tuple<std::remove_cvref_t<T>>) {
            return [&]<uZ... Is> PCX_LAINLINE(std::index_sequence<Is...>) {
                return tuple_cat_t{}(make_flat_tuple_t{}(get<Is>(std::forward<T>(tuple)))...);
            }(index_sequence_for_tuple<T>{});
        } else {
            return make_tuple_t{}(tuple);
        }
    }
};
}    // namespace detail_

template<uZ I>
inline constexpr auto get              = detail_::get_t<I>{};
inline constexpr auto make_tuple       = detail_::make_tuple_t{};
inline constexpr auto forward_as_tuple = detail_::forward_as_tuple_t{};
inline constexpr auto tuple_cat        = detail_::tuple_cat_t{};
template<uZ TupleSize>
inline constexpr auto make_broadcast_tuple = detail_::make_broadcast_tuple_t<TupleSize>{};
inline constexpr auto make_flat_tuple      = detail_::make_flat_tuple_t{};

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
template<typename... Ts>
struct interim_tuple
: public tuple<Ts...>
, std::conditional_t<(final_result<Ts> && ...), decltype([] {}), interim_result_base> {
    using tuple<Ts...>::tuple;
};
}    // namespace detail_
}    // namespace pcx::tupi
template<typename... Ts>
struct std::tuple_size<pcx::tupi::detail_::interim_tuple<Ts...>>
: std::integral_constant<std::size_t, sizeof...(Ts)> {};
template<std::size_t I, typename... Ts>
struct std::tuple_element<I, pcx::tupi::detail_::interim_tuple<Ts...>> {
    using type = pcx::tupi::detail_::index_into_types<I, Ts...>::type;
};
namespace pcx::tupi {
namespace detail_ {
template<typename F, typename Tup>
concept appliable = tuple_like_cvref<Tup> && requires(F&& f, Tup&& args_tup) {
    []<uZ... Is>(std::index_sequence<Is...>, auto&& g, auto&& args) {
        std::forward<decltype(g)>(g)(get<Is>(std::forward<decltype(args)>(args))...);
    }(std::make_index_sequence<tuple_cvref_size_v<Tup>>{}, std::forward<F>(f), std::forward<Tup>(args_tup));
};

template<typename F>
struct to_apply_t;
template<typename F>
struct applied_functor_t;
struct apply_t {
    template<typename F, typename Tup>
        requires appliable<F, Tup>
    static auto operator()(F&& f, Tup&& arg) -> decltype(auto) {
        return [&]<uZ... Is>(std::index_sequence<Is...>) -> decltype(auto) {
            if constexpr (compound_op_cvref<F>) {
                [&]<uZ I, typename... IArgs>(this auto invoker, uZc<I>, IArgs&&... args) -> decltype(auto) {
                    using res_t = decltype(get_stage<I>(std::forward<F>(f))(std::forward<IArgs>(args)...));
                    if constexpr (final_result_cvref<res_t>) {
                        return get_stage<I>(std::forward<F>(f))(std::forward<IArgs>(args)...);
                    } else {
                        return invoker(uZc<I + 1>{},
                                       get_stage<I>(std::forward<F>(f))(std::forward<IArgs>(args)...));
                    }
                }(uZc<0>{}, get<Is>(std::forward<Tup>(arg))...);
            } else {
                return std::forward<F>(f)(get<Is>(std::forward<Tup>(arg))...);
            }
        }(index_sequence_for_tuple<Tup>{});
    }
    template<typename F>
    constexpr friend auto operator|(F&& f, apply_t) {
        return to_apply_t<compound_functor_t<std::remove_cvref_t<F>>>{{.ops{std::forward<F>(f)}}};
    }
    template<typename F>
    constexpr auto operator|(F&& f) const {
        return applied_functor_t<F>{.op = std::forward<F>(f)};
    };
};
template<typename F>
struct applied_functor_t
: public compound_op_base
, public pipe_mixin
, public call_mixin {
    template<typename G, typename Tup>
        requires appliable<F, Tup>
    constexpr auto operator()(this G&& g, Tup&& args) -> decltype(auto) {
        return std::forward<G>(g).call(std::forward<Tup>(args));
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

template<typename F0, typename... Fs>
struct compound_functor_t
: compound_op_base
, public pipe_mixin
, public call_mixin {
    template<typename F, typename... Args>
        requires std::invocable<F0, Args...>
    constexpr auto operator()(this F&& f, Args&&... args) -> decltype(auto) {
        return std::forward<F>(f).call(std::forward<Args>(args)...);
    };
    template<uZ I, typename G>
        requires(std::derived_from<std::remove_cvref_t<G>, compound_functor_t>)
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
                if constexpr (final_result<res_t>) {
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
                if constexpr (final_result_cvref<res_t>) {
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
struct pass_t {
    template<typename Arg>
    PCX_AINLINE static auto operator()(Arg&& arg) -> decltype(auto) {
        return std::forward<Arg>(arg);
    }
    template<typename F>
        requires(!std::same_as<std::remove_cvref_t<F>, detail_::apply_t>)
    constexpr auto operator|(F&& f) const {
        return detail_::compound_functor_t<std::remove_cvref_t<F>>{.ops{std::forward<F>(f)}};
    }
};
struct invoke_t
: compound_op_base
, public pipe_mixin
, public call_mixin {
    template<typename F, typename... Args>
        requires std::invocable<F, Args...>
    PCX_AINLINE static constexpr auto operator()(F&& f, Args&&... args) -> decltype(auto) {
        return invoke_t{}.call(std::forward<F>(f), std::forward<Args>(args)...);
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
        constexpr static auto operator()(F&& f, Args&&... args) -> decltype(auto) {
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
        constexpr static auto operator()(interim_wrapper<Fptr, IR> wrapper) -> decltype(auto) {
            using res_t = decltype(get_stage<I>(*wrapper.fptr)(wrapper.result));
            if constexpr (final_result<res_t>) {
                return get_stage<I>(*wrapper.fptr)(wrapper.result);
            } else {
                return wrap_interim(wrapper.fptr, get_stage<I>(*wrapper.fptr)(wrapper.result));
            }
        }
    };
};
template<typename... Fs>
struct pipelined_t
: compound_op_base
, public pipe_mixin
, public call_mixin {
    static constexpr auto op_count = sizeof...(Fs);

    template<typename F, typename Tup>
        requires tuple_like_cvref<Tup> && (tuple_cvref_size_v<Tup> == op_count)
    auto operator()(this F&& f, Tup&& args) {
        return std::forward<F>(f).call(std::forward<Tup>(args));
    }
    template<uZ I, typename F>
        requires std::derived_from<std::remove_cvref_t<F>, pipelined_t>
    friend constexpr auto get_stage(F&& f) {    // NOLINT(*std-forward*)
        return stage_t<std::add_pointer_t<F>, I>{.ref = &f};
    }
    using op_t = tuple<Fs...>;
    op_t ops;

private:
    template<typename Pptr, uZ I>
    struct stage_t {
        Pptr ref;

        template<tuple_like_cvref Tup>
            requires(tuple_cvref_size_v<Tup> == op_count)
        constexpr auto operator()(Tup&& args) {
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
                    (final_result_cvref<decltype(invoke_stage(get<OpIs>(ref->ops),
                                                              get<OpIs>(std::forward<Tup>(args))))>
                     && ...);
                using ret_t = std::conditional_t<
                    final,
                    tuple<decltype(invoke_stage(get<OpIs>(ref->ops), get<OpIs>(std::forward<Tup>(args))))...>,
                    interim_tuple<decltype(invoke_stage(get<OpIs>(ref->ops),
                                                        get<OpIs>(std::forward<Tup>(args))))...>>;
                return ret_t{invoke_stage(get<OpIs>(ref->ops), get<OpIs>(std::forward<Tup>(args)))...};
            }(std::make_index_sequence<op_count>{});
        }
    };
};
struct group_invoke_t
: compound_op_base
, public pipe_mixin
, public call_mixin {
    template<typename F>
    static constexpr auto operator()(F&& f) {
        // clang-format off
        return pass_t{}                                          
               | [f = std::forward<F>(f)](auto&&... args){
                   return std::forward_as_tuple(f, std::forward<decltype(args)>(args)...);
                 }
               | apply_t{}    //
               | group_invoke_t{};
        // clang-format on
    }
    template<typename F, tuple_like_cvref... Args>
        requires([](auto s0, auto... s) { return ((s0 == s) && ...); }(tuple_cvref_size_v<Args>...))
    static constexpr auto operator()(F&& f, Args&&... args) {
        return group_invoke_t{}.call(std::forward<F>(f), std::forward<Args>(args)...);
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
        template<typename F, tuple_like_cvref... Tups>
            requires(I == 0)
        static constexpr auto operator()(F&& f, Tups&&... arg_tups) {    // NOLINT(*std-forward*)
            constexpr auto group_count = std::min({tuple_cvref_size_v<Tups>...});
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
                    using res_t = interim_tuple<decltype(invoke_group(uZc<Is>{}))...>;
                    return wrap_interim(&f, res_t{invoke_group(uZc<Is>{})...});
                }
            }(std::make_index_sequence<group_count>{});
        };

        template<typename Fptr, typename... Ts>
        static constexpr auto operator()(interim_wrapper<Fptr, interim_tuple<Ts...>> wrapper) {
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
                    using res_t = interim_tuple<decltype(invoke_stage(get<Is>(wrapper.result)))...>;
                    return wrap_interim(wrapper.fptr, res_t{invoke_stage(get<Is>(wrapper.result))...});
                }
            }(std::make_index_sequence<sizeof...(Ts)>{});
        }
    };
};
template<typename Fptr, typename IR>
struct group_invoke_t::is_interim_wrapper<group_invoke_t::interim_wrapper<Fptr, IR>>
: public std::true_type {};

struct pipeline_t : public detail_::pipe_mixin {
    template<typename... Fs>
    static constexpr auto operator()(Fs&&... fs) {
        return detail_::pipelined_t<std::remove_cvref_t<Fs>...>{.ops{std::forward<Fs>(fs)...}};
    }
};
}    // namespace detail_

inline constexpr auto apply = detail_::apply_t{};
/**
 * @brief Compound functor factory.
 * Example: `pass | [](auto&&... args){ /.../ }` will create a functor accepting `auto&&... args`.  
 * When used as functor will forward a single input argument.
 */
inline constexpr auto pass         = detail_::pass_t{};
inline constexpr auto invoke       = detail_::invoke_t{};
inline constexpr auto group_invoke = detail_::group_invoke_t{};
/**
 * @brief Combines the passed functors. If the passed functors are compound, 
 * they will be executed inteleaved. The input must be 
 *
 * @param  f... functors to be pipelined. 
 * @return Compound functor, accepting a return value of `distribute(args...)` 
 * with `sizeof...(args) == sizeof...(f)`.
 */
inline constexpr auto pipeline = detail_::pipeline_t{};

}    // namespace pcx::tupi
