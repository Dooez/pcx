#include "pcx/include/types.hpp"

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

#undef PCX_AINLINE
}    // namespace pcx::h
