 
Pcx uses a concept of compound operations and explicit pipelining.
An operation consiting of multiple other operations is a compound operation, and the components of a compound op are called 'stages'.
Explicitly pipelining operations means performing operation stages 

# Classes
## *detail_::compound_functor_t*
A basic compound functor that captures another functors and provides compound interface for them.
`constexpr auto operator(Args&&.. args)` calls captured functors with the forwarded `args`.
`constexpr auto operator|(this F&& f, G&& g)` constructs a compound functor that calls `f` and forwards the output to `g`.

## *detail_::to_apply_t*
An intemediate object that captures a functor to be combined with another functor.
`constexpr auto operator|(this F&& f, G&& g)` constructs a compound functor thaj calls `f` and applies the output to `g`.

## *detail_::distributed_t*
A tuple-like object.
# Special objects
## pass
```c++
template<typename Arg>
static auto operator()(Arg&& arg) -> decltype(auto)
```
Forwards the input argument.
```c++
template<typename F>
    requires(!std::same_as<std::remove_cvref_t<F>, detail_::apply_t>)
constexpr auto operator|(F&& f) const;
```
Returns a *detail_::compound_functor_t* that invokes `f`. This is a recommended way to construct a compound functor 
that starts with a lambda lambda or any other functor not from `tupi` namespace.

## apply
```c++
template<typename F, typename Tup>
    requires appliable<F, Tup>
static auto operator()(F&& f, Tup&& arg) -> decltype(auto);
```
invokes `f` with elements of `args` as arguments.
```c++ (1)
template<typename F>
constexpr auto operator|(F&& f) const;
```
Constructs a functor that accepts `tuple_like` objects and applies them to `f`.
```c++ (2)
template<typename F>
constexpr friend auto operator|(F&& f, apply_t);
```
Constructs an object of type *detail_::to_apply_t* that captures `f` to be further combined using `operator|`.

# Pipeable objects {#pipeable}
Pipeable objects share `operator|` to be combined into a compound functor.

```c++
template<typename G, typename F>
    requires(!std::same_as<std::remove_cvref_t<F>, apply_t>)
constexpr auto operator|(this G&& g, F&& f) 
```
Constructs a combined functor from `g` and `f`. The resulting functor forwards input arguments to `g`,
and forwards output of `g` to `f`.

## get<uZ I>
```c++
template<tuple_like_cvref T>
        requires(I < tuple_cvref_size_v<T>)
constexpr static auto operator()(T&& v) -> decltype(auto)
```
Extract I'th element of tuple-like object.

## make_tuple
```c++
template<typename... Args>
constexpr static auto operator()(Args&&... args);
```
Constructs a tuple of values.

## forward_as_tuple
```c++
template<typename... Args>
constexpr static auto operator()(Args&&... args);
```
Constructs a tuple of references.

## tuple_cat
```c++
template<typename... Tups>
    requires(any_tuple<std::remove_cvref_t<Tups>> && ...)
constexpr static auto operator()(Tups&&... tups);
```
Constructs a tuple that is concatenation of input arguments. The types of elements of ther resulting tuple
corrsepond to the types of elements of argument tuples.

## make_broadcast_tuple<uZ TupleSize>
```c++
static constexpr auto operator()(const auto& v);
```
Construct a tuple which elements are copies of the input argument. The size of ther resulting tuple is `TupleSize`.

## make_flat_tuple
```c++
template<typename T>
static constexpr auto operator(T&& tuple);
```
Constructs a flattened tuple of values, where any element of T that is a tuple or a tuple reference gets flattened, 
and replaced by it's elements in the resulting tuple.

## invoke
```c++
template<typename F, typename... Args>
    requires std::invocable<F, Args...>
static constexpr auto operator()(F&& f, Args&&... args) -> decltype(auto);
```
Invokes `f` with arguments `args` and forwards the returned value.

## group_invoke
```c++ (1)
template<typename F, tuple_like_cvref... Args>
    requires([](auto s0, auto... s) { return ((s0 == s) && ...); }(tuple_cvref_size_v<Args>...))
static constexpr auto operator()(F&& f, Args&&... args);
```
Invokes `f` mulptipel groups of arguments, with I'th group being I'th elements of `args`. 
The returned value is a tuple with I'th elements being the result of invokation of `f` with I'th group.

```c++ (2)
template<typename F>
static constexpr auto operator()(F&& f);
```
Constructs a compund functor that group invokes `f` (see `(1)`).

`constexpr static auto operator()(F&& f, Args&&... args)` invokes `f` with `args`.
`constexpr auto operator|(F&& f)` construct a compound functor that accepts a functor and variadic list of arguments and forwards the output to `f`.

## distribute
`constexpr static auto operator()(auto&&... args)` returns a `tuple_like` object that can be passed to the pipelined functor.
`constexpr auto operator|(F&& f)` construct a compound functor that accepts arguments and passes *detail_::distributed_t* to `f`.

## pipeline
A functor that pipelines composing compound functors to be executed stage-by-stage. 
`constexpr static auto operator()(F&&... fs)` constructs a functor accepting and returning *detail_::distributed_t*.
TODO: add `operator|()` that autoamically distributes the arguments.

Example:
```
auto x = tupi::pass | xstage0 | xstage1;
auto y = tupi::pass | ystage0 | ystage1;
auto p = tupi::distribute | tupi::pipeline(x, y);
p(0, 1);
```
will result in 
1. xtmp = xstage0(0);
2. ytmp = ystage0(1);
3. xstage1(xtmp);
4. ystage2(ytmp);

## group_invoke
`constexpr static auto operator()(F&& f, Args&&... args)`  Invokes `f` with argument sets `f(get<0>(args)...)`, `f(get<1>(args)...)`
    Args are tuple-like objects of the same tuple size, each tuple representing a single input argument. 
    Invocation with different argument sets are pipelined.
