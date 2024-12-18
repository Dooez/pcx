Pcx uses a concept of compound operations and explicit pipelining.
An operation consiting of multiple other operations is a compound operation, 
and the components of a compound operation are called 'stages'.
Explicitly pipelining operations is performing stages of multiple pipelined opeartions
one stage number at a time for all operations interleaved.

Special inline objects:
[pass](#pass)
[apply](#apply)
Pipeable inline object
[get](#get)                
[make_tuple](#make_tuple)         
[forward_as_tuple](#forward_asup)   
[tuple_cat](#tuple_cat)          
[make_broadcast_tuple](#make_broadcast_tuple) 
[make_flat_tuple](#make_flat_tuple)      
[invoke](#invoke)               
[group_invoke](#group_invoke)       
[pipeline](#pipeline)            

# Concepts
## any_tuple
Concept is satisfied by any specialization of `tupi::tuple`.

## tuple_like
Concept requires `std::tuple_size_v<T>` to be definded and convertable to `uZ`
and `get<I>(T)` to be valid for all `I < std::tuple_size_<T>`.

## tuple_like_cvref
```C++
template<typename T>
concepct tuple_like_cvref = tuple_like<std::remove_cvref_t<T>>;
```

# Classes
## *detail_::compound_functor_t*
A basic compound functor that captures another functors and provides compound interface for them.
`constexpr auto operator(Args&&.. args)` calls captured functors with the forwarded `args`.
`constexpr auto operator|(this F&& f, G&& g)` constructs a compound functor that calls `f` and forwards the output to `g`.

## *detail_::to_apply_t*
An intemediate object that captures a functor to be combined with another functor.
`constexpr auto operator|(this F&& f, G&& g)` constructs a compound functor thaj calls `f` and applies the output to `g`.

# Special inline objects
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
Functor of type `apply_t`.
```c++
template<typename F, typename Tup>
    requires appliable<F, Tup>
static auto operator()(F&& f, Tup&& arg) -> decltype(auto);
```
Invokes `f` with elements of `args` as arguments.
```c++ (1)
template<typename F>
constexpr auto operator|(F&& f) const;
```
```c++ (2)
template<typename F>
constexpr friend auto operator|(F&& f, apply_t);
```
1) Constructs a functor that accepts [tuple_like](#tuple_like) objects and applies them to `f`.
2) Constructs an object of type *detail_::to_apply_t* that captures `f` to be further combined using `operator|`.

# Pipeable inline objects
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
```c++ (2)
template<typename F>
static constexpr auto operator()(F&& f);
```
1) Invokes `f` with multiple groups of arguments, I'th group being I'th elements of `args`. 
   The returned value is a tuple with I'th elements being the result of invokation of `f` with I'th group.
2) Constructs a compund functor that group invokes `f` (see `(1)`).

## pipeline
```c++
template<typename... Fs>
static constexpr auto operator()(Fs&&... fs)
```
Constructs a compound functor that invokes `fs` interleaving their stages.

# Usage examples
```c++
auto x = tupi::pass | xstage0 | xstage1;
auto y = tupi::pass | ystage0 | ystage1;
auto p = tupi::make_tuple | tupi::pipeline(x, y);
auto [a, b] = p(0, 1);
auto [c ,d] = tupi::group_invoke(x, tupi::make_tuple(10, 11));
```
will result in 
1. atmp = xstage0(0);
2. btmp = ystage0(1);
3. a = xstage1(xtmp);
4. b = ystage2(ytmp);

5. ctmp = xstage0(10);
6. dtmp = xstage0(11);
7. c = xstage1(ctmp);
8. d = xstage2(dtmp);
