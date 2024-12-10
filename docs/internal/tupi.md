 
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

# Objects
## pass
`constexpr static auto operator()(T&& arg) -> T&&` forwards singular input value.
`constexpr auto operator|(F&& f) const` constructs a compound functor that forwards any input arguments to `f`.

## apply
`constexpr static auto operator()(F&& f, tuple_like auto&& args)` invokes `f` with elements of `args` as arguments.
`constexpr auto operator|(F&& f) const` constructs a compound functor that accepts `tuple_like` argument and applies it to the captured functor.
`constexpr friend auto operator|(F&& f, cosnt apply_t&)` constructs an object

## invoke
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
