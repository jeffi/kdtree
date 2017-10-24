#pragma once
#ifndef UNC_ROBOTICS_KDTREE_TUPLE_HPP
#define UNC_ROBOTICS_KDTREE_TUPLE_HPP

// helper methods for dealing with tuples used by CompoundSpace.

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

// computes the result type of a compound sum
// e.g. float + double => double
template <typename _T, typename ... _Rest>
struct SumResultType { typedef _T type; };
template <typename _T, typename _U, typename ... _Rest>
struct SumResultType<_T, _U, _Rest...> { typedef typename SumResultType<decltype(_T() + _U()), _Rest...>::type type; };


template <typename _Fn, typename _T>
constexpr decltype(auto) reduceArgs(_Fn&& fn, _T a) { return a; }
template <typename _Fn, typename _First, typename _Second, typename ... _Rest>
constexpr decltype(auto) reduceArgs(_Fn&& fn, _First&& a, _Second&& b, _Rest&& ... args) {
    return reduceArgs(std::forward<_Fn>(fn), std::forward<_Fn>(fn)(a, b), args...);
}
template <typename _Fn, typename _Tuple, std::size_t... I>
constexpr decltype(auto) reduce_impl(_Fn&& fn, _Tuple&& args, std::index_sequence<I...>) {
    return reduceArgs(std::forward<_Fn>(fn), std::get<I>(std::forward<_Tuple>(args))...);
}
template <typename _Fn, typename _Tuple>
constexpr decltype(auto) reduce(_Fn&& fn, _Tuple&& args) {
    return reduce_impl(
        std::forward<_Fn>(fn),
        std::forward<_Tuple>(args),
        std::make_index_sequence<std::tuple_size<typename std::decay<_Tuple>::type>::value>{});
}
template <typename _First, typename ... _Rest>
constexpr decltype(auto) sum(_First&& a, _Rest&& ... args) {
    return reduce(std::plus<typename SumResultType<_First, _Rest...>::type>(), a, args...);
}
template <typename ... _T>
constexpr decltype(auto) sum(std::tuple<_T...>&& args) {
    return reduce(std::plus<typename SumResultType<_T...>::type>(), args);
}
template <typename _Fn, typename _Tuple, std::size_t ... I>
constexpr decltype(auto) map_impl(_Fn&& fn, _Tuple&& args, std::index_sequence<I...>) {
    return std::make_tuple(std::forward<_Fn>(fn)(std::get<I>(std::forward<_Tuple>(args)))...);
}
template <typename _Fn, typename ... _T>
constexpr decltype(auto) map(_Fn&& fn, std::tuple<_T...>&& args) {
    return map_impl(fn, args, std::make_index_sequence<sizeof...(_T)>{});
}
template <typename _Fn, typename _Tuple, std::size_t ... I>
constexpr decltype(auto) apply_impl(_Fn&& fn, _Tuple&& args, std::index_sequence<I...>) {
    return std::forward<_Fn>(fn)(std::get<I>(std::forward<_Tuple>(args))...);
}
template <typename _Fn, typename _Tuple>
constexpr decltype(auto) apply(_Fn&& fn, _Tuple&& t) {
    return apply_impl(
        std::forward<_Fn>(fn),
        std::forward<_Tuple>(t),
        std::make_index_sequence<std::tuple_size<_Tuple>::value>{});
}
template <std::size_t I, typename _Tuple, std::size_t ... J>
constexpr decltype(auto) slice_impl(_Tuple&& tuple, std::index_sequence<J...>) {
    return std::make_tuple(std::get<I>(std::get<J>(std::forward<_Tuple>(tuple)))...);
}

// Returns a tuple containing the get<I> of each element of the argument
template <std::size_t I, typename _Tuple>
constexpr decltype(auto) slice(_Tuple&& tuple) {
    return slice_impl<I>(
        std::forward<_Tuple>(tuple),
        std::make_index_sequence<std::tuple_size<_Tuple>::value>{});
}
template <typename _Fn, typename _Args, std::size_t ... I>
constexpr decltype(auto) zip_impl(_Fn&& fn, _Args&& args, std::index_sequence<I...>) {
    return std::make_tuple(
        apply(std::forward<_Fn>(fn), slice<I>(std::forward<_Args>(args)))...);
}
template <typename _Fn, typename _First, typename ... _Rest>
constexpr decltype(auto) zip(_Fn&& fn, _First&& first, _Rest&& ... rest) {
    return zip_impl(
        fn,
        std::make_tuple(std::forward<_First>(first), std::forward<_Rest>(rest)...),
        std::make_index_sequence<std::tuple_size<typename std::decay<_First>::type>::value>{});
}


}
}
}
}


#endif // UNC_ROBOTICS_KDTREE_TUPLE_HPP
