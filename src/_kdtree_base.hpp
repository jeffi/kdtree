#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP

#include "_spaces.hpp"

namespace unc { namespace robotics { namespace kdtree {

struct MidpointSplit {};
struct MedianSplit {};

template <
    typename _T,
    typename _Space,
    typename _GetKey,
    typename _SplitStrategy,
    bool _dynamic = true,
    bool _lockfree = false>
struct KDTree;

namespace detail {
template <typename _Node, typename _Space>
struct MidpointAddTraversal;
template <typename _Node, typename _Space>
struct MidpointNearestTraversal;
template <typename _Space>
struct MedianAccum;
template <typename _Space>
struct MedianNearestTraversal;

struct CompareFirst {
    template <typename _First, typename _Second>
    constexpr bool operator() (const std::pair<_First,_Second>& a, const std::pair<_First,_Second>& b) const {
        return a.first < b.first;
    }
};

// helper to enable builtin wrapper for long types.  The problem this
// resolves is that `int` and `long` may be the same type on some
// systems and not on others.
template <typename T>
struct enable_builtin_long {
    static constexpr bool value = std::is_integral<T>::value
        && sizeof(T)==sizeof(long) && sizeof(T) != sizeof(int);
};

// helper to enable builtin wrapper for long long types.  The problem
// this resolves is that `long` and `long long` may be the same type
// on some systems and not on others.
template <typename T>
struct enable_builtin_long_long {
    static constexpr bool value = std::is_integral<T>::value
        && sizeof(T)==sizeof(long long) && sizeof(T) != sizeof(long);
};

// clz returns the number of leading 0-bits in argument, starting with
// the most significant bit.  If x is 0, the result is undefined.  The
// builtins make use of processor instructions, and are defined for
// unsigned, unsigned long, and unsigned long long types.
constexpr int clz(unsigned x) { return __builtin_clz(x); }

template <typename T>
constexpr typename std::enable_if<enable_builtin_long<T>::value, int>::type
clz(T x) { return __builtin_clzl(x); }

template <typename T>
constexpr typename std::enable_if<enable_builtin_long_long<T>::value, int>::type
clz(T x) { return __builtin_clzll(x); }

template <typename UInt>
constexpr int log2(UInt x) { return sizeof(x)*8 - 1 - clz(x); }


}}}}

#include "_l2space.hpp"
#include "_so3space.hpp"
#include "_wtspace.hpp"
#include "_compoundspace.hpp"

#endif // UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
