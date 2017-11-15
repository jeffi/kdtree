// Copyright (c) 2017 Jeffrey Ichnowski
// All rights reserved.
//
// BSD 3 Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP

#include "_spaces.hpp"

namespace unc { namespace robotics { namespace kdtree {

struct MidpointSplit {};
struct MedianSplit {};

struct DynamicBuild {};
struct StaticBuild {};

struct SingleThread {};
struct MultiThread {};

template <
    typename _T,
    typename _Space,
    typename _GetKey,
    typename _SplitStrategy,
    typename _Construction = DynamicBuild,
    typename _Locking = SingleThread,
    typename _Allocator = std::allocator<_T>>
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

struct CompareSecond {
    template <typename _First, typename _Second>
    constexpr bool operator() (const std::pair<_First,_Second>& a, const std::pair<_First,_Second>& b) const {
        return a.second < b.second;
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


template <typename _Allocator>
struct AllocatorDestructor {
    typedef std::allocator_traits<_Allocator> AllocatorTraits;
    typedef typename AllocatorTraits::pointer pointer;
    typedef typename AllocatorTraits::size_type size_type;

    _Allocator& allocator_;
    size_type count_;

    inline AllocatorDestructor(_Allocator& allocator, size_type count)
        : allocator_(allocator),
          count_(count)
    {
    }

    inline void operator() (pointer p) {
        AllocatorTraits::deallocate(allocator_, p, count_);
    }
};

}}}}

#include "_l2space.hpp"
#include "_so3space.hpp"
#include "_so3altspace.hpp"
#include "_wtspace.hpp"
#include "_compoundspace.hpp"

#endif // UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
