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
#ifndef UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP
#define UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Node, int _index, typename ... _Spaces>
struct MidpointCompoundHelper {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    typedef MidpointCompoundHelper<_Node, _index+1, _Spaces...> Next;

    template <typename _Traversals>
    static constexpr unsigned dimensions(_Traversals& traversals, unsigned sum) {
        return Next::dimensions(traversals, sum + std::get<_index>(traversals).dimensions());
    }
    
    template <typename _Traversals>
    static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
        unsigned axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        if (d > bestDist) {
            *bestAxis = dimBefore + axis;
            bestDist = d;
        }
        return Next::maxAxis(traversals, dimBefore + std::get<_index>(traversals).dimensions(), bestDist, bestAxis);
    }

    template <typename _Traversals, typename _Adder>
    static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            std::get<_index>(traversals).addImpl(adder, axis - dimBefore, p, n);
        } else {
            Next::addImpl(traversals, adder, dimAfter, axis, p, n);
        }
    }

    template <typename _Traversals, typename _State>
    static inline Distance keyDistance(const _Traversals& traversals, const _State& q, Distance sum) {
        return Next::keyDistance(traversals, q, sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q)));
    }

    template <typename _Traversals>
    static inline Distance distToRegion(const _Traversals& traversals, Distance sum) {
        return Next::distToRegion(traversals, sum + std::get<_index>(traversals).distToRegion());
    }

    template <typename _Traversals, typename _Nearest>
    static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore);
        } else {
            Next::traverse(traversals, nearest, n, dimAfter, axis);
        }
    }
};


template <typename _Node, typename ...  _Spaces>
struct MidpointCompoundHelper<_Node, sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;

    template <typename _Traversals>
    static constexpr unsigned dimensions(_Traversals& traversals, unsigned sum) {
        return sum + std::get<_index>(traversals).dimensions();
    }
    
    template <typename _Traversals>
    static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
        unsigned axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        if (d > bestDist) {
            *bestAxis = dimBefore + axis;
            bestDist = d;
        }
        return bestDist;
    }

    template <typename _Traversals, typename _Adder>
    static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
        std::get<_index>(traversals).addImpl(adder, axis - dimBefore, p, n);
    }

    template <typename _Traversals, typename _State>
    static inline Distance keyDistance(_Traversals& traversals, const _State& q, Distance sum) {
        return sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q));
    }

    template <typename _Traversals>
    static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
        return sum + std::get<_index>(traversals).distToRegion();
    }

    template <typename _Traversals, typename _Nearest>
    static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        assert(axis < dimAfter);
        std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore);
    }
};


// Alternate base case for MidpointCompoundHelper, 
//
// template <typename _Node, typename ...  _Spaces>
// struct MidpointCompoundHelper<_Node, sizeof...(_Spaces), _Spaces...> {
//     typedef CompoundSpace<_Spaces...> Space;
//     typedef typename Space::State Key;
//     typedef typename Space::Distance Distance;

//     template <typename _Traversals>
//     static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
//         return bestDist;
//     }
    
//     template <typename _Traversals, typename _Adder>
//     static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
//         assert(false); // should not happen
//     }

//     template <typename _Traversals, typename _State>
//     static inline Distance keyDistance(_Traversals& traversals, const _State& q, Distance sum) {
//         return sum;
//     }

//     template <typename _Traversals>
//     static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
//         return sum;
//     }

//     template <typename _Traversals, typename _Nearest>
//     static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
//         assert(false); // should not happen
//     }
// };

template <typename _Node, typename ... _Spaces>
struct MidpointAddTraversal<_Node, CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple<_Spaces...> Tuple;

    const Space& space_;
    std::tuple<MidpointAddTraversal<_Node, _Spaces>...> traversals_;

    template <std::size_t ... I>
    MidpointAddTraversal(const Space& space, const Key& key, std::index_sequence<I...>)
        : space_(space),
          traversals_(MidpointAddTraversal<_Node, typename std::tuple_element<I, Tuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }

    MidpointAddTraversal(const Space& space, const Key& key)
        : MidpointAddTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    constexpr unsigned dimensions() const {
        return MidpointCompoundHelper<_Node, 0, _Spaces...>::dimensions(traversals_, 0);
    }

    inline Distance maxAxis(unsigned* axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::maxAxis(
            traversals_, std::get<0>(space_).dimensions(), d, axis);
    }

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node* n) {
        MidpointCompoundHelper<_Node, 0, _Spaces...>::addImpl(
            traversals_, adder, 0, axis, p, n);
    }
};

template <typename _Node, typename ... _Spaces>
struct MidpointNearestTraversal<_Node, CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple<_Spaces...> Tuple;

    const Space& space_;
    std::tuple<MidpointNearestTraversal<_Node, _Spaces>...> traversals_;

    template <std::size_t ... I>
    MidpointNearestTraversal(
        const Space& space, const Key& key, std::index_sequence<I...>)
        : space_(space),
          traversals_(MidpointNearestTraversal<_Node, typename std::tuple_element<I, Tuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointNearestTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    constexpr unsigned dimensions() const {
        return MidpointCompoundHelper<_Node, 0, _Spaces...>::dimensions(traversals_, 0);
    }

    template <typename _State>
    Distance keyDistance(const _State& q) const {
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::keyDistance(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    inline Distance distToRegion() const {
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _Nearest>
    void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        MidpointCompoundHelper<_Node, 0, _Spaces...>::traverse(
            traversals_, nearest, n, 0, axis);
    }
};

template <int _index, typename ... _Spaces>
struct CompoundMedianHelper {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef std::tuple<MedianAccum<_Spaces>...> Accums;
    typedef std::tuple<MedianNearestTraversal<_Spaces>...> Traversals;
    typedef CompoundMedianHelper<_index+1, _Spaces...> Next;
    
    template <typename _Accums>
    static unsigned dimensions(_Accums& accums, unsigned sum) {
        return Next::dimensions(accums, sum + std::get<_index>(accums).dimensions());
    }

    static void init(Accums& accums, const Key& q) {
        std::get<_index>(accums).init(std::get<_index>(q));
        return Next::init(accums, q);
    }

    static void accum(Accums& accums, const Key& q) {
        std::get<_index>(accums).accum(std::get<_index>(q));
        return Next::accum(accums, q);
    }

    static Distance maxAxis(Accums& accums, unsigned dimBefore, Distance dist, unsigned *axis) {
        unsigned a;
        Distance d = std::get<_index>(accums).maxAxis(&a);
        if (d > dist) {
            dist = d;
            *axis = a + dimBefore;
        }
        return Next::maxAxis(accums, dimBefore + std::get<_index>(accums).dimensions(), dist, axis);
    }
    
    template <typename _Builder, typename _Iter, typename _GetKey>
    static void partition(
        Accums& accums, _Builder& builder, unsigned axis,
        _Iter begin, _Iter end,
        const _GetKey& getKey)
    {
        unsigned dim = std::get<_index>(accums).dimensions();
        if (axis < dim) {
            std::get<_index>(accums).partition(
                builder, axis, begin, end,
                [&] (auto& t) -> auto& { return std::get<_index>(getKey(t)); });
        } else {
            Next::partition(accums, builder, axis - dim, begin, end, getKey);
        }
    }

    static Distance distToRegion(const Traversals& traversals, Distance sum) {
        return Next::distToRegion(traversals, sum + std::get<_index>(traversals).distToRegion());
    }

    template <typename _Key>
    static Distance keyDistance(const Traversals& traversals, const _Key& q, Distance sum) {
        return Next::keyDistance(
            traversals, q,
            sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q)));
    }

    template <typename _Nearest, typename _Iter>
    static void traverse(Traversals& traversals, _Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        unsigned dim = std::get<_index>(traversals).dimensions();
        if (axis < dim) {
            std::get<_index>(traversals).traverse(nearest, axis, begin, end);
        } else {
            Next::traverse(traversals, nearest, axis - dim, begin, end);
        }
    }
};

template <typename ... _Spaces>
struct CompoundMedianHelper<sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef std::tuple<MedianAccum<_Spaces>...> Accums;
    typedef std::tuple<MedianNearestTraversal<_Spaces>...> Traversals;
    static constexpr int _index = sizeof...(_Spaces)-1;

    template <typename _Accums>
    static unsigned dimensions(_Accums& accums, unsigned sum) {
        return sum + std::get<_index>(accums).dimensions();
    }

    static void init(Accums& accums, const Key& q) {
        std::get<_index>(accums).init(std::get<_index>(q));
    }

    static void accum(Accums& accums, const Key& q) {
        std::get<_index>(accums).accum(std::get<_index>(q));
    }

    static Distance maxAxis(Accums& accums, unsigned dimBefore, Distance dist, unsigned *axis) {
        unsigned a;
        Distance d = std::get<_index>(accums).maxAxis(&a);
        if (d > dist) {
            dist = d;
            *axis = a + dimBefore;
        }
        return dist;
    }
    
    template <typename _Builder, typename _Iter, typename _GetKey>
    static void partition(
        Accums& accums, _Builder& builder, unsigned axis,
        _Iter begin, _Iter end,
        const _GetKey& getKey)
    {
        std::get<_index>(accums).partition(
            builder, axis, begin, end,
            [&] (auto& t) -> auto& { return std::get<_index>(getKey(t)); });
    }

    static Distance distToRegion(const Traversals& traversals, Distance sum)  {
        return sum + std::get<_index>(traversals).distToRegion();
    }

    template <typename _Key>
    static Distance keyDistance(const Traversals& traversals, const _Key& q, Distance sum) {
        return sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q));
    }

    template <typename _Nearest, typename _Iter>
    static void traverse(Traversals& traversals, _Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        std::get<_index>(traversals).traverse(nearest, axis, begin, end);
    }

};

template <typename ... _Spaces>
struct MedianAccum<CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    std::tuple<MedianAccum<_Spaces>...> accums_;

    template <std::size_t ... I>
    MedianAccum(const Space& space, std::index_sequence<I...>)
        : accums_(std::get<I>(space)...)
    {
    }
    
    MedianAccum(const Space& space)
        : MedianAccum(space, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    constexpr unsigned dimensions() const {
        return CompoundMedianHelper<0, _Spaces...>::dimensions(accums_, 0);
    }

    template <typename _Key>
    void init(const _Key& q) {
        CompoundMedianHelper<0, _Spaces...>::init(accums_, q);
    }

    template <typename _Key>
    void accum(const _Key& q) {
        CompoundMedianHelper<0, _Spaces...>::accum(accums_, q);
    }

    Distance maxAxis(unsigned *axis) {
        Distance d = std::get<0>(accums_).maxAxis(axis);
        return CompoundMedianHelper<1, _Spaces...>::maxAxis(
            accums_, std::get<0>(accums_).dimensions(), d, axis);
    }

    template <typename _Builder, typename _Iter, typename _GetKey>
    void partition(_Builder& builder, unsigned axis, _Iter begin, _Iter end, const _GetKey& getKey) {
        CompoundMedianHelper<0, _Spaces...>::partition(
            accums_, builder, axis, begin, end, getKey);
    }
};

template <typename ... _Spaces>
struct MedianNearestTraversal<CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    
    typedef std::tuple<_Spaces...> Tuple;

    std::tuple<MedianNearestTraversal<_Spaces>...> traversals_;

    template <std::size_t ... I>
    MedianNearestTraversal(const Space& space, const Key& key, std::index_sequence<I...>)
        : traversals_(
            MedianNearestTraversal<typename std::tuple_element<I, Tuple>::type>(
                std::get<I>(space),
                std::get<I>(key))...)
    {
    }
    
    MedianNearestTraversal(const Space& space, const Key& key)
        : MedianNearestTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    constexpr unsigned dimensions() const {
        return CompoundMedianHelper<0, _Spaces...>::dimensions(traversals_, 0);
    }
    
    Distance distToRegion() const {
        return CompoundMedianHelper<1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _State>
    Distance keyDistance(const _State& q) const {
        return CompoundMedianHelper<1, _Spaces...>::keyDistance(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        CompoundMedianHelper<0, _Spaces...>::traverse(
            traversals_, nearest, axis, begin, end);
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP
