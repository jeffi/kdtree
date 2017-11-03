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
#ifndef UNC_ROBOTICS_KDTREE_WTSPACE_HPP
#define UNC_ROBOTICS_KDTREE_WTSPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Node, typename _Space, typename _Ratio>
struct MidpointAddTraversal<_Node, RatioWeightedSpace<_Space, _Ratio>>
    : MidpointAddTraversal<_Node, _Space>
{
    typedef RatioWeightedSpace<_Space, _Ratio> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    // inherit constructor
    using MidpointAddTraversal<_Node, _Space>::MidpointAddTraversal;

    template <typename _State>
    constexpr Distance keyDistance(const _State& q) const {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    constexpr Distance maxAxis(unsigned *axis) const {
        return MidpointAddTraversal<_Node, _Space>::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Node, typename _Space>
struct MidpointAddTraversal<_Node, WeightedSpace<_Space>>
    : MidpointAddTraversal<_Node, _Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;

    MidpointAddTraversal(const Space& space, const State& key)
        : MidpointAddTraversal<_Node, _Space>(space, key),
          weight_(space.weight())
    {
    }

    template <typename _State>
    constexpr Distance keyDistance(const _State& q) const {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    constexpr Distance maxAxis(unsigned *axis) const {
        return MidpointAddTraversal<_Node, _Space>::maxAxis(axis) * weight_;
    }
};

template <typename _Node, typename _Space, typename _Ratio>
struct MidpointNearestTraversal<_Node, RatioWeightedSpace<_Space, _Ratio>>
    : MidpointNearestTraversal<_Node, _Space>
{
    typedef RatioWeightedSpace<_Space, _Ratio> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    // inherit constructor
    using MidpointNearestTraversal<_Node, _Space>::MidpointNearestTraversal;

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with MidpointAddTraversal.  Would be nice to merge them.
    template <typename _State>
    constexpr Distance keyDistance(const _State& q) const {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    constexpr Distance maxAxis(unsigned *axis) const {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
    
    constexpr Distance distToRegion() const {
        return MidpointNearestTraversal<_Node, _Space>::distToRegion() * _Ratio::num / _Ratio::den;
    }

    // template <typename _Nearest>
    // inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
    //     MidpointNearestTraversal<_Node, _Space>::traverse(nearest, n, axis);
    // }
};

template <typename _Node, typename _Space>
struct MidpointNearestTraversal<_Node, WeightedSpace<_Space>>
    : MidpointNearestTraversal<_Node, _Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;
    
    MidpointNearestTraversal(const Space& space, const State& key)
        : MidpointNearestTraversal<_Node, _Space>(space, key),
          weight_(space.weight())
    {
    }

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with MidpointAddTraversal.  Would be nice to merge them.
    template <typename _State>
    constexpr Distance keyDistance(const _State& q) const {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    constexpr Distance maxAxis(unsigned *axis) const {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * weight_;
    }
    
    constexpr Distance distToRegion() const {
        return MidpointNearestTraversal<_Node, _Space>::distToRegion() * weight_;
    }

    // template <typename _Nearest>
    // void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
    //     MidpointNearestTraversal<_Node, _Space>::traverse(nearest, n, axis);
    // }
};

template <typename _Space, typename _Ratio>
struct MedianAccum<RatioWeightedSpace<_Space, _Ratio>>
    : MedianAccum<_Space>
{
    typedef MedianAccum<_Space> Base;
    typedef typename _Space::Distance Distance;
    
    using Base::Base;
    
    constexpr Distance maxAxis(unsigned *axis) const {
        return Base::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Space, typename _Ratio>
struct MedianNearestTraversal<RatioWeightedSpace<_Space, _Ratio>>
    : MedianNearestTraversal<_Space>
{
    typedef MedianNearestTraversal<_Space> Base;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;

    using Base::Base;
    
    constexpr Distance distToRegion() const {
        return Base::distToRegion() * _Ratio::num / _Ratio::den;
    }

    template <typename _Key>
    constexpr Distance keyDistance(const _Key& key) const {
        return Base::keyDistance(key) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Space>
struct MedianAccum<WeightedSpace<_Space>>
    : MedianAccum<_Space>
{
    typedef MedianAccum<_Space> Base;
    typedef typename _Space::Distance Distance;

    Distance weight_;
    
    MedianAccum(const WeightedSpace<_Space>& space)
        : Base(space),
          weight_(space.weight())
    {
    }
    
    constexpr Distance maxAxis(unsigned *axis) const {
        return Base::maxAxis(axis) * weight_;
    }
};

template <typename _Space>
struct MedianNearestTraversal<WeightedSpace<_Space>>
    : MedianNearestTraversal<_Space>
{
    typedef MedianNearestTraversal<_Space> Base;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;

    Distance weight_;
    
    MedianNearestTraversal(
        const WeightedSpace<_Space>& space,
        const Key& key)
        : Base(space, key),
          weight_(space.weight())
    {
    }

    constexpr Distance distToRegion() const {
        return Base::distToRegion() * weight_;
    }

    template <typename _Key>
    constexpr Distance keyDistance(const _Key& key) const {
        return Base::keyDistance(key) * weight_;
    }

    using Base::traverse;
};
    


}}}}

#endif // UNC_ROBOTICS_KDTREE_WTSPACE_HPP
