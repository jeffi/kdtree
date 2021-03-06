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
#ifndef UNC_ROBOTICS_KDTREE_L2SPACE_HPP
#define UNC_ROBOTICS_KDTREE_L2SPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Scalar, int _dimensions>
struct MidpointBoundedL2TraversalBase {
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;

    const Key key_;
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;

    template <typename _Derived>
    inline MidpointBoundedL2TraversalBase(const Space& space, const Eigen::MatrixBase<_Derived>& key)
        : key_(key), bounds_(space.bounds())
    {
    }

    constexpr unsigned dimensions() const {
        return _dimensions;
    }
};

template <typename _Scalar>
struct MidpointBoundedL2TraversalBase<_Scalar, Eigen::Dynamic> {
    typedef BoundedL2Space<_Scalar, Eigen::Dynamic> Space;
    typedef typename Space::State Key;

    const Key& key_;
    Eigen::Array<_Scalar, Eigen::Dynamic, 2> bounds_;
    unsigned dimensions_;

    template <typename _Derived>
    inline MidpointBoundedL2TraversalBase(const Space& space, const Eigen::MatrixBase<_Derived>& key)
        : key_(key),
          bounds_(space.bounds()),
          dimensions_(space.dimensions())
    {
    }

    constexpr unsigned dimensions() const {
        return dimensions_;
    }
};


template <typename _Node, typename _Scalar, int _dimensions>
struct MidpointAddTraversal<_Node, BoundedL2Space<_Scalar, _dimensions>>
    : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::bounds_;
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::MidpointBoundedL2TraversalBase;

    constexpr _Scalar maxAxis(unsigned *axis) const {
        return (this->bounds_.col(1) - this->bounds_.col(0)).maxCoeff(axis);
    }

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node *n) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;
        int childNo = (split - this->key_[axis]) < 0;
        _Node* c = _Adder::child(p, childNo);
        while (c == nullptr)
            if (_Adder::update(p, childNo, c, n))
                return;

        bounds_(axis, 1-childNo) = split;
        adder(c, n);
    }
};

template <typename _Node, typename _Scalar, int _dimensions>
struct MidpointNearestTraversal<_Node, BoundedL2Space<_Scalar, _dimensions>>
    : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::key_;
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::bounds_;

    _Scalar distToRegionCache_ = 0;
    _Scalar distToRegionSum_ = 0;
    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;

    template <typename _Derived>
    MidpointNearestTraversal(const Space& space, const Eigen::MatrixBase<_Derived>& key)
        : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>(space, key),
          regionDeltas_(space.dimensions(), 1)
    {
        regionDeltas_.setZero();
    }

    template <typename _Derived>
    constexpr _Scalar keyDistance(const Eigen::MatrixBase<_Derived>& q) const {
        return (this->key_ - q).norm();
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_; // std::sqrt(regionDeltas_.sum());
    }

    template <typename _Nearest>
    inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;
        _Scalar delta = (split - key_[axis]);
        int childNo = delta < 0;

        if (const _Node* c = _Nearest::child(n, childNo)) {
            std::swap(bounds_(axis, 1-childNo), split);
            nearest(c);
            std::swap(bounds_(axis, 1-childNo), split);            
        }

        nearest.update(n);

        if (const _Node* c = _Nearest::child(n, 1-childNo)) {
            Distance oldDelta = regionDeltas_[axis];
            Distance oldSum = distToRegionSum_;
            Distance oldDist = distToRegionCache_;
            delta *= delta;
            regionDeltas_[axis] = delta;
            distToRegionSum_ = distToRegionSum_ - oldDelta + delta;
            distToRegionCache_ = std::sqrt(distToRegionSum_);
            if (nearest.shouldTraverse()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c);
                std::swap(bounds_(axis, childNo), split);
            }
            regionDeltas_[axis] = oldDelta;
            distToRegionSum_ = oldSum;
            distToRegionCache_ = oldDist;
        }
    }
};

template <typename _Scalar, int _dimensions>
struct MedianAccum<L2Space<_Scalar, _dimensions>> {
    typedef _Scalar Scalar;
    typedef L2Space<Scalar, _dimensions> Space;
    
    Eigen::Array<_Scalar, _dimensions, 1> min_;
    Eigen::Array<_Scalar, _dimensions, 1> max_;

    inline MedianAccum(const Space& space)
        : min_(space.dimensions()),
          max_(space.dimensions())
    {
    }

    constexpr unsigned dimensions() const {
        return min_.rows();
    }

    template <typename _Derived>
    void init(const Eigen::MatrixBase<_Derived>& q) {
        min_ = q;
        max_ = q;
    }

    template <typename _Derived>
    void accum(const Eigen::MatrixBase<_Derived>& q) {
        min_ = min_.min(q.array());
        max_ = max_.max(q.array());
    }

    constexpr Scalar maxAxis(unsigned *axis) const {
        return (max_ - min_).maxCoeff(axis);
    }

    template <typename _Builder, typename _Iter, typename _GetKey>
    void partition(_Builder& builder, unsigned axis, _Iter begin, _Iter end, const _GetKey& getKey) {
        _Iter mid = begin + (std::distance(begin, end)-1)/2;
        std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
            return getKey(a)[axis] < getKey(b)[axis];
        });
        std::iter_swap(begin, mid);
        
        _Builder::setSplit(*begin, getKey(*begin)[axis]);
        // begin->split_ = getKey(*begin)[axis];
        
        builder(++begin, ++mid);
        builder(mid, end);
    }
};

template <typename _Scalar, int _dimensions>
struct MedianAccum<BoundedL2Space<_Scalar, _dimensions>>
    : MedianAccum<L2Space<_Scalar, _dimensions>>
{
    using MedianAccum<L2Space<_Scalar, _dimensions>>::MedianAccum;
};

template <typename _Scalar, int _dimensions>
struct MedianNearestTraversal<L2Space<_Scalar, _dimensions>> {
    typedef L2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    
    const Key key_;

    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;

    template <typename _Derived>
    MedianNearestTraversal(const Space& space, const Eigen::MatrixBase<_Derived>& key)
        : key_(key),
          regionDeltas_(space.dimensions())
    {
        regionDeltas_.setZero();
    }

    constexpr unsigned dimensions() const {
        return regionDeltas_.rows();
    }

    Distance distToRegion() const {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Derived>
    Distance keyDistance(const Eigen::MatrixBase<_Derived>& q) const {
        return (key_ - q).norm();
    }
    
    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        const auto& n = *begin++;
        std::array<_Iter, 3> iters{{begin, begin + std::distance(begin, end)/2, end}};
        Distance delta = _Nearest::split(n) - key_[axis];
        int childNo = delta < 0;
        nearest(iters[childNo], iters[childNo+1]);
        nearest.update(n);
        delta *= delta;
        std::swap(regionDeltas_[axis], delta);
        if (nearest.shouldTraverse())
            nearest(iters[1-childNo], iters[2-childNo]);
        regionDeltas_[axis] = delta;
    }
};

template <typename _Scalar, int _dimensions>
struct MedianNearestTraversal<BoundedL2Space<_Scalar, _dimensions>>
    : MedianNearestTraversal<L2Space<_Scalar, _dimensions>>
{
    using MedianNearestTraversal<L2Space<_Scalar, _dimensions>>::MedianNearestTraversal;
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_L2SPACE_HPP
