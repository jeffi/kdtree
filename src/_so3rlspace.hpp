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
#ifndef UNC_ROBOTICS_KDTREE_SO3RLSPACE_HPP
#define UNC_ROBOTICS_KDTREE_SO3RLSPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Scalar>
struct MidpointSO3RLTraversalBase {
    typedef SO3RLSpace<_Scalar> Space;
    typedef typename Space::State Key;

    Key key_;
    Eigen::Array<_Scalar, 4, 2> bounds_;
    
    MidpointSO3RLTraversalBase(const Space& space, const Key& key)
        : key_(key)
    {
        bounds_.col(0) = -1;
        bounds_.col(1) = 1;
    }

    constexpr unsigned dimensions() const {
        return 4;
    }
};

template <typename _Node, typename _Scalar>
struct MidpointAddTraversal<_Node, SO3RLSpace<_Scalar>>
    : MidpointSO3RLTraversalBase<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3RLSpace<Scalar> Space;
    typedef typename Space::State Key;

    using MidpointSO3RLTraversalBase<_Scalar>::bounds_;
    using MidpointSO3RLTraversalBase<_Scalar>::MidpointSO3RLTraversalBase;

    constexpr _Scalar maxAxis(unsigned *axis) const {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis) * M_PI_2;
    }

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node *n) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;
        int childNo = split < this->key_.coeffs()[axis];
        _Node *c = _Adder::child(p, childNo);
        while (c == nullptr)
            if (_Adder::update(p, childNo, c, n))
                return;
        bounds_(axis, 1-childNo) = split;
        adder(c, n);
    }
};

template <typename _Scalar>
_Scalar distSide2(_Scalar min, _Scalar pt, _Scalar max) {
    _Scalar d;
    if (pt < min) {
        d = min - pt;
    } else if (pt < max) {
        return 0;
    } else {
        d = max - pt;
    }
    return d*d;
}

template <typename _Derived1, typename _Derived2, typename _Derived3>
inline auto distPtRect(const Eigen::DenseBase<_Derived1>& min,
                       const Eigen::DenseBase<_Derived2>& max,
                       const Eigen::DenseBase<_Derived3>& q)
{
    return distSide2(min[0], q[0], max[0])
        +  distSide2(min[1], q[1], max[1])
        +  distSide2(min[2], q[2], max[2])
        +  distSide2(min[3], q[3], max[3]);
}


template <typename _Min, typename _Max, typename _Split>
auto so3RLdistPointRect(
    const Eigen::ArrayBase<_Min>& min,
    const Eigen::ArrayBase<_Max>& max,
    const Eigen::QuaternionBase<_Split>& split)
{
    const auto& pt = split.coeffs().array();

    //   -2  1
    // -3        => 1
    //         3 => 2
    //     0     => 0
    // 
    // (-2 - -3).max(-3 - 1).max(0) = 1
    // (-2 -  3).max( 3 - 1).max(0) = 2
    // (-2 -  0).max( 0 - 1).max(0) = 0

    // std::cout << (min - pt).max(pt - max).max(0).transpose() << std::endl
    //           << (min + pt).max(-pt - max).max(0).transpose() << std::endl;
            
    auto r = std::min(
        (min - pt).max(pt - max).max(0).matrix().squaredNorm(),
        (min + pt).max(-pt -max).max(0).matrix().squaredNorm());

    // auto c = std::min(distPtRect(min, max, pt),
    //                   distPtRect(min, max, -pt));


    // if (std::abs(r - c) > 1e-5) {
    //     std::cout << std::abs(r - c) << std::endl;
    //     abort();
    // }

    return r;
}

template <typename _Bounds, typename _Split>
auto so3RLdistPointRect(
    const Eigen::ArrayBase<_Bounds>& bounds,
    const Eigen::QuaternionBase<_Split>& split)
{
    return so3RLdistPointRect(bounds.col(0), bounds.col(1), split);
}

// TODO: use this for bounds update instead of std::swap or similar.
template <typename _Scalar>
struct PushVal {
    _Scalar& var_;
    _Scalar prev_;

    PushVal(_Scalar& v, _Scalar n) : var_(v), prev_(v) { var_ = n; }
    ~PushVal() { var_ = prev_; }
};

template <typename _Node, typename _Scalar>
struct MidpointNearestTraversal<_Node, SO3RLSpace<_Scalar>>
    : MidpointSO3RLTraversalBase<_Scalar>
{
    typedef SO3RLSpace<_Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    
    Distance distToRegionCache_ = 0;

    using MidpointSO3RLTraversalBase<_Scalar>::key_;
    using MidpointSO3RLTraversalBase<_Scalar>::bounds_;

    template <typename _Derived>
    MidpointNearestTraversal(const Space& space, const Eigen::QuaternionBase<_Derived>& key)
        : MidpointSO3RLTraversalBase<_Scalar>(space, key)
    {
    }

    template <typename _Derived>
    constexpr Distance keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        _Scalar dot = std::abs(key_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Nearest>
    inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;

        std::swap(bounds_(axis, 1), split);
        Distance d0 = so3RLdistPointRect(bounds_, key_);
        std::swap(bounds_(axis, 1), split);
        
        std::swap(bounds_(axis, 0), split);
        Distance d1 = so3RLdistPointRect(bounds_, key_);
        std::swap(bounds_(axis, 0), split);

        int childNo = d0 > d1;

        if (const _Node* c = _Nearest::child(n, childNo)) {
            std::swap(bounds_(axis, 1-childNo), split);
            nearest(c);
            std::swap(bounds_(axis, 1-childNo), split);
        }
        
        nearest.update(n);

        if (const _Node* c = _Nearest::child(n, 1-childNo)) {
            Distance oldDist = distToRegionCache_;
            distToRegionCache_ = oldDist + std::abs(d1 - d0);
            if (nearest.shouldTraverse()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c);
                std::swap(bounds_(axis, childNo), split);
            }
            distToRegionCache_ = oldDist;
        }
    }
};

template <typename _Scalar>
struct MedianAccum<SO3RLSpace<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3RLSpace<Scalar> Space;

    Eigen::Array<Scalar, 4, 1> min_;
    Eigen::Array<Scalar, 4, 1> max_;

    inline MedianAccum(const Space& space) {
        min_ = -1;
        max_ = 1;
    }

    constexpr unsigned dimensions() const {
        return 4;
    }

    template <typename _Derived>
    void init(const Eigen::QuaternionBase<_Derived>& q) {
        min_ = q.coeffs();
        max_ = q.coeffs();
    }

    template <typename _Derived>
    void accum(const Eigen::QuaternionBase<_Derived>& q) {
        min_ = min_.min(q.coeffs().array());
        max_ = max_.max(q.coeffs().array());
    }

    constexpr Scalar maxAxis(unsigned *axis) const {
        return (max_ - min_).maxCoeff(axis);
    }

    template <typename _Builder, typename _Iter, typename _GetKey>
    void partition(_Builder& builder, unsigned axis, _Iter begin, _Iter end, const _GetKey& getKey) {
        _Iter mid = begin + (std::distance(begin, end) - 1)/2;
        std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
            return getKey(a).coeffs()[axis] < getKey(b).coeffs()[axis];
        });
        std::iter_swap(begin, mid);
        _Builder::setSplit(*begin, getKey(*begin).coeffs()[axis]);
        builder(++begin, ++mid);
        builder(mid, end);
    }
};

template <typename _Scalar>
struct MedianNearestTraversal<SO3RLSpace<_Scalar>> {
    typedef SO3RLSpace<_Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    const Key key_;
    Eigen::Array<_Scalar, 4, 2> bounds_;
    Distance distToRegionCache_;

    template <typename _Derived>
    MedianNearestTraversal(const Space& space, const Eigen::QuaternionBase<_Derived>& key)
        : key_(key), distToRegionCache_(0)
    {
        bounds_.col(0) = -1;
        bounds_.col(1) = 1;
    }

    constexpr unsigned dimensions() const {
        return 4;
    }

    Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Derived>
    Distance keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        _Scalar dot = std::abs(key_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        const auto& n = *begin++;
        Distance split = _Nearest::split(n);
        std::array<_Iter, 3> iters{{begin, begin + std::distance(begin,end)/2, end}};


        std::swap(bounds_(axis, 1), split);
        Distance d0 = so3RLdistPointRect(bounds_, key_);
        std::swap(bounds_(axis, 1), split);
        
        std::swap(bounds_(axis, 0), split);
        Distance d1 = so3RLdistPointRect(bounds_, key_);
        std::swap(bounds_(axis, 0), split);

        int childNo = d0 > d1;

        std::swap(bounds_(axis, 1-childNo), split);
        nearest(iters[childNo], iters[childNo+1]);
        std::swap(bounds_(axis, 1-childNo), split);
        
        nearest.update(n);

        Distance oldDist = distToRegionCache_;
        distToRegionCache_ += std::abs(d1 - d0);
        if (nearest.shouldTraverse()) {
            std::swap(bounds_(axis, childNo), split);
            nearest(iters[1-childNo], iters[2-childNo]);
            std::swap(bounds_(axis, childNo), split);
        }
        distToRegionCache_ = oldDist;
    }
          
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_SO3RLSPACE_HPP
