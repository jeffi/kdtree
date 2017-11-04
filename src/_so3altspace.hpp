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
#ifndef UNC_ROBOTICS_KDTREE_SO3MINSPACE_HPP
#define UNC_ROBOTICS_KDTREE_SO3MINSPACE_HPP

// The SO(3) space with an alternative traversal strategy.  This one
// is considerbly simpler than the full SO(3) traversal strategy as it
// only requires std::max and std::asin on each midpoint.  This does
// not sacrifice accuracy, but it does not cull out branch traversal
// as tightly as the full traversal--so given an equal overhead
// between the two this should be slower.  However this method has a
// significantly lower overhead, and can thus sometimes be faster.

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Scalar>
struct MidpointSO3MinTraversalBase {
    typedef SO3AltSpace<_Scalar> Space;
    typedef typename Space::State Key;

    Eigen::Matrix<_Scalar, 4, 1> key_;
    std::array<Eigen::Array<_Scalar, 2, 3, Eigen::RowMajor>, 2> soBounds_;
    unsigned soDepth_;
    unsigned keyVol_;

    MidpointSO3MinTraversalBase(const Space& space, const Key& key)
        : soDepth_(2),
          keyVol_(so3VolumeIndex(key))
    {
        key_ = rotateCoeffs(key.coeffs(), keyVol_ + 1);
        if (key_[3] < 0)
            key_ = -key_;

        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<_Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    constexpr unsigned dimensions() const {
        return 3;
    }

    constexpr _Scalar maxAxis(unsigned *axis) const {
        *axis = soDepth_ % 3;
        return M_PI / (1 << (soDepth_ / 3));
    }
};

template <typename _Node, typename _Scalar>
struct MidpointAddTraversal<_Node, SO3AltSpace<_Scalar>>
    : MidpointSO3MinTraversalBase<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3AltSpace<Scalar> Space;
    typedef typename Space::State Key;
    
    using MidpointSO3MinTraversalBase<_Scalar>::soDepth_;
    using MidpointSO3MinTraversalBase<_Scalar>::keyVol_;
    using MidpointSO3MinTraversalBase<_Scalar>::key_;

    MidpointAddTraversal(const Space& space, const Key& key)
        : MidpointSO3MinTraversalBase<_Scalar>(space, key)
    {
    }
    
    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node *n) {
        int childNo;
        _Node *c;
        
        if (soDepth_ < 3) {
            c = _Adder::child(p, childNo = keyVol_ & 1);
            while (c == nullptr)
                if (_Adder::update(p, childNo, c, n))
                    return;
            
            // if ((c = p->children_[childNo = keyVol_ & 1]) == nullptr) {
            //     p->children_[childNo] = n;
            //     return;
            // }
            p = c;

            c = _Adder::child(p, childNo = keyVol_ >> 1);
            while (c == nullptr)
                if (_Adder::update(p, childNo, c, n))
                    return;
            
            // if ((c = p->children_[childNo = keyVol_ >> 1]) == nullptr) {
            //     p->children_[childNo] = n;
            //     return;
            // }
            
            ++soDepth_;
            adder(c, n);
        } else {
            Eigen::Matrix<Scalar, 2, 1> mp = (this->soBounds_[0].col(axis) + this->soBounds_[1].col(axis))
                .matrix().normalized();

            // assert(inSoBounds(keyVol_, 0, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 1, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 2, soBounds_, key_));
                
            Scalar dot = mp[0]*key_[3] + mp[1]*key_[axis];
            // if ((c = p->children_[childNo = (dot > 0)]) == nullptr) {
            //     p->children_[childNo] = n;
            //     return;
            // }
            c = _Adder::child(p, childNo = (dot > 0));
            while (c == nullptr)
                if (_Adder::update(p, childNo, c, n))
                    return;
            
            this->soBounds_[1-childNo].col(axis) = mp;
            ++soDepth_;
            adder(c, n);
        }
    }
};

template <typename _Node, typename _Scalar>
struct MidpointNearestTraversal<_Node, SO3AltSpace<_Scalar>>
    : MidpointSO3MinTraversalBase<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3AltSpace<_Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    using MidpointSO3MinTraversalBase<_Scalar>::soBounds_;
    using MidpointSO3MinTraversalBase<_Scalar>::soDepth_;
    using MidpointSO3MinTraversalBase<_Scalar>::keyVol_;
    using MidpointSO3MinTraversalBase<_Scalar>::key_;

    Key origKey_;
    Distance distToRegionCache_ = 0;

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointSO3MinTraversalBase<_Scalar>(space, key),
          origKey_(key)
    {
    }

    template <typename _Derived>
    inline _Scalar keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        _Scalar dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Derived>
    inline Distance dotBounds(int b, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        // assert(b == 0 || b == 1);
        // assert(0 <= axis && axis < 3);
            
        return soBounds_[b](0, axis)*q[3]
            +  soBounds_[b](1, axis)*q[axis];
    }

    Distance initialBounds() {
        Distance d = 0;
        for (int a=0 ; a<3 ; ++a) {
            Distance d0 = dotBounds(0, a, key_);
            Distance d1 = dotBounds(1, a, key_);
            if (d0 < 0 || d1 > 0)
                d = std::max(d, std::min(-d0, d1)); // std::min(std::abs(d0), std::abs(d1)));
        }
        return std::asin(d);
    }

    template <typename _Nearest>
    inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        if (soDepth_ < 3) {
            ++soDepth_;
            if (const _Node *c = _Nearest::child(n, keyVol_ & 1)) {
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const _Node *g = _Nearest::child(c, keyVol_ >> 1)) {
                    // assert(std::abs(origKey_.coeffs()[keyVol_]) == key_[3]);
                    nearest(g);
                }
                // TODO: can we gain so efficiency by exploring the
                // nearest of the remaining 3 volumes first?
                nearest.update(c);
                if (const _Node *g = _Nearest::child(c, 1 - (keyVol_ >> 1))) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 2) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 2]) == key_[3]);
                    distToRegionCache_ = initialBounds();
                    if (nearest.shouldTraverse())
                        nearest(g);
                }
            }
            nearest.update(n);
            if (const _Node *c = _Nearest::child(n, 1 - (keyVol_ & 1))) {
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const _Node *g = _Nearest::child(c, keyVol_ >> 1)) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 1) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 1]) == key_[3]);
                    distToRegionCache_ = initialBounds();
                    if (nearest.shouldTraverse())
                        nearest(g);
                }
                nearest.update(c);
                if (const _Node *g = _Nearest::child(c, 1 - (keyVol_ >> 1))) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 3) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 3]) == key_[3]);
                    distToRegionCache_ = initialBounds();
                    if (nearest.shouldTraverse())
                        nearest(g);
                }
            }
            // setting vol_ to keyVol_ is only needed when part of a compound space
            // if (key_[vol_ = keyVol_] < 0)
            //     key_ = -key_;
            distToRegionCache_ = 0;
            key_ = rotateCoeffs(origKey_.coeffs(), keyVol_ + 1);
            if (key_[3] < 0)
                key_ = -key_;
            --soDepth_;
            // assert(distToRegion() == 0);
            // assert(soDepth_ == 2);
        } else {
            Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(axis) + soBounds_[1].col(axis))
                .matrix().normalized();
            Scalar dot = mp[0]*key_[3]
                +        mp[1]*key_[axis];
            ++soDepth_;
            int childNo = (dot > 0);
            if (const _Node *c = _Nearest::child(n, childNo)) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(axis);
                soBounds_[1-childNo].col(axis) = mp;
// #ifdef KD_PEDANTIC
//                 Scalar soBoundsDistNow = soBoundsDist();
//                 if (soBoundsDistNow + rvBoundsDistCache_ <= dist_) {
//                     std::swap(soBoundsDistNow, soBoundsDistCache_);
// #endif
                nearest(c);
// #ifdef KD_PEDANTIC
//                     soBoundsDistCache_ = soBoundsDistNow;
//                 }
// #endif
                soBounds_[1-childNo].col(axis) = tmp;
            }
            nearest.update(n);
            if (const _Node *c = _Nearest::child(n, 1-childNo)) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(axis);
                soBounds_[childNo].col(axis) = mp;
                Scalar distToSplit = std::asin(std::abs(dot));
                Scalar oldDistToRegion = distToRegionCache_;
                distToRegionCache_ = std::max(oldDistToRegion, distToSplit);
                // distToRegionCache_ = computeDistToRegion();
                if (nearest.shouldTraverse())
                    nearest(c);
                distToRegionCache_ = oldDistToRegion;
                soBounds_[childNo].col(axis) = tmp;
            }
            --soDepth_;
        }
    }
};

template <typename _Scalar>
struct MedianAccum<SO3AltSpace<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3AltSpace<_Scalar> Space;

    Eigen::Array<Scalar, 2, 3> min_;
    Eigen::Array<Scalar, 2, 3> max_;

    int vol_ = -1;

    MedianAccum(const Space& space) {}

    constexpr unsigned dimensions() const {
        return 3;
    }

    template <typename _Derived>
    void init(const Eigen::QuaternionBase<_Derived>& q) {
        if (vol_ < 0) return;
        for (unsigned axis = 0 ; axis<3 ; ++axis)
            min_.col(axis) = max_.col(axis) = projectToAxis(q, vol_, axis);
    }

    template <typename _Derived>
    void accum(const Eigen::QuaternionBase<_Derived>& q) {
        if (vol_ < 0) return;
        for (unsigned axis = 0 ; axis<3 ; ++axis) {
            Eigen::Matrix<Scalar, 2, 1> split = projectToAxis(q, vol_, axis);
            if (split[0] < min_(0, axis))
                min_.col(axis) = split;
            if (split[0] > max_(0, axis))
                max_.col(axis) = split;
        }
    }

    constexpr Scalar maxAxis(unsigned *axis) const {
        if (vol_ < 0) {
            *axis = 0;
            return M_PI;
        } else {
            // Compute:
            //   (x_min * x_max) + (w_min * w_max) for wach axis
            //
            // This is the dot product between the min and max
            // boundaries.  By finding the minimum we find the maximum
            // acos distance.

            return (min_ * max_).colwise().sum().minCoeff(axis);
        }
    }

    template <typename _Builder, typename _Iter, typename _GetKey>
    void partition(_Builder& builder, unsigned axis, _Iter begin, _Iter end, const _GetKey& getKey) {
        if (vol_ < 0) {
            if (std::distance(begin, end) < 4) {
                for (_Iter it = begin ; it != end ; ++it)
                    _Builder::setOffset(*it, 0);
                return;
            }

            // radix sort into 4 partitions, one for each volume
            Eigen::Array<std::size_t, 4, 1> counts;
            counts.setZero();

            for (_Iter it = begin ; it != end ; ++it)
                counts[so3VolumeIndex(getKey(*it))]++;

            std::array<_Iter, 4> its;
            std::array<_Iter, 3> stops;
            its[0] = begin;
            for (int i=0 ; i<3 ; ++i)
                its[i+1] = stops[i] = its[i] + counts[i];
            assert(its[3]+counts[3] == end);
            for (int i=0 ; i<3 ; ++i)
                for (int v ; its[i] != stops[i] ; ++(its[v]))
                    if ((v = so3VolumeIndex(getKey(*its[i]))) != i)
                        std::iter_swap(its[i], its[v]);

            // after sorting, organize the range s.t. the first 3
            // elements are roots of a tree of 4 volumes.  This makes
            // use of the offset_ member of the union to determine
            // where the subtrees split.
            
            // [begin q0                                             end)
            // begin [q0 ..                q2) [q2 ..                end)
            // begin  q0 (q0 .. q1) [q1 .. q2)  q2 (q2 .. q3) [q3 .. end)
            
            // select the volume with the most elements to be the root
            // this will help balance the subtrees out.

            for (int i=0, v ; i<3 ; ++i) {
                counts.maxCoeff(&v);

                for (int j=0 ; j<v ; ++j)
                    std::iter_swap(begin+i, stops[j]++);

                counts[v]--;
            }

            for (int i=0 ; i<3 ; ++i)
                _Builder::setOffset(begin[i], std::distance(begin, stops[i]));

            assert(begin+3 <= stops[0]);
            assert(stops[0] <= stops[1]);
            assert(stops[1] <= stops[2]);
            assert(stops[2] <= end);

            // std::cout << "=== " << std::distance(begin, end) << " @ " << &*begin << std::endl;
            // for (int i=0 ; i<3 ; ++i)
            //     std::cout << "B " << std::distance(begin, stops[i]) << std::endl;

            vol_ = 0; builder(begin+3, stops[0]);
            vol_ = 1; builder(stops[0], stops[1]);
            vol_ = 2; builder(stops[1], stops[2]);
            vol_ = 3; builder(stops[2], end);
            vol_ = -1;
        } else {
            // in one of the 4 volumes
            
            _Iter mid = begin + (std::distance(begin, end)-1)/2;
            std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
                Eigen::Matrix<Scalar, 2, 1> aProj = projectToAxis(getKey(a), vol_, axis);
                Eigen::Matrix<Scalar, 2, 1> bProj = projectToAxis(getKey(b), vol_, axis);
                return aProj[0] < bProj[0];
            });
            std::iter_swap(begin, mid);
            Eigen::Matrix<Scalar, 2, 1> split = projectToAxis(getKey(*begin), vol_, axis);

            // split[0] may be positive or negative, whereas split[1]
            // is always non-negative.  Given that, split.norm() == 1,
            // we only need to store split[0] and can recomput
            // split[1] from it when necessary.
            _Builder::setSplit(*begin, split[0]);

            ++mid;

            builder(begin+1, mid);
            builder(mid, end);
        }
    }
};

template <typename _Scalar>
struct MedianNearestTraversal<SO3AltSpace<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3AltSpace<Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    Key key_;
    int keyVol_;
    int vol_ = -1;
    Distance distToRegionCache_;

    std::array<Eigen::Array<Scalar, 2, 3, Eigen::RowMajor>, 2> soBounds_;

    MedianNearestTraversal(const Space& space, const Key& key)
        : key_(key),
          keyVol_(so3VolumeIndex(key))
    {
        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    constexpr unsigned dimensions() const {
        return 3;
    }

    template <typename _Derived>
    Distance keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        Distance dot = std::abs(key_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Derived>
    inline Scalar dotBounds(int b, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        // assert(b == 0 || b == 1);
        // assert(0 <= axis && axis < 3);
        assert(q[vol_] >= 0);
        return soBounds_[b](0, axis)*q[vol_]
            +  soBounds_[b](1, axis)*q[(vol_ + axis + 1)%4];
    }


    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        if (vol_ < 0) {
            if (std::distance(begin, end) < 4) {
                for (_Iter it = begin ; it != end ; ++it)
                    nearest.update(*it);
                return;
            }

            std::array<_Iter, 5> iters{{
                begin + 3,
                begin + _Nearest::offset(begin[0]),
                begin + _Nearest::offset(begin[1]),
                begin + _Nearest::offset(begin[2]),
                end
            }};

            // std::cout << "--- " << std::distance(begin, end) << " @ " << &*begin << std::endl;
            // for (int i=0 ; i<3 ; ++i)
            //     std::cout << _Nearest::offset(begin[i]) << std::endl;

            for (int i=0 ; i<4 ; ++i)
                assert(std::distance(iters[i], iters[i+1]) >= 0);

            for (int v=0 ; v<4 ; ++v) {
                if (key_.coeffs()[vol_ = (keyVol_ + v)%4] < 0)
                    key_.coeffs() = -key_.coeffs();
                // TODO: add back
                // if (v != 0)
                //     distToRegionCache_ = computeDistToRegion();
                if (v) {
                    Distance d = 0;
                    for (unsigned a=0 ; a<3 ; ++a) {
                        Distance d0 = dotBounds(0, a, key_.coeffs());
                        Distance d1 = dotBounds(1, a, key_.coeffs());
                        if (d0 < 0 || d1 > 0)
                            d = std::max(d, std::min(-d0, d1)); // std::abs(d0), std::abs(d1)));
                    }
                    distToRegionCache_ = std::asin(d);
                }
                
                if (nearest.shouldTraverse()) {
                    // std::cout << "q" << v << ": " << std::distance(iters[vol_], iters[vol_+1]) << std::endl;
                    nearest(iters[vol_], iters[vol_+1]);
                }
            }
            vol_ = -1;
            distToRegionCache_ = 0;

            for (int i=0 ; i<3 ; ++i)
                nearest.update(begin[i]);
            
        } else {
            const auto& n = *begin++;

            _Iter mid = begin + std::distance(begin, end)/2;
            // std::cout << std::distance(begin, end) << " " << std::distance(begin, mid) << std::endl;
            assert(std::distance(begin, mid) >= 0);
            assert(std::distance(mid, end) >= 0);
            Distance q0 = key_.coeffs()[vol_];
            Distance qa = key_.coeffs()[(vol_ + axis + 1)%4];

            Eigen::Matrix<Scalar, 2, 1> split;
            split[0] = _Nearest::split(n);
            split[1] = std::sqrt(1 - split[0]*split[0]);

            Distance dot = split[0] * q0 + split[1] * qa;
            int childNo = (dot > 0);

            Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(axis);
            soBounds_[1-childNo].col(axis) = split;
            if (nearest.shouldTraverse()) {
                if (childNo) {
                    nearest(begin, mid);
                } else {
                    nearest(mid, end);
                }
            }
            soBounds_[1-childNo].col(axis) = tmp;
            
            tmp = soBounds_[childNo].col(axis);
            soBounds_[childNo].col(axis) = split;
            Scalar prevDistToRegion = distToRegionCache_;
            Scalar distToSplit = std::asin(std::abs(dot));
            distToRegionCache_ = std::max(prevDistToRegion, distToSplit);
            if (nearest.shouldTraverse()) {
                if (childNo) {
                    nearest(mid, end);
                } else {
                    nearest(begin, mid);
                }
            }
            soBounds_[childNo].col(axis) = tmp;
            distToRegionCache_ = prevDistToRegion;

            nearest.update(n);
        }
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_SO3MINSPACE_HPP
