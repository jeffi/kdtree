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
#ifndef UNC_ROBOTICS_KDTREE_SO3SPACE_HPP
#define UNC_ROBOTICS_KDTREE_SO3SPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Derived>
unsigned so3VolumeIndex(const Eigen::MatrixBase<_Derived>& q) {
    unsigned index;
    q.array().abs().maxCoeff(&index);
    return index;
}

template <typename _Scalar>
unsigned so3VolumeIndex(const Eigen::QuaternionBase<_Scalar>& q) {
    return so3VolumeIndex(q.coeffs());
}

template <typename _Derived>
Eigen::Matrix<typename _Derived::Scalar, 4, 1> rotateCoeffs(const Eigen::DenseBase<_Derived>& m, unsigned shift) {
    // 0: 0 1 2 3
    // 1: 1 2 3 0
    // 2: 2 3 0 1
    // 3: 3 0 1 2

    return Eigen::Matrix<typename _Derived::Scalar, 4, 1>(
        m[shift%4],
        m[(shift+1)%4],
        m[(shift+2)%4],
        m[(shift+3)%4]);
}

template <typename _Scalar>
class SO3Region {
    typedef _Scalar Scalar;
    
    std::array<Eigen::Array<_Scalar, 2, 3, Eigen::RowMajor>, 2> bounds_;

public:
    SO3Region() {
        bounds_[0] = M_SQRT1_2;
        bounds_[1].colwise() = Eigen::Array<_Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    Eigen::Matrix<_Scalar, 2, 1> midPoint(unsigned axis) {
        return (bounds_[0].col(axis) + bounds_[1].col(axis)).matrix().normalized();
    }

    auto operator() (int which, unsigned axis) {
        return bounds_[which].col(axis);
    }

    template <typename _Derived>
    inline _Scalar dotBounds(int which, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        return bounds_[which](0, axis)*q[3]
            +  bounds_[which](1, axis)*q[axis];
    }
    
    template <typename _Derived>
    inline _Scalar computeDistToRegion(const Eigen::MatrixBase<_Derived>& q) {
        int edgesToCheck = 0;
        
        // check faces
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            Eigen::Matrix<Scalar, 2, 1> dot(dotBounds(0, a0, q), dotBounds(1, a0, q));
            int b0 = dot[0] >= 0;
            if (b0 && dot[1] <= 0)
                continue; // in bounds

            Eigen::Matrix<Scalar, 4, 1> p0 = q;
            p0[3]  -= bounds_[b0](0, a0) * dot[b0];
            p0[a0] -= bounds_[b0](1, a0) * dot[b0];

            int a1 = (a0+1)%3;
            if (dotBounds(1, a1, p0) > 0 || dotBounds(0, a1, p0) < 0) {
                edgesToCheck |= 1 << (a0+a1);
                continue; // not on face with this axis
            }
            int a2 = (a0+2)%3;
            if (dotBounds(1, a2, p0) > 0 || dotBounds(0, a2, p0) < 0) {
                edgesToCheck |= 1 << (a0+a2);
                continue; // not on face with this axis
            }
            // the projected point is on this face, the distance to
            // the projected point is the closest point in the bounded
            // region to the query key.  Use asin of the dot product
            // to the bounding face for the distance, instead of the
            // acos of the dot product to p, since p0 is not
            // normalized for efficiency.
            return std::asin(std::abs(dot[b0]));
        }

        // if the query point is within all bounds of all 3 axes, then it is within the region.
        if (edgesToCheck == 0)
            return 0;

        // int cornerChecked = 0;
        int cornersToCheck = 0;
        Eigen::Matrix<Scalar, 2, 3> T;
        T.row(0) = bounds_[0].row(0) / bounds_[0].row(1);
        T.row(1) = bounds_[1].row(0) / bounds_[1].row(1);
        
        // check edges
        // ++, +-, --, -+ for 01, 12, 20
        Scalar dotMax = 0;
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            int a1 = (a0 + 1)%3;
            int a2 = (a0 + 2)%3;
            
            if ((edgesToCheck & (1 << (a0+a1))) == 0)
                continue;

            for (int edge = 0 ; edge < 4 ; ++edge) {
                int b0 = edge & 1;
                int b1 = edge >> 1;

                Eigen::Matrix<Scalar, 4, 1> p1;
                Scalar t0 = T(b0, a0); // bounds_[b0](0, a0) / bounds_[b0](1, a0);
                Scalar t1 = T(b1, a1); // bounds_[b1](0, a1) / bounds_[b1](1, a1);
                Scalar r = q[3] - t0*q[a0] - t1*q[a1];
                Scalar s = t0*t0 + t1*t1 + 1;

                // bounds check only requires p1[3] and p1[a2], and
                // p1[3] must be non-negative.  If in bounds, then
                // [a0] and [a1] are required to compute the distance
                // to the edge.
                p1[3] = r;
                // p1[a0] = -t0*r;
                // p1[a1] = -t1*r;
                p1[a2] = q[a2] * s;
                
                int b2;
                if ((b2 = dotBounds(0, a2, p1) >= 0) && dotBounds(1, a2, p1) <= 0) {
                    // projection onto edge is in bounds of a2, this
                    // point will be closer than the corners.
                    p1[a0] = -t0*r;
                    p1[a1] = -t1*r;
                    dotMax = std::max(dotMax, std::abs(p1.dot(q)) / p1.norm());
                    continue;
                }
                if (r < 0) b2 = 1-b2;

                int cornerCode = 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
                cornersToCheck |= cornerCode;
                
                // if (cornerChecked & cornerCode)
                //     continue;
                // cornerChecked |= cornerCode;
                // // edge is not in bounds, use the distance to the corner
                // Eigen::Matrix<Scalar, 4, 1> p2;
                // Scalar aw = bounds_[b0](0, a0);
                // Scalar ax = bounds_[b0](1, a0);
                // Scalar bw = bounds_[b1](0, a1);
                // Scalar by = bounds_[b1](1, a1);
                // Scalar cw = bounds_[b2](0, a2);
                // Scalar cz = bounds_[b2](1, a2);

                // p2[a0] =  aw*by*cz;
                // p2[a1] =  ax*bw*cz;
                // p2[a2] =  ax*by*cw;
                // p2[ 3] = -ax*by*cz;

                // // // p2 should be on both bounds
                // // assert(std::abs(dotBounds(b0, a0, p2)) < 1e-7);
                // // assert(std::abs(dotBounds(b1, a1, p2)) < 1e-7);
                // // assert(std::abs(dotBounds(b2, a2, p2)) < 1e-7);
            
                // dotMax = std::max(dotMax, std::abs(q.dot(p2)) / p2.norm());
            }
        }

        for (int i=0 ; i<8 ; ++i) {
            if ((cornersToCheck & (1 << i)) == 0)
                continue;

            int b0 = i&1;
            int b1 = (i>>1)&1;
            int b2 = i>>2;
            
            Eigen::Matrix<Scalar, 4, 1> p2;
            Scalar aw = bounds_[b0](0, 0);
            Scalar ax = bounds_[b0](1, 0);
            Scalar bw = bounds_[b1](0, 1);
            Scalar by = bounds_[b1](1, 1);
            Scalar cw = bounds_[b2](0, 2);
            Scalar cz = bounds_[b2](1, 2);

            p2[0] =  aw*by*cz;
            p2[1] =  ax*bw*cz;
            p2[2] =  ax*by*cw;
            p2[3] = -ax*by*cz;

            // // p2 should be on both bounds
            // assert(std::abs(dotBounds(b0, a0, p2)) < 1e-7);
            // assert(std::abs(dotBounds(b1, a1, p2)) < 1e-7);
            // assert(std::abs(dotBounds(b2, a2, p2)) < 1e-7);
            
            dotMax = std::max(dotMax, std::abs(q.dot(p2)) / p2.norm());
        }
        
        return std::acos(dotMax);
    }

};

template <typename _Scalar>
struct MidpointSO3TraversalBase {
    typedef SO3Space<_Scalar> Space;
    typedef typename Space::State Key;

    Eigen::Matrix<_Scalar, 4, 1> key_;
    SO3Region<_Scalar> soBounds_;
    unsigned soDepth_;
    unsigned keyVol_;

    MidpointSO3TraversalBase(const Space& space, const Key& key)
        : soDepth_(2),
          keyVol_(so3VolumeIndex(key))
    {
        key_ = rotateCoeffs(key.coeffs(), keyVol_ + 1);
        if (key_[3] < 0)
            key_ = -key_;
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
struct MidpointAddTraversal<_Node, SO3Space<_Scalar>>
    : MidpointSO3TraversalBase<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State Key;
    
    using MidpointSO3TraversalBase<_Scalar>::soDepth_;
    using MidpointSO3TraversalBase<_Scalar>::keyVol_;
    using MidpointSO3TraversalBase<_Scalar>::key_;

    MidpointAddTraversal(const Space& space, const Key& key)
        : MidpointSO3TraversalBase<_Scalar>(space, key)
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
            Eigen::Matrix<Scalar, 2, 1> mp = this->soBounds_.midPoint(axis);

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
            
            this->soBounds_(1-childNo, axis) = mp;
            ++soDepth_;
            adder(c, n);
        }
    }
};

template <typename _Node, typename _Scalar>
struct MidpointNearestTraversal<_Node, SO3Space<_Scalar>>
    : MidpointSO3TraversalBase<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3Space<_Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    using MidpointSO3TraversalBase<_Scalar>::soBounds_;
    using MidpointSO3TraversalBase<_Scalar>::soDepth_;
    using MidpointSO3TraversalBase<_Scalar>::keyVol_;
    using MidpointSO3TraversalBase<_Scalar>::key_;

    Key origKey_;
    Distance distToRegionCache_ = 0;

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointSO3TraversalBase<_Scalar>(space, key),
          origKey_(key)
    {
    }

    template <typename _Derived>
    inline _Scalar keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        _Scalar dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    inline Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Nearest>
    inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        if (soDepth_ < 3) {
            ++soDepth_;
            // 27.6 before change

            // 0 -> SO(3)F: 33.538 us/op
            // 0 -> SO(3)F: 33.3035 us/op
            // 1 -> SO(3)F: 33.6403 us/op
            // 1 -> SO(3)F: 33.2282 us/op

            // 0 -> SO(3)F: 44.1291 us/op 44.537 us/op
            // 1.0 ->       44.6697 us/op 50.3254 us/op
            // 1.1 ->       44.9184 us/op 45.0832 us/op 44.7288 us/op
#if 1
            std::array<const _Node*,4> roots{};

            nearest.update(n);
            
            if (const _Node *c = _Nearest::child(n, 0)) {
                nearest.update(c);
                roots[0] = _Nearest::child(c, 0);
                roots[2] = _Nearest::child(c, 1);
            }
            if (const _Node *c = _Nearest::child(n, 1)) {
                nearest.update(c);
                roots[1] = _Nearest::child(c, 0);
                roots[3] = _Nearest::child(c, 1);
            }

            if (const _Node *c = roots[keyVol_]) {
                nearest(c);
            }

#if 1
            std::array<std::pair<unsigned, Distance>, 3> volDists;
            for (unsigned i=1 ; i<4 ; ++i) {
                unsigned vol = volDists[i-1].first = (keyVol_ + i)%4;
                if (roots[vol]) {
                    if ((key_ = rotateCoeffs(origKey_.coeffs(), vol + 1))[3] < 0)
                        key_ = -key_;
                    volDists[i-1].second = soBounds_.computeDistToRegion(key_);
                } else {
                    volDists[i-1].second = std::numeric_limits<Distance>::infinity();
                }
            }

            std::sort(volDists.begin(), volDists.end(), CompareSecond());
            for (int i=0 ; i<3 &&
                     (distToRegionCache_ = volDists[i].second) < std::numeric_limits<Distance>::infinity() ; ++i) {
                unsigned vol = volDists[i].first;
                if (nearest.shouldTraverse()) {
                    if ((key_ = rotateCoeffs(origKey_.coeffs(), vol + 1))[3] < 0)
                        key_ = -key_;
                    nearest(roots[vol]);
                }
            }
#else       
            for (unsigned i=1 ; i<4 ; ++i) {
                unsigned vol = (keyVol_ + i)%4;
                if (const _Node *c = roots[vol]) {
                    key_ = rotateCoeffs(origKey_.coeffs(), vol + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.shouldTraverse())
                        nearest(c);
                }
            }
#endif
            
#else       
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
                    distToRegionCache_ = computeDistToRegion();
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
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.shouldTraverse())
                        nearest(g);
                }
                nearest.update(c);
                if (const _Node *g = _Nearest::child(c, 1 - (keyVol_ >> 1))) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 3) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 3]) == key_[3]);
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.shouldTraverse())
                        nearest(g);
                }
            }
#endif
            
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
            Eigen::Matrix<Scalar, 2, 1> mp = soBounds_.midPoint(axis);
            Scalar dot = mp[0]*key_[3]
                +        mp[1]*key_[axis];
            ++soDepth_;
            int childNo = (dot > 0);
            if (const _Node *c = _Nearest::child(n, childNo)) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_(1-childNo, axis);
                soBounds_(1-childNo, axis) = mp;
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
                soBounds_(1-childNo, axis) = tmp;
            }
            nearest.update(n);
            if (const _Node *c = _Nearest::child(n, 1-childNo)) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_(childNo, axis);
                soBounds_(childNo, axis) = mp;
                Scalar oldDistToRegion = distToRegionCache_;
                distToRegionCache_ = soBounds_.computeDistToRegion(key_);
                if (nearest.shouldTraverse())
                    nearest(c);
                distToRegionCache_ = oldDistToRegion;
                soBounds_(childNo, axis) = tmp;
            }
            --soDepth_;
        }
    }
};

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 2, 1> projectToAxis(
    const Eigen::QuaternionBase<Derived>& q, int vol, int axis)
{
    typedef typename Derived::Scalar Scalar;
    
    Eigen::Matrix<Scalar, 2, 1> vec(-q.coeffs()[(vol + 1 + axis)%4], q.coeffs()[vol]);
    Scalar norm = 1 / vec.norm();
    if (vec[1] < 0) norm = -norm;
    return vec*norm;
}


template <typename _Scalar>
struct MedianAccum<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<_Scalar> Space;

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
                return aProj[0] > bProj[0];
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
struct MedianNearestTraversal<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    Key origKey_;
    Eigen::Matrix<_Scalar, 4, 1> key_;
    int keyVol_;
    Distance distToRegionCache_{0};
    bool atRoot_{true};
    SO3Region<_Scalar> soBounds_;

    MedianNearestTraversal(const Space& space, const Key& key)
        : origKey_(key),
          keyVol_(so3VolumeIndex(key))
    {
    }

    constexpr unsigned dimensions() const {
        return 3;
    }

    template <typename _Derived>
    Distance keyDistance(const Eigen::QuaternionBase<_Derived>& q) const {
        Distance dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_;
    }

    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        if (atRoot_) {
            if (std::distance(begin, end) < 4) {
                for (_Iter it = begin ; it != end ; ++it)
                    nearest.update(*it);
                return;
            }

            atRoot_ = false;
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

            if ((key_ = rotateCoeffs(origKey_.coeffs(), keyVol_ + 1))[3] < 0)
                key_ = -key_;
            
            nearest(iters[keyVol_], iters[keyVol_+1]);

            for (unsigned v=1 ; v<4 ; ++v) {
                unsigned vol = (keyVol_ + v)%4;
                if ((key_ = rotateCoeffs(origKey_.coeffs(), vol + 1))[3] < 0)
                    key_ = -key_;
                distToRegionCache_ = soBounds_.computeDistToRegion(key_);
                if (nearest.shouldTraverse()) {
                    // std::cout << "q" << v << ": " << std::distance(iters[vol_], iters[vol_+1]) << std::endl;
                    nearest(iters[vol], iters[vol+1]);
                }
            }
            distToRegionCache_ = 0;

            for (int i=0 ; i<3 ; ++i)
                nearest.update(begin[i]);

            atRoot_ = true;
        } else {
            const auto& n = *begin++;

            _Iter mid = begin + std::distance(begin, end)/2;
            // std::cout << std::distance(begin, end) << " " << std::distance(begin, mid) << std::endl;
            assert(std::distance(begin, mid) >= 0);
            assert(std::distance(mid, end) >= 0);

            std::array<_Iter,3> iters{{ begin, mid, end }};
            Eigen::Matrix<Scalar, 2, 1> split;
            split[0] = _Nearest::split(n);
            split[1] = std::sqrt(1 - split[0]*split[0]);

            Distance dot = split[0]*key_[3] + split[1]*key_[axis];
            int childNo = (dot > 0);

            nearest.update(n);

            if (iters[childNo] != iters[childNo+1]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_(1-childNo, axis);
                soBounds_(1-childNo, axis) = split;
                nearest(iters[childNo], iters[childNo+1]); // 0 -> (0, 1),  1 -> (1, 2)
                soBounds_(1-childNo, axis) = tmp;
            }

            if (iters[1-childNo] != iters[2-childNo]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_(childNo, axis);
                soBounds_(childNo, axis) = split;
                Scalar prevDistToRegion = distToRegionCache_;
                distToRegionCache_ = soBounds_.computeDistToRegion(key_);
                if (nearest.shouldTraverse())
                    nearest(iters[1-childNo], iters[2-childNo]); // 0 -> (1, 2),  1 -> (0, 1)
                soBounds_(childNo, axis) = tmp;
                distToRegionCache_ = prevDistToRegion;
            }
        }
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_SO3SPACE_HPP
