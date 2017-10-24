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
unsigned so3VolumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    return so3VolumeIndex(q.coeffs());
}

template <typename _Derived>
Eigen::Matrix<typename _Derived::Scalar, 4, 1> rotateCoeffs(const Eigen::DenseBase<_Derived>& m, unsigned shift) {
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
struct MidpointSO3TraversalBase {
    typedef SO3Space<_Scalar> Space;
    typedef typename Space::State Key;

    Eigen::Matrix<_Scalar, 4, 1> key_;
    std::array<Eigen::Array<_Scalar, 2, 3, Eigen::RowMajor>, 2> soBounds_;
    unsigned soDepth_;
    unsigned keyVol_;

    MidpointSO3TraversalBase(const Space& space, const Key& key)
        : soDepth_(2),
          keyVol_(so3VolumeIndex(key))
    {
        key_ = rotateCoeffs(key.coeffs(), keyVol_ + 1);
        if (key_[3] < 0)
            key_ = -key_;

        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<_Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    inline constexpr unsigned dimensions() const {
        return 3;
    }

    inline _Scalar maxAxis(unsigned *axis) {
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
    inline _Scalar keyDistance(const Eigen::QuaternionBase<_Derived>& q) {
        _Scalar dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    inline Distance distToRegion() {
        return distToRegionCache_;
    }

    template <typename _Derived>
    inline Distance dotBounds(int b, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        // assert(b == 0 || b == 1);
        // assert(0 <= axis && axis < 3);
            
        return soBounds_[b](0, axis)*q[3]
            +  soBounds_[b](1, axis)*q[axis];
    }

    inline Distance computeDistToRegion() {
        const auto& q = key_;
        int edgesToCheck = 0;
        
        // check faces
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            Eigen::Matrix<Scalar, 2, 1> dot(dotBounds(0, a0, q), dotBounds(1, a0, q));
            int b0 = dot[0] >= 0;
            if (b0 && dot[1] <= 0)
                continue; // in bounds

            Eigen::Matrix<Scalar, 4, 1> p0 = q;
            p0[3]  -= soBounds_[b0](0, a0) * dot[b0];
            p0[a0] -= soBounds_[b0](1, a0) * dot[b0];

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
            return std::asin(dot[b0]);
        }

        // if the query point is within all bounds of all 3 axes, then it is within the region.
        if (edgesToCheck == 0)
            return 0;

        // int cornerChecked = 0;
        int cornersToCheck = 0;
        Eigen::Matrix<Scalar, 2, 3> T;
        T.row(0) = soBounds_[0].row(0) / soBounds_[0].row(1);
        T.row(1) = soBounds_[1].row(0) / soBounds_[1].row(1);
        
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
                Scalar t0 = T(b0, a0); // soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
                Scalar t1 = T(b1, a1); // soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
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
                // Scalar aw = soBounds_[b0](0, a0);
                // Scalar ax = soBounds_[b0](1, a0);
                // Scalar bw = soBounds_[b1](0, a1);
                // Scalar by = soBounds_[b1](1, a1);
                // Scalar cw = soBounds_[b2](0, a2);
                // Scalar cz = soBounds_[b2](1, a2);

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
            Scalar aw = soBounds_[b0](0, 0);
            Scalar ax = soBounds_[b0](1, 0);
            Scalar bw = soBounds_[b1](0, 1);
            Scalar by = soBounds_[b1](1, 1);
            Scalar cw = soBounds_[b2](0, 2);
            Scalar cz = soBounds_[b2](1, 2);

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

    template <typename _Nearest>
    void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
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
                Scalar oldDistToRegion = distToRegionCache_;
                distToRegionCache_ = computeDistToRegion();
                if (nearest.shouldTraverse())
                    nearest(c);
                distToRegionCache_ = oldDistToRegion;
                soBounds_[childNo].col(axis) = tmp;
            }
            --soDepth_;
        }
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_SO3SPACE_HPP
