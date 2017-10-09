#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_HPP

#include "spaces.hpp"
#include <array>
#include <iostream> // TODO: remove

namespace unc {
namespace robotics {
namespace kdtree {

namespace detail {
template <typename _T>
struct KDNode {
    _T value_;
    std::array<KDNode*, 2> children_;
    int debugAxisRemoveMe_;

    KDNode(const _T& v)
        : value_(v),
          children_{{nullptr, nullptr}}
    {
        debugAxisRemoveMe_ = 0;
    }

    ~KDNode() {
        delete children_[0];
        delete children_[1];
    }
};

template <typename _Space>
struct KDAddTraversal;

template <typename _Space>
struct KDNearestTraversal;

template <typename _Scalar, int _dimensions>
struct KDBoundedL2Traversal {
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    const State& key_;
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;

    KDBoundedL2Traversal(const Space& space, const State& key)
        : key_(key),
          bounds_(space.bounds())
    {
    }

    _Scalar keyDistance(const State& q) {
        return (key_ - q).norm();
    }

    _Scalar maxAxis(int *axis) {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
    }
};

template <typename _Derived>
int so3VolumeIndex(const Eigen::MatrixBase<_Derived>& q) {
    int index;
    q.array().abs().maxCoeff(&index);
    return index;
}

template <typename _Scalar>
int so3VolumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    return so3VolumeIndex(q.coeffs());
}

template <typename _Scalar>
struct KDSO3Traversal {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    // copy of the key, since we mutate it
    Eigen::Matrix<Scalar, 4, 1> key_;
    std::array<Eigen::Array<Scalar, 2, 3, Eigen::RowMajor>, 2> soBounds_;
    unsigned soDepth_;
    int keyVol_;

    KDSO3Traversal(const Space& space, const State& key)
        : key_(key.coeffs()),
          soDepth_(2),
          keyVol_(so3VolumeIndex(key))
    {
        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    _Scalar maxAxis(int *axis) {
        *axis = soDepth_ % 3;
        return M_PI/(1 << (soDepth_ / 3));
    }
};

template <typename _Scalar, int _dimensions>
struct KDAddTraversal<BoundedL2Space<_Scalar, _dimensions>>
    : KDBoundedL2Traversal<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    using KDBoundedL2Traversal<_Scalar, _dimensions>::key_;
    using KDBoundedL2Traversal<_Scalar, _dimensions>::bounds_;
    
    KDAddTraversal(const Space& space, const State& key)
        : KDBoundedL2Traversal<_Scalar, _dimensions>(space, key)
    {
    }

    template <typename _Adder, typename _T>
    unsigned addImpl(_Adder& adder, int axis, typename _Adder::Distance d, KDNode<_T>* p, KDNode<_T> *n, unsigned depth)
    {
        int axisCheck;
        assert(d == this->maxAxis(&axisCheck) && axis == axisCheck); // TODO: remove
        _Scalar split = (bounds_(axis, 0) + bounds_(axis,1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T>* c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return adder(c, n, depth+1);
        } else {
            p->debugAxisRemoveMe_ = axis + 10;
            p->children_[childNo] = n;
            return depth;
        }
    }
};

template <typename Scalar, typename _Derived>
static bool inSoBounds(
    int vol, int axis,
    const std::array<Eigen::Array<Scalar, 2, 3, Eigen::RowMajor>, 2>& soBounds,
    const Eigen::DenseBase<_Derived>& c)
{
    Scalar d0 = soBounds[0](0, axis)*c[vol] + soBounds[0](1, axis)*c[(vol + axis + 1)%4];
    Scalar d1 = soBounds[1](0, axis)*c[vol] + soBounds[1](1, axis)*c[(vol + axis + 1)%4];
    
    return d0 > 0 && d1 < 0;
}

template <typename _Derived>
Eigen::Matrix<typename _Derived::Scalar, 4, 1> rotateCoeffs(const Eigen::DenseBase<_Derived>& m, int shift) {
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
struct KDAddTraversal<SO3Space<_Scalar>>
    : KDSO3Traversal<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    using KDSO3Traversal<Scalar>::key_;
    using KDSO3Traversal<Scalar>::soBounds_;
    using KDSO3Traversal<Scalar>::soDepth_;
    using KDSO3Traversal<Scalar>::keyVol_;

    KDAddTraversal(const Space& space, const State& key)
        : KDSO3Traversal<Scalar>(space, key)
    {
        assert(space.isValid(key));

        key_ = rotateCoeffs(key_, keyVol_ + 1);
        
        if (key_[3] < 0)
            key_ = -key_;
    }

    template <typename _Adder, typename _T>
    unsigned addImpl(
        _Adder& adder,
        int axis,
        typename _Adder::Distance d,
        KDNode<_T>* p,
        KDNode<_T>* n,
        unsigned depth)
    {
        int axisCheck;
        assert(d == this->maxAxis(&axisCheck) && axis == axisCheck); // TODO: remove

        int childNo;
        KDNode<_T> *c;
        
        if (soDepth_ < 3) {
            if ((c = p->children_[childNo = keyVol_ & 1]) == nullptr) {
                p->children_[childNo] = n;
                p->debugAxisRemoveMe_ = 20;
                return depth;
            }
            p = c;
            if ((c = p->children_[childNo = keyVol_ >> 1]) == nullptr) {
                p->children_[childNo] = n;
                p->debugAxisRemoveMe_ = 21;
                return depth+1;
            }
            ++soDepth_;
            return adder(c, n, depth+2);
        } else {
            int soAxis = soDepth_ % 3;
            assert(axis == soAxis); // TODO: remove soAxis
            Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis))
                .matrix().normalized();

            // assert(inSoBounds(keyVol_, 0, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 1, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 2, soBounds_, key_));
                
            Scalar dot = mp[0]*key_[3] + mp[1]*key_[soAxis];
            if ((c = p->children_[childNo = (dot > 0)]) == nullptr) {
                p->children_[childNo] = n;
                p->debugAxisRemoveMe_ = 22 + axis;
                return depth;
            }
            
            soBounds_[1-childNo].col(soAxis) = mp;
            ++soDepth_;
            return adder(c, n, depth+2);
        }
    }
};

// TODO: switch to has-a instead of is-a for subspace.
template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDAddTraversal<RatioWeightedSpace<_Space, _num, _den>>
    : KDAddTraversal<_Space>
{
    typedef RatioWeightedSpace<_Space, _num, _den> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    // inherit constructor
    using KDAddTraversal<_Space>::KDAddTraversal;

    Distance keyDistance(const State& q) {
        return KDAddTraversal<_Space>::keyDistance(q) * _num / _den;
    }

    Distance maxAxis(int *axis) {
        return KDAddTraversal<_Space>::maxAxis(axis) * _num / _den;
    }

    template <typename _Adder, typename _T>
    unsigned addImpl(
        _Adder& adder,
        int axis,
        typename _Adder::Distance d,
        KDNode<_T>* p,
        KDNode<_T>* n,
        unsigned depth)
    {
        return KDAddTraversal<_Space>::addImpl(
            adder, axis, d * _den / _num, p, n, depth);
    }
};

template <int _index, typename ... _Spaces>
struct CompoundKeyDistance {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance accum(
        _TraversalsTuple& traversals,
        const typename Space::State& q,
        typename Space::Distance sum)
    {
        return CompoundKeyDistance<_index + 1, _Spaces...>::accum(
            traversals, q, sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q)));
    }
};

template <typename ... _Spaces>
struct CompoundKeyDistance<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance accum(
        _TraversalsTuple& traversals,
        const typename Space::State& q,
        typename Space::Distance sum)
    {
        return sum;
    }
};

// Compute the number of dimensions in the subspaces before the specified index.
template <int _index, typename ... _Spaces>
struct DimensionsBefore {
    static constexpr int value = DimensionsBefore<_index-1, _Spaces...>::value +
        std::tuple_element<_index-1, std::tuple<_Spaces...>>::type::dimensions;
};
// Base case, there are 0 dimensions before the subspace 0.
template <typename ... _Spaces>
struct DimensionsBefore<0, _Spaces...> { static constexpr int value = 0; };

template <int _index, typename ... _Spaces>
struct CompoundMaxAxis {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance maxAxis(
        _TraversalsTuple& traversals,
        typename Space::Distance bestDist,
        int *bestAxis)
    {
        int axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        assert((0 <= axis && axis < std::tuple_element<_index, std::tuple<_Spaces...>>::type::dimensions));
        if (d > bestDist) {
            *bestAxis = DimensionsBefore<_index, _Spaces...>::value + axis;
            bestDist = d;
        }
        return CompoundMaxAxis<_index+1, _Spaces...>::maxAxis(
            traversals, bestDist, bestAxis);
    }
};

template <typename ... _Spaces>
struct CompoundMaxAxis<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance maxAxis(
        _TraversalsTuple& traversals,
        typename Space::Distance bestDist,
        int *bestAxis)
    {
        return bestDist;
    }
};

template <int _index, typename ... _Spaces>
struct CompoundAddImpl {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;

    template <typename _Traversals, typename _Adder, typename _T>
    static inline unsigned addImpl(
        _Traversals& traversals,
        _Adder& adder,
        int axis,
        typename _Adder::Distance d, // TODO: be consistent with CompoundTraverse's Distance
        KDNode<_T>* p,
        KDNode<_T>* n,
        unsigned depth)
    {
        assert(axis >= dimBefore);
        if (axis < dimBefore + CurrentSpace::dimensions) {
            return std::get<_index>(traversals).addImpl(
                adder, axis - dimBefore, d, p, n, depth);
        } else {
            return CompoundAddImpl<_index+1, _Spaces...>::addImpl(
                traversals, adder, axis, d, p, n, depth);
        }
    }
};

template <typename ... _Spaces>
struct CompoundAddImpl<sizeof...(_Spaces)-1, _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;

    template <typename _Traversals, typename _Adder, typename _T>
    static inline unsigned addImpl(
        _Traversals& traversals,
        _Adder& adder,
        int axis,
        typename _Adder::Distance d, // TODO: be consistent with CompoundTraverse's Distance
        KDNode<_T>* p,
        KDNode<_T>* n,
        unsigned depth)
    {
        assert(axis >= dimBefore);
        assert(axis < dimBefore + CurrentSpace::dimensions);
        return std::get<_index>(traversals).addImpl(
            adder, axis - dimBefore, d, p, n, depth);
    }
};

template <typename ... _Spaces>
struct KDAddTraversal<CompoundSpace<_Spaces...>> {
    typedef std::tuple<_Spaces...> SpaceTuple;
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    std::tuple<KDAddTraversal<_Spaces>...> traversals_;

    template <std::size_t ... I>
    KDAddTraversal(const Space& space, const State& key, std::index_sequence<I...>)
        : traversals_(KDAddTraversal<typename std::tuple_element<I, SpaceTuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }
    
    KDAddTraversal(const Space& space, const State& key)
        : KDAddTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }
    
    Distance maxAxis(int *axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        assert((0 <= *axis && *axis < std::tuple_element<0, SpaceTuple>::type::dimensions));
        return CompoundMaxAxis<1, _Spaces...>::maxAxis(traversals_, d, axis);
    }

    template <typename _Adder, typename _T>
    unsigned addImpl(
        _Adder& adder,
        int axis,
        typename _Adder::Distance d,
        KDNode<_T>* p,
        KDNode<_T>* n,
        unsigned depth)
    {
        return CompoundAddImpl<0, _Spaces...>::addImpl(traversals_, adder, axis, d, p, n, depth);
    }
};

template <typename _Scalar, int _dimensions>
struct KDNearestTraversal<BoundedL2Space<_Scalar, _dimensions>>
    : KDBoundedL2Traversal<_Scalar, _dimensions>
{
    typedef _Scalar Scalar;
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    using KDBoundedL2Traversal<_Scalar, _dimensions>::key_;
    using KDBoundedL2Traversal<_Scalar, _dimensions>::bounds_;
    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;

    KDNearestTraversal(const Space& space, const State& key)
        : KDBoundedL2Traversal<_Scalar, _dimensions>(space, key)
    {
        regionDeltas_.setZero();
    }

    Distance distToRegion() {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, int axis, typename _Nearest::Distance d) {
        int axisCheck;
        assert(d == this->maxAxis(&axisCheck) && axis == axisCheck); // TODO: remove
        assert(n->debugAxisRemoveMe_ == 10 + axis || n->debugAxisRemoveMe_ == 0);
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
        _Scalar delta = (split - key_[axis]);
        int childNo = delta < 0;

        if (const KDNode<_T>* c = n->children_[childNo]) {
            std::swap(bounds_(axis, 1-childNo), split);
            nearest(c);
            std::swap(bounds_(axis, 1-childNo), split);
        }

        nearest.update(n);

        if (const KDNode<_T>* c = n->children_[1-childNo]) {
            Scalar oldDelta = regionDeltas_[axis];
            regionDeltas_[axis] = delta*delta;
            if (nearest.distToRegion() <= nearest.dist()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c);
                std::swap(bounds_(axis, childNo), split);
            }
            regionDeltas_[axis] = oldDelta;
        }
    }
};

template <typename _Scalar>
struct KDNearestTraversal<SO3Space<_Scalar>>
    : KDSO3Traversal<_Scalar>
{
    typedef _Scalar Scalar;
    typedef SO3Space<_Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    using KDSO3Traversal<_Scalar>::key_;
    using KDSO3Traversal<_Scalar>::soBounds_;
    using KDSO3Traversal<_Scalar>::soDepth_;
    using KDSO3Traversal<_Scalar>::keyVol_;

    State origKey_;

    KDNearestTraversal(const Space& space, const State& key)
        : KDSO3Traversal<_Scalar>(space, key),
          origKey_(key)
    {
        key_ = rotateCoeffs(key_, keyVol_ + 1);
        if (key_[3] < 0)
            key_ = -key_;
    }

    _Scalar keyDistance(const State& q) {
        _Scalar dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    template <typename _Derived>
    Scalar dotBounds(int b, int axis, const Eigen::DenseBase<_Derived>& q) {
        assert(b == 0 || b == 1);
        assert(0 <= axis && axis < 3);
            
        return soBounds_[b](0, axis)*q[3]
            +  soBounds_[b](1, axis)*q[axis];
    }

    Scalar distToRegion() {
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
                edgesToCheck |= 1 << ((a0+a1) << 2);
                continue; // not on face with this axis
            }
            int a2 = (a0+2)%3;
            if (dotBounds(1, a2, p0) > 0 || dotBounds(0, a2, p0) < 0) {
                edgesToCheck |= 1 << ((a0+a2) << 2);
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

        int cornerChecked = 0;

        Eigen::Matrix<Scalar, 2, 3> T;
        T.row(0) = soBounds_[0].row(0) / soBounds_[0].row(1);
        T.row(1) = soBounds_[1].row(0) / soBounds_[1].row(1);
        
        // check edges
        // ++, +-, --, -+ for 01, 12, 20
        Scalar dotMax = 0;
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            int a1 = (a0 + 1)%3;
            int a2 = (a0 + 2)%3;

            if ((edgesToCheck & (1 << ((a0+a1) << 2))) == 0)
                continue;
            
            for (int edge = 0 ; edge < 4 ; ++edge) {
                int b0 = edge & 1;
                int b1 = edge >> 1;

                Eigen::Matrix<Scalar, 4, 1> p1;
                Scalar t0 = T(b0, a0); // soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
                Scalar t1 = T(b1, a1); // soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
                Scalar r = q[3] - t0*q[a0] - t1*q[a1];
                Scalar s = t0*t0 + t1*t1 + 1;
                p1[3] = r;
                p1[a0] = -t0*r;
                p1[a1] = -t1*r;
                p1[a2] = q[a2] * s;
                if (p1[3] < 0)
                    p1 = -p1;
                
                int b2;
                if ((b2 = dotBounds(0, a2, p1) >= 0) && dotBounds(1, a2, p1) <= 0) {
                    dotMax = std::max(dotMax, std::abs(p1.normalized().dot(q))); // in bounds
                    continue;
                }
                
                int cornerCode = 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
                if (cornerChecked & cornerCode)
                    continue;
                cornerChecked |= cornerCode;
                // edge is not in bounds, use the distance to the corner
                Eigen::Matrix<Scalar, 4, 1> p2;
                Scalar aw = soBounds_[b0](0, a0);
                Scalar ax = soBounds_[b0](1, a0);
                Scalar bw = soBounds_[b1](0, a1);
                Scalar by = soBounds_[b1](1, a1);
                Scalar cw = soBounds_[b2](0, a2);
                Scalar cz = soBounds_[b2](1, a2);

                p2[a0]   =  aw*by*cz;
                p2[a1]   =  ax*bw*cz;
                p2[a2]   =  ax*by*cw;
                p2[3] = -ax*by*cz;
                p2.normalize();

                // // p2 should be on both bounds
                // assert(std::abs(dotBounds(b0, a0, p2)) < 1e-7);
                // assert(std::abs(dotBounds(b1, a1, p2)) < 1e-7);
                // assert(std::abs(dotBounds(b2, a2, p2)) < 1e-7);
            
                dotMax = std::max(dotMax, std::abs(q.dot(p2)));
            }
        }
        
        return std::acos(dotMax);
    }
    
    // Distance distToRegionOld() {
    //     const auto& q = key_;
    //     assert(q[vol_] >= 0);
    //     Scalar dotMax = 0; // dist = std::numeric_limits<Scalar>::infinity();
    //     Scalar qv = q[vol_];
    //     int edgeChecked = 0;
    //     int cornerChecked = 0;
    //     for (int a0 = 0 ; a0 < 3 ; ++a0) {
    //         int i0 = (vol_ + a0 + 1) % 4;
    //         Scalar qa0 = q[i0];
    //         Eigen::Matrix<Scalar, 2, 1> dot0(
    //             soBounds_[0](0, a0)*qv + soBounds_[0](1, a0)*qa0,
    //             soBounds_[1](0, a0)*qv + soBounds_[1](1, a0)*qa0);
    //         int b0 = dot0[0] >= 0;
    //         if (b0 && dot0[1] <= 0) {
    //             // face @ a0 is in bounds
    //             continue;
    //         }

    //         Eigen::Matrix<Scalar, 4, 1> p0 = q;
    //         p0[vol_] -= soBounds_[b0](0, a0) * dot0[b0];
    //         p0[i0  ] -= soBounds_[b0](1, a0) * dot0[b0];
    //         if (p0[vol_] < 0)
    //             p0 = -p0;

    //         // check that the projected point is on the bound
    //         assert(std::abs(dotBounds(b0, a0, p0)) < 1e-6);
    //         // check that the distance to the projected point is
    //         // the same as the distance to the bound.
    //         // assert(std::abs(std::acos(std::abs(p0.normalized().dot(q))) -
    //         //                 std::asin(std::abs(dot0[b0]))) < 1e-9);
                
    //         bool faceInBounds = true;
    //         for (int a1 = a0+1 ; (a1 = a1%3) != a0 ; ++a1) {
    //             int a2 = 3 - (a0 + a1);
    //             assert(a1 != a0 && a2 != a0 && a1 != a2 && a2 < 3);
    //             int i1 = (vol_ + a1 + 1) % 4;
    //             int i2 = (vol_ + a2 + 1) % 4;
    //             Eigen::Matrix<Scalar, 2, 1> dot1(
    //                 soBounds_[0](0, a1)*p0[vol_] + soBounds_[0](1, a1)*p0[i1],
    //                 soBounds_[1](0, a1)*p0[vol_] + soBounds_[1](1, a1)*p0[i1]);
    //             int b1 = dot1[0] >= 0;
    //             if (b1 && dot1[1] <= 0) {
    //                 // p0 @ a1 is in bounds
    //                 continue;
    //             }
    //             // std::cout << "face " << a0 << " out of bounds at " << a1 << "," << b1 << std::endl;
    //             faceInBounds = false;

    //             int edgeCode = 1 << ((a2 << 2) | (b1 << 1) | b0);
    //             if (edgeChecked & edgeCode)
    //                 continue;
    //             edgeChecked |= edgeCode;

    //             // p1 = project q onto the edge
    //             Eigen::Matrix<Scalar, 4, 1> p1;
    //             Scalar t0 = soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
    //             Scalar t1 = soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
    //             Scalar r = q[vol_] - t0*q[i0] - t1*q[i1];
    //             Scalar s = t0*t0 + t1*t1 + 1;
    //             p1[vol_] = r;
    //             p1[i0] = -t0*r;
    //             p1[i1] = -t1*r;
    //             p1[i2] = q[i2] * s;
    //             // p1.normalize();
    //             if (p1[vol_] < 0)
    //                 p1 = -p1;
                
    //             // check that p1 is in bounds of remaining axis
    //             Eigen::Matrix<Scalar, 2, 1> dot2(
    //                 soBounds_[0](0, a2)*p1[vol_] + soBounds_[0](1, a2)*p1[i2],
    //                 soBounds_[1](0, a2)*p1[vol_] + soBounds_[1](1, a2)*p1[i2]);
    //             int b2 = dot2[0] >= 0;
    //             if (b2 && dot2[1] <= 0) {
    //                 // edge is in bounds, use the distance to this edge
    //                 // Scalar edgeDist = std::acos(std::abs(p1.normalized().dot(q)));
    //                 // std::cout << "edge "
    //                 //           << a0 << "," << b0 << " - "
    //                 //           << a1 << "," << b1 << " = " << edgeDist << std::endl;
    //                 // dist = std::min(dist, edgeDist);
    //                 dotMax = std::max(dotMax, std::abs(p1.normalized().dot(q)));
    //             } else {
    //                 int cornerCode = 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
    //                 if (cornerChecked & cornerCode)
    //                     continue;
    //                 cornerChecked |= cornerCode;
                    
    //                 // edge is not in bounds, use the distance to the corner
    //                 Eigen::Matrix<Scalar, 4, 1> cp;
    //                 Scalar aw = soBounds_[b0](0, a0);
    //                 Scalar ax = soBounds_[b0](1, a0);
    //                 Scalar bw = soBounds_[b1](0, a1);
    //                 Scalar by = soBounds_[b1](1, a1);
    //                 Scalar cw = soBounds_[b2](0, a2);
    //                 Scalar cz = soBounds_[b2](1, a2);
                    
    //                 cp[i0]   = aw*by*cz;
    //                 cp[i1]   = ax*bw*cz;
    //                 cp[i2]   = ax*by*cw;
    //                 cp[vol_] = -ax*by*cz;
    //                 cp.normalize();
                    
    //                 // Scalar cornerDist = std::acos(std::abs(q.dot(cp)));
    //                 // int corner[3];
    //                 // corner[a0] = b0;
    //                 // corner[a1] = b1;
    //                 // corner[a2] = b2;
    //                 // std::cout << "corner "
    //                 //           << corner[0]
    //                 //           << corner[1]
    //                 //           << corner[2]
    //                 //           << " = " << cornerDist << std::endl;
    //                 // dist = std::min(dist, cornerDist);
    //                 dotMax = std::max(dotMax, std::abs(q.dot(cp)));
    //             }
    //         }
            
    //         if (faceInBounds) {
    //             Scalar faceDist = std::asin(std::abs(dot0[b0]));
    //             // std::cout << "face " << a0 << " = " << faceDist << std::endl;
    //             // dist = std::min(dist, faceDist);
    //             return faceDist;
    //         }
    //     }

    //     return dotMax == 0 ? 0 : std::acos(dotMax);
    // }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, int axis, typename _Nearest::Distance d) {
        int axisCheck;
        assert(d == this->maxAxis(&axisCheck) && axis == axisCheck); // TODO: remove
        // std::cout << n->value_.name_ << " " << soDepth_ << std::endl;
        if (soDepth_ < 3) {
            ++soDepth_;
            if (const KDNode<_T> *c = n->children_[keyVol_ & 1]) {
                assert(n->debugAxisRemoveMe_ == 20);
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const KDNode<_T> *g = c->children_[keyVol_ >> 1]) {
                    assert(c->debugAxisRemoveMe_ == 21);
                    assert(std::abs(origKey_.coeffs()[keyVol_]) == key_[3]);
                    nearest(g);
                }
                // TODO: can we gain so efficiency by exploring the
                // nearest of the remaining 3 volumes first?
                nearest.update(c);
                if (const KDNode<_T> *g = c->children_[1 - (keyVol_ >> 1)]) {
                    assert(c->debugAxisRemoveMe_ == 21);
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 2) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    assert(std::abs(origKey_.coeffs()[keyVol_ ^ 2]) == key_[3]);
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g);
                }
            }
            nearest.update(n);
            if (const KDNode<_T> *c = n->children_[1 - (keyVol_ & 1)]) {
                assert(n->debugAxisRemoveMe_ == 20);
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const KDNode<_T> *g = c->children_[keyVol_ >> 1]) {
                    assert(c->debugAxisRemoveMe_ == 21);
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 1) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    assert(std::abs(origKey_.coeffs()[keyVol_ ^ 1]) == key_[3]);
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g);
                }
                nearest.update(c);
                if (const KDNode<_T> *g = c->children_[1 - (keyVol_ >> 1)]) {
                    assert(c->debugAxisRemoveMe_ == 21);
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 3) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    assert(std::abs(origKey_.coeffs()[keyVol_ ^ 3]) == key_[3]);
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g);
                }
            }
            // setting vol_ to keyVol_ is only needed when part of a compound space
            // if (key_[vol_ = keyVol_] < 0)
            //     key_ = -key_;
            key_ = rotateCoeffs(origKey_.coeffs(), keyVol_ + 1);
            if (key_[3] < 0)
                key_ = -key_;
            --soDepth_;
            assert(distToRegion() == 0);
            assert(soDepth_ == 2);
        } else {
            int soAxis = soDepth_ % 3;
            assert(axis == soAxis); // TODO: remove soAxis
            assert(n->debugAxisRemoveMe_ == 0 || n->debugAxisRemoveMe_ == 22 + axis);
            Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis))
                .matrix().normalized();
            Scalar dot = mp[0]*key_[3]
                +        mp[1]*key_[soAxis];
            ++soDepth_;
            int childNo = (dot > 0);
            if (const KDNode<_T> *c = n->children_[childNo]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(soAxis);
                soBounds_[1-childNo].col(soAxis) = mp;
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
                soBounds_[1-childNo].col(soAxis) = tmp;
            }
            nearest.update(n);
            if (const KDNode<_T> *c = n->children_[1-childNo]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(soAxis);
                soBounds_[childNo].col(soAxis) = mp;
                if (nearest.distToRegion() <= nearest.dist())
                    nearest(c);
                soBounds_[childNo].col(soAxis) = tmp;
            }
            --soDepth_;
        }
    }
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDNearestTraversal<RatioWeightedSpace<_Space, _num, _den>>
    : KDNearestTraversal<_Space>
{
    typedef RatioWeightedSpace<_Space, _num, _den> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    // inherit constructor
    using KDNearestTraversal<_Space>::KDNearestTraversal;

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with KDAddTraversal.  Would be nice to merge them.
    Distance keyDistance(const State& q) {
        return KDNearestTraversal<_Space>::keyDistance(q) * _num / _den;
    }

    Distance maxAxis(int *axis) {
        return KDNearestTraversal<_Space>::maxAxis(axis) * _num / _den;
    }
    
    Distance distToRegion() {
        return KDNearestTraversal<_Space>::distToRegion() * _num / _den;
    }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, int axis, typename _Nearest::Distance d) {
        KDNearestTraversal<_Space>::traverse(nearest, n, axis, d * _den / _num);
    }
};

template <int _index, typename ... _Spaces>
struct CompoundDistToRegion {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    template <typename _Traversals>
    static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
        return CompoundDistToRegion<_index + 1, _Spaces...>::distToRegion(
            traversals, sum + std::get<_index>(traversals).distToRegion());
    }
};
template <typename ... _Spaces>
struct CompoundDistToRegion<sizeof...(_Spaces), _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    template <typename _Traversals>
    static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
        return sum;
    }
};
template <int _index, typename ... _Spaces>
struct CompoundTraverse {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;
    
    template <typename _Traversals, typename _Nearest, typename _T>
    static inline void traverse(
        _Traversals& traversals,
        _Nearest& nearest,
        const KDNode<_T>* n,
        int axis,
        Distance d)
    {
        assert(axis >= dimBefore);
        if (axis < dimBefore + CurrentSpace::dimensions) {
            std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore, d);
        } else {
            CompoundTraverse<_index+1, _Spaces...>::traverse(
                traversals, nearest, n, axis, d);
        }
    }
};
template <typename ... _Spaces>
struct CompoundTraverse<sizeof...(_Spaces)-1, _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;
    
    template <typename _Traversals, typename _Nearest, typename _T>
    static inline void traverse(
        _Traversals& traversals,
        _Nearest& nearest,
        const KDNode<_T>* n,
        int axis,
        Distance d)
    {
        assert(axis >= dimBefore);
        assert(axis - dimBefore < CurrentSpace::dimensions);
        std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore, d);
    }
};

template <typename ... _Spaces>
struct KDNearestTraversal<CompoundSpace<_Spaces...>> {
    typedef std::tuple<_Spaces...> SpaceTuple;
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    std::tuple<KDNearestTraversal<_Spaces>...> traversals_;

    template <std::size_t ... I>
    KDNearestTraversal(const Space& space, const State& key, std::index_sequence<I...>)
        : traversals_(KDNearestTraversal<typename std::tuple_element<I, SpaceTuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }
    
    KDNearestTraversal(const Space& space, const State& key)
        : KDNearestTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }
    
    Distance keyDistance(const State& q) {
        return CompoundKeyDistance<1, _Spaces...>::accum(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    Distance maxAxis(int *axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        assert((0 <= *axis && *axis < std::tuple_element<0, SpaceTuple>::type::dimensions));
        return CompoundMaxAxis<1, _Spaces...>::maxAxis(traversals_, d, axis);
    }

    Distance distToRegion() {
        // if (true) return std::get<0>(traversals_).distToRegion();
        // return std::get<0>(traversals_).distToRegion()
        //     + std::get<1>(traversals_).distToRegion();
        return CompoundDistToRegion<1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, int axis, Distance d) {
        CompoundTraverse<0, _Spaces...>::traverse(traversals_, nearest, n, axis, d);
    }
};

template <typename _Space>
struct KDAdder {
    typedef typename _Space::Distance Distance;
    
    KDAddTraversal<_Space> traversal_;
    
    KDAdder(const _Space& space, const typename _Space::State& key)
        : traversal_(space, key)
    {
    }

    template <typename _T>
    unsigned operator() (KDNode<_T>* p, KDNode<_T>*n, unsigned depth) {
        int axis;
        Distance d = traversal_.maxAxis(&axis);
        return traversal_.addImpl(*this, axis, d, p, n, depth);
    }
};

template <typename _Derived, typename _Space, typename _T, typename _TtoKey>
struct KDNearestBase {
    typedef typename _Space::Distance Distance;

    KDNearestTraversal<_Space> traversal_;
    _TtoKey tToKey_;
    Distance dist_;
    _Space space_; // TODO: remove (not needed, as KDNearestTraversal specialization has the info)
    typename _Space::State key_; // TODO: remove

    KDNearestBase(
        const _Space& space,
        const typename _Space::State& key, _TtoKey tToKey,
        Distance dist = std::numeric_limits<Distance>::infinity())
        : traversal_(space, key),
          tToKey_(tToKey),
          dist_(dist),
          space_(space), // TODO: remvoe
          key_(key) // TODO: remove
    {
    }

    Distance dist() {
        return dist_;
    }

    Distance distToRegion() {
        Distance d = traversal_.distToRegion();
        assert(d == d); // not NaN
        return d;
    }

    void operator() (const KDNode<_T>* n) {
        int axis;
        Distance d = traversal_.maxAxis(&axis);
        // std::cout << axis << ": " << d << std::endl;
        assert(d == d);
        traversal_.traverse(static_cast<_Derived&>(*this), n, axis, d);
    }

    void update(const KDNode<_T>* n) {
        Distance d = traversal_.keyDistance(this->tToKey_(n->value_));
        assert(d == space_.distance(this->tToKey_(n->value_), key_));
        // std::cout << "update " << n->value_.name_ << " " << d << std::endl;
        if (d < dist_) {
            static_cast<_Derived*>(this)->updateImpl(d, n);
        }
    }
};

template <typename _Space, typename _T, typename _TtoKey>
struct KDNearest1 : KDNearestBase<KDNearest1<_Space,_T,_TtoKey>, _Space, _T, _TtoKey> {
    typedef typename _Space::Distance Distance;

    const KDNode<_T>* nearest_;

    KDNearest1(const _Space& space, const typename _Space::State& key, _TtoKey tToKey)
        : KDNearestBase<KDNearest1, _Space, _T, _TtoKey>(space, key, tToKey),
          nearest_(nullptr)
    {
    }
    
    void updateImpl(Distance d, const KDNode<_T> *n) {
        this->dist_ = d;
        nearest_ = n;
    }
};

struct DistValuePairCompare {
    template <typename _Dist, typename _Value>
    inline bool operator() (const std::pair<_Dist, _Value>& a, const std::pair<_Dist, _Value>& b) const {
        return a.first < b.first;
    }
};

template <typename _Space, typename _T, typename _TtoKey>
struct KDNearestK : KDNearestBase<KDNearestK<_Space,_T,_TtoKey>, _Space, _T, _TtoKey> {
    typedef typename _Space::Distance Distance;

    std::size_t k_;
    std::vector<std::pair<Distance, _T>>& nearest_;

    KDNearestK(
        std::vector<std::pair<Distance, _T>>& nearest,
        std::size_t k,
        Distance r,
        const _Space& space, const typename _Space::State& key, _TtoKey tToKey)
        : KDNearestBase<KDNearestK, _Space, _T, _TtoKey>(space, key, tToKey, r),
          k_(k),
          nearest_(nearest)
    {
        // update() will not work unless the following to initial criteria hold:
        assert(k > 0);
        assert(nearest.size() <= k);
    }

    void updateImpl(Distance d, const KDNode<_T>* n) {
        assert(nearest_.size() <= k_);
        assert(std::is_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare())); // TODO: REMOVE
        if (nearest_.size() == k_) {
            std::pop_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
            nearest_.pop_back();
        }
        nearest_.emplace_back(d, n->value_);
        std::push_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
        if (nearest_.size() == k_)
            this->dist_ = nearest_[0].first;
        assert(nearest_.size() <= k_);
        assert(std::is_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare())); // TODO: REMOVE
    }
};


// template <typename _Space>
// struct KDAdderBase {
//     typedef KDAdder<_Space> Derived;
//     typedef typename _Space::Distance Distance;

//     template <typename _T>
//     unsigned operator() (KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
//         int axis;
//         Distance d = static_cast<Derived*>(this)->maxAxis(&axis);
//         // tail recursion
//         return static_cast<Derived*>(this)->addImpl(*this, axis, d, p, n, depth);
//     }
// };

// template <typename _Space>
// struct KDAdder;

// template <typename _Space>
// struct KDTraversal;




// template <typename _Scalar, int _dimensions>
// struct KDAdder<BoundedL2Space<_Scalar, _dimensions>> : KDAdderBase<BoundedL2Space<_Scalar, _dimensions>> {
//     const State& key_;
//     Eigen::Array<_Scalar, _dimensions, 2> bounds_;

//     KDAdder(const State& key, const Space& space)
//         : key_(key),
//           bounds_(space.bounds())
//     {
//     }

//     _Scalar maxAxis(int *axis) {
//         return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
//     }

//     template <typename _RootAdder, typename _T>
//     unsigned addImpl(
//         _RootAdder& adder,
//         int axis,
//         typename _RootAdder::Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
//         int childNo = (split - key_[axis]) < 0;
//         if (KDNode<_T>* c = p->children_[childNo]) {
//             bounds_(axis, 1-childNo) = split;
//             return adder(c, n, depth+1); // tail recur
//         } else {
//             p->children_[childNo] = n;
//             return depth;
//         }
//     }
// };

// template <typename _Scalar, int _dimensions>
// struct KDTraversal<BoundedL2Space<_Scalar, _dimensions>> {
//     typedef BoundedL2Space<_Scalar, _dimensions> Space;
//     typedef typename Space::State State;

//     const State& key_;
//     Eigen::Array<_Scalar, _dimensions, 2> bounds_;
//     Eigen::Array<_Scalar, _dimensions, 1> deltas_;
    
//     KDTraversal(const State& key, const Space& space) {
//         deltas_.setZero();
//     }

//     _Scalar distToRegion() {
//         return std::sqrt(deltas_.sum());
//     }

//     _Scalar maxAxis(int *axis) {
//         return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
//     }

//     template <typename _Nearest, typename _T>
//     void traverse(_Nearest& nearest, const KDNode<_T> *n, int axis, _Scalar d) {
//         _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
//         _Scalar delta = (split - key_[axis]);
//         int childNo = delta < 0;

//         if (const KDNode<_T>* c = n->children_[childNo]) {
//             std::swap(bounds_(axis, 1-childNo), split);
//             nearest.traverse(c);
//             std::swap(bounds_(axis, 1-childNo), split);
//         }

//         update(n);

//         if (const KDNode<_T>* c = n->children_[1-childNo]) {
//             Scalar oldDelta = deltas_[axis];
//             deltas_[axis] = delta*delta;
//             if (nearest.distToRegion() <= nearest.dist()) {
//                 std::swap(bounds_(axis, childNo), split);
//                 nearest.traverse(c);
//                 std::swap(bounds_(axis, childNo), split);
//             }
//             deltas_[axis] = oldDelta;
//         }
//     }
// };

// template <typename _Scalar>
// struct KDAdder<SO3Space<_Scalar>> : KDAdderBase<SO3Space<_Scalar>> {
//     typedef _Scalar Scalar;
//     typedef SO3Space<Scalar> Space;
//     typedef typename Space::State State;
//     typedef typename Space::Distance Distance;

//     int vol_;
//     int soDepth_;
//     Eigen::Matrix<Scalar, 4, 1> soKey_;

//     std::array<Eigen::Array<Scalar, 2, 3>, 2> bounds_;
    
//     KDAdder(const State& key, const Space& space)
//         : vol_(volumeIndex(key)),
//           soDepth_(2),
//           soKey_(key.coeffs()[vol_] < 0 ? -key.coeffs() : key.coeffs())
//     {
//         static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
//         bounds_[0] = rt;
//         bounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
//     }

//     Distance maxAxis(int* axis) {
//         return M_PI / (1 << (depth_ / 3));
//     }
    
//     template <typename _RootAdder, typename _T>
//     unsigned addImpl(
//         _RootAdder& adder,
//         int axis,
//         typename _RootAdder::Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         KDNode<_T>* t;
//         int childNo;
//         if (soDepth_ < 3) {
//             if ((t = p->children_[childNo = vol_ & 1]) == nullptr) {
//                 p->children_[childNo] = n;
//                 return depth;
//             }
//             p = t;
//             ++depth;
//             if ((t = p->children_[childNo = vol_ >> 1]) == nullptr) {
//                 p->children_[childNo] = n;
//                 return depth;
//             }
//         } else {
//             axis = depth_ % 3;
//             Eigen::Matrix<Scalar, 2, 1> mp = (bounds_[0].col(axis) + bounds_[1].col(axis))
//                 .matrix().normalized();
//             Scalar dot = mp[0]*soKey_[vol_] + mp[1]*soKey_[(vol + axis + 1)%4];
//             if ((t = p->children_[childNo = (dot > 0)]) == nullptr) {
//                 p->children_[childNo] = n;
//                 return depth;
//             }
//             bounds_[1-childNo].col(axis) = mp;
//         }
//         ++soDepth_;
//         return adder(t, n, depth+1);
//     }
// };

// template <typename _Scalar>
// struct KDTraversal<SO3Space<_Scalar>> {
//     typedef SO3Space<_Scalar> Space;
//     typedef typename Space::State State;

//     int vol_;
//     int soDepth_;
//     Eigen::Matrix<_Scalar, 4, 1> soKey_;
//     std::array<Eigen::Array<_Scalar, 2, 3>, 2> soBounds_;

//     KDTraversal(const State& key, const Space& space)
//         : vol_(volumeIndex(key.coeffs())),
//           soDepth_(2)
//     {
//         static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
//         soBounds_[0] = rt;
//         soBounds_[1] = colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
//     }

//     _Scalar distToRegion() {
//         return 0; // TODO
//     }

//     Distance maxAxis(int* axis) {
//         return M_PI / (1 << (soDepth_ / 3));
//     }

//     template <typename _Traversal, typename _Nearest>
//     void traverse(_Nearest& nearest, const KDNode<_T> *n, int axis, _Scalar d) {
//         if (soDepth_ < 3) {
//             ++soDepth_;
//             if (const Node *c = n->children_[keyVol_ & 1]) {
//                 if (const Node *g = n->children_[keyVol_ >> 1]) {
//                     if (key_[vol_ = keyVol_] < 0)
//                         key_ = -key_;
//                     nearest.traverse(g);
//                 }
//                 nearest.update(c);
//                 if (const Node *g = n->children_[1 - (keyVol_ >> 1)]) {
//                     if (key_[vol_ = keyVol_ ^ 2] < 0)
//                         key_ = -key_;
//                     // TODO: region check
//                     nearest.traverse(g);
//                 }
//             }
//             nearest.update(n);
//             if (const Node *c = n->children_[1 - (keyVol_ & 1)]) {
//                 if (const Node *g = n->children_[keyVol_ >> 1]) {
//                     if (key_[vol_ = keyVol_ ^ 1] < 0)
//                         key_ = -key_;
//                     // TODO: region check
//                     nearest.traverse(g);
//                 }
//                 nearest.update(c);
//                 if (const Node *g = n->children_[1 - (keyVol_ >> 1)]) {
//                     if (key_[vol_ = keyVol_ ^ 3] < 0)
//                         key_ = -key_;
//                     // TODO: region check
//                     nearest.traverse(g);
//                 }
//             }
//             --soDepth_;
//         } else {
//             int soAxis = soDepth_ % 3;
//             Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis))
//                 .matrix().normalized();
//             Scalar dot = mp[0]*std::get<0>(key_).coeffs()[vol_]
//                 + mp[1]*std::get<0>(key_).coeffs()[(vol_ + soAxis + 1)%4];
//             ++soDepth_;
//             int childNo = (dot > 0);
//             if (const Node *c = n->children_[childNo]) {
//                 Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(soAxis);
//                 soBounds_[1-childNo].col(soAxis) = mp;
// #ifdef KD_PEDANTIC
//                 Scalar soBoundsDistNow = soBoundsDist();
//                 if (soBoundsDistNow + rvBoundsDistCache_ <= dist_) {
//                     std::swap(soBoundsDistNow, soBoundsDistCache_);
// #endif
//                     nearest.traverse(c);
// #ifdef KD_PEDANTIC
//                     soBoundsDistCache_ = soBoundsDistNow;
//                 }
// #endif
//                 soBounds_[1-childNo].col(soAxis) = tmp;
//             }
//             nearest.update(n);
//             if (const Node *c = n->children_[1-childNo]) {
//                 Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(soAxis);
//                 soBounds_[childNo].col(soAxis) = mp;
//                 //Scalar faceDist = computeSOFaceDist(childNo, soAxis);
//                 Scalar soBoundsDistNow = soBoundsDist();
//                 if (soBoundsDistNow + rvBoundsDistCache_ <= dist_) {
//                     std::swap(soBoundsDistNow, soBoundsDistCache_);
//                     nearest.traverse(c);
//                     soBoundsDistCache_ = soBoundsDistNow;
//                 }
//                 soBounds_[childNo].col(soAxis) = tmp;
//             }
//             --soDepth_;
//         }
//     }

// };


// template <typename _Space, std::intmax_t _num, std::intmax_t _den>
// struct KDAdder<RatioWeightedSpace<_Space, _num, _den>> : KDAdder<_Space> {
//     // inherit constructor
//     using KDAdder<_Space>::KDAdder;

//     typename _Space::Distance maxAxis(int *axis) {
//         return KDAdder<_Space>::maxAxis(axis) * _num / _den;
//     }

//     template <typename _RootAdder, typename _T>
//     unsigned addImpl(
//         _RootAdder& adder,
//         int axis,
//         typename _RootAdder::Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         return KDAdder<_Space>::addImpl(adder, axis, distance * _den / _num, p, n, depth);
//     }
// };

// template <typename _Space, std::intmax_t _num, std::intmax_t _den>
// struct KDTraversal<RatioWeightedSpace<_Space, _num, _den>> : KDTraversal<_Space> {
//     // inherit constructor
//     using KDTraversal<_Space>::KDTraversal;

//     typedef typename _Space::Distance Distance;

//     Distance maxAxis(int *axis) {
//         return KDTraversal<_Space>::maxAxis(axis) * _num / _den;
//     }

//     template <typename _Nearest, typename _T>
//     void traverse(_Nearest& nearest, const KDNode<_T> *n, int axis, Distance d) {
//         KDTraversal<_Space>::traverse(nearest, n, axis, d * _den / _num);
//     }
// };

// // Compute the number of dimensions in the subspaces before the specified index.
// template <int _index, typename ... _Spaces>
// struct DimensionsBefore {
//     static constexpr int value = DimensionsBefore<_index-1, _Spaces...>::value +
//         std::tuple_element<_index-1, std::tuple<_Spaces...>>::type::dimensions;
// };
// // Base case, there are 0 dimensions before the subspace 0.
// template <typename ... _Spaces>
// struct DimensionsBefore<0, _Spaces...> { static constexpr int value = 0; };

// // Compute the maximum axis of a compute space.
// template <int _index, typename ... _Spaces>
// struct CompoundMaxAxis {
//     typedef typename CompoundSpace<_Spaces...>::Distance Distance;

//     static Distance compute(
//         std::tuple<KDAdder<_Spaces>...>& adders,
//         Distance dist,
//         int *axis)
//     {
//         int a;
//         Distance d = std::get<_index>(adders).maxAxis(&a);
//         if (d > dist) {
//             *axis = a + DimensionsBefore<_index, _Spaces...>::value;
//             dist = d;
//         }
//         // tail recursion
//         return CompoundMaxAxis<_index+1, _Spaces...>::compute(
//             adders, dist, axis);
//     }
// };

// template <typename ... _Spaces>
// struct CompoundMaxAxis<sizeof...(_Spaces), _Spaces...> {
//     static inline Distance compute(
//         std::tuple<KDAdder<_Spaces>...>& adders,
//         Distance dist,
//         int *axis)
//     {
//         return dist;
//     }
// };

// template <int _index, typename ... _Spaces>
// struct CompoundAddImpl {
//     template <typename _RootAdder, typename _T>
//     static unsigned apply(
//         std::tuple<KDAdder<_Spaces>...>& adders,
//         _RootAdder& adder,
//         int axis,
//         typename _RootAdder::Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;
        
//         if (axis < DimensionsBefore<_index, _Spaces...>::value + Subspace::dimensions) {
//             return std::get<_index>(adders).addImpl(
//                 adder,
//                 axis - DimensionsBefore<_index, _Spaces...>::value,
//                 distance,
//                 p, n, depth);
//         } else {
//             return CompoundAddImpl<_index+1, _Spaces...>::apply(
//                 adders, adder, axis, distance, p, n, depth);
//         }
//     }
// };
// template <typename ... _Spaces>
// struct CompoundAddImpl<sizeof...(_Spaces)-1, _Spaces...> {
//     template <typename _RootAdder, typename _T>
//     static unsigned apply(
//         std::tuple<KDAdder<_Spaces>...>& adders,
//         _RootAdder& adder,
//         int axis,
//         typename _RootAdder::Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         return std::get<sizeof...(_Spaces)-1>(adders).addImpl(
//             adder,
//             axis - DimensionsBefore<_index, _Spaces...>::value,
//             distance,
//             p, n, depth);
//     }
// };

// template <typename ... _Spaces>
// struct KDAdder<CompoundSpace<_Spaces...>> : KDAdderBase<CompoundSpace<_Spaces...>> {
//     typedef CompoundSpace<_Spaces...> Space;
//     typedef typename Space::State State;
//     typedef typename Space::Distance Distance;
//     typedef std::make_index_sequence<sizeof...(_Spaces)> IndexSeq;
    
//     std::tuple<KDAdder<_Spaces>...> adders_;

//     template <std::size_t ... I>
//     KDAdder(const State& key, const Space& space, std::index_sequence<I...>)
//         : adders_(KDAdder<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>(
//                       std::get<I>(key),
//                       std::get<I>(space))...)
//     {
//     }
    
//     KDAdder(const State& key, const Space& space)
//         : KDAdder(key, space, IndexSeq{})
//     {
//     }

//     Distance maxAxis(int *axis) {
//         Scalar dist = std::get<0>(adders_).maxAxis(axis);
//         return CompoundMaxAxis<1, _Spaces...>::compute(adders_, dist, axis);
//     }

//     template <typename _RootAdder, typename _T>
//     unsigned addImpl(
//         _RootAdder& adder,
//         int axis,
//         Distance distance,
//         KDNode<_T>* p,
//         KDNode<_T>* n,
//         unsigned depth)
//     {
//         return CompoundAddImpl<0, _Spaces...>::apply(adders_, axis, distance, p, n, depth);
//     }
// };

// template <typename ... _Spaces>
// struct KDTraversal<CompoundSpace<_Spaces...>> {
//     typedef std::make_index_sequence<sizeof...(_Spaces)> IndexSeq;

//     std::tuple<KDTraversal<_Spaces>...> traversals_;
    
//     template <std::size_t ... I>
//     KDTraversal(const State& key, const Space& space, std::index_sequence<I...>)
//         : traversals_(KDTraversal<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>(
//                           std::get<I>(key),
//                           std::get<I>(space))...)
//     {
//     }
    
//     KDTraversal(const State& key, const Space& space)
//         : KDTraversal(key, space, IndexSeq{})
//     {
//     }

//     Distance maxAxis(int *axis) {
//         Scalar dist = std::get<0>(adders_).maxAxis(axis);
//         return CompoundMaxAxis<1, _Spaces...>::compute(traversals_, dist, axis);
//     }

//     // TODO: rest
// };


// template <typename _Space, typename _T>
// struct KDNearest1 {
//     typedef typename _Space::Distance Distance;

//     const _Space& space_;
//     KDTraversal<_Space> traverse_;

//     Distance dist_;
//     const KDNode<_T> *nearest_;

//     KDNearest1(const _Space& space)
//         : space_(space),
//           traverse_(space),
//           dist_(std::numeric_limits<Distance>::infinity()),
//           nearest_(nullptr)
//     {
//     }

//     void traverse(const KDNode<_T>* n) {
//         int axis;
//         Distance d = traversal_.maxAxis(&axis);
//         traversal_(*this, n, axis, d);
//     }

//     inline Distance distToRegion() const {
//         return traverse_.distToRegion();
//     }

//     inline Distance dist() const {
//         return dist_;
//     }

//     void update(const KDNode<_T>* n) {
//         Distance d = space_.distance(tToKey_(n->value_), key_);
//         if (d < dist_) {
//             dist_ = d;
//             nearest_ = n;
//         }
//     }
// };

// template <typename _Space, typename _T>
// struct KDNearestK {
// };
    
} // namespace unc::robotics::kdtree::detail

template <typename _T, typename _Space, typename _TtoKey>
class KDTree {
    typedef detail::KDNode<_T> Node;
    typedef _Space Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    // TODO: _Space may be empty, exploit empty base-class
    // optimization
    Space space_;

    // TODO: tToKey_ may also be empty
    _TtoKey tToKey_;

    Node *root_;
    std::size_t size_;
    unsigned depth_;

public:
    KDTree(_TtoKey tToKey, const Space& space = _Space())
        : space_(space),
          tToKey_(tToKey),
          root_(nullptr),
          size_(0),
          depth_(0)
    {
    }

    ~KDTree() {
        delete root_;
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    unsigned depth() const {
        return depth_;
    }

    void add(const _T& value) {
        Node* n = new Node(value);
        unsigned depth = 1;

        if (root_ == nullptr) {
            root_ = n;
        } else {
            const Key& key = tToKey_(value);
            detail::KDAdder<Space> adder(space_, key);
            depth = adder(root_, n, 2);
        }
        
        ++size_;
        if (depth_ < depth)
            depth_ = depth;
    }

    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        detail::KDNearest1<_Space, _T, _TtoKey> nearest(space_, key, tToKey_);
        nearest(root_);
        if (distOut)
            *distOut = nearest.dist_;
        return nearest.nearest_ ? &nearest.nearest_->value_ : nullptr;
    }

    void nearest(
        std::vector<std::pair<Distance, _T>>& result,
        const Key& key,
        std::size_t k,
        Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
    {
        result.clear();
        
        if (k == 0)
            return;

        detail::KDNearestK<_Space, _T, _TtoKey> nearestK(result, k, maxRadius, space_, key, tToKey_);
        nearestK(root_);
        std::sort_heap(result.begin(), result.end(), detail::DistValuePairCompare());
    }
};

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_KDTREE_HPP
