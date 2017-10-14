#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_HPP

#include "spaces.hpp"
#include <array>
#include <vector>
#include <unordered_set>
#include <utility>
#include <iostream> // TODO: remove

namespace unc {
namespace robotics {
namespace kdtree {

namespace detail {

template <typename _T>
struct KDNode {
    _T value_;
    std::array<KDNode*, 2> children_{};

    KDNode(const _T& v) : value_(v) {}

    ~KDNode() {
        delete children_[0];
        delete children_[1];
    }
};

template <typename _T>
struct KDValueSplitNode {
    _T value_;
    std::array<KDValueSplitNode*, 2> children_{};

    KDValueSplitNode(const _T& v) : value_(v) {}

    ~KDValueSplitNode() {
        delete children_[0];
        delete children_[1];
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDTreeBase {
    // TODO: _Space may be empty, exploit empty base-class
    // optimization
    _Space space_;

    // TODO: tToKey_ may also be empty
    _TtoKey tToKey_;

    std::vector<unsigned> axisCache_;
    std::unordered_set<const KDNode<_T>*> removedSet_;

    std::size_t size_ = 0;

    inline KDTreeBase(_TtoKey tToKey, const _Space& space)
        : space_(space),
          tToKey_(tToKey)
    {
        axisCache_.reserve(32);
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

    inline constexpr unsigned dimensions() const {
        return _dimensions;
    }

    template <typename _Derived>
    inline _Scalar keyDistance(const Eigen::MatrixBase<_Derived>& q) {
        return (key_ - q).norm();
    }

    inline _Scalar maxAxis(unsigned *axis) {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
    }
};

// TODO: this shares a lot with the non-dynamic version.
template <typename _Scalar>
struct KDBoundedL2Traversal<_Scalar, Eigen::Dynamic> {
    typedef BoundedL2Space<_Scalar, Eigen::Dynamic> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    unsigned dimensions_;
    const State& key_;
    Eigen::Array<_Scalar, Eigen::Dynamic, 2> bounds_;

    KDBoundedL2Traversal(const Space& space, const State& key)
        : dimensions_(space.dimensions()),
          key_(key),
          bounds_(space.bounds())
    {
    }

    inline unsigned dimensions() const {
        return dimensions_;
    }

    template <typename _Derived>
    inline _Scalar keyDistance(const Eigen::MatrixBase<_Derived>& q) {
        return (key_ - q).norm();
    }

    inline _Scalar maxAxis(unsigned *axis) {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
    }
};


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
    unsigned keyVol_;

    KDSO3Traversal(const Space& space, const State& key)
        : key_(key.coeffs()),
          soDepth_(2),
          keyVol_(so3VolumeIndex(key))
    {
        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    inline constexpr unsigned dimensions() const {
        return 3;
    }
    
    inline _Scalar maxAxis(unsigned *axis) {
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
    KDNode<_T>** addImpl(_Adder& adder, unsigned axis, KDNode<_T>* p, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis,1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T>* c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return adder(c, depth+1);
        } else {
            return &p->children_[childNo];
        }
    }

    template <typename _Remover, typename _T>
    bool removeImpl(_Remover& remover, unsigned axis, KDNode<_T>* p, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis,1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T>* c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return remover(c, depth+1);
        } else {
            return false;
        }
    }
};

template <typename Scalar, typename _Derived>
static bool inSoBounds(
    unsigned vol, unsigned axis,
    const std::array<Eigen::Array<Scalar, 2, 3, Eigen::RowMajor>, 2>& soBounds,
    const Eigen::DenseBase<_Derived>& c)
{
    Scalar d0 = soBounds[0](0, axis)*c[vol] + soBounds[0](1, axis)*c[(vol + axis + 1)%4];
    Scalar d1 = soBounds[1](0, axis)*c[vol] + soBounds[1](1, axis)*c[(vol + axis + 1)%4];
    
    return d0 > 0 && d1 < 0;
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
        // assert(space.isValid(key));

        key_ = rotateCoeffs(key_, keyVol_ + 1);
        
        if (key_[3] < 0)
            key_ = -key_;
    }

    template <typename _Adder, typename _T>
    KDNode<_T>** addImpl(
        _Adder& adder,
        unsigned axis,
        KDNode<_T>* p,
        unsigned depth)
    {
        int childNo;
        KDNode<_T> *c;
        
        if (soDepth_ < 3) {
            if ((c = p->children_[childNo = keyVol_ & 1]) == nullptr) {
                return &p->children_[childNo];
            }
            p = c;
            if ((c = p->children_[childNo = keyVol_ >> 1]) == nullptr) {
                return &p->children_[childNo];
            }
            ++soDepth_;
            return adder(c, depth+1); // this should be depth+2, but that doesn't work with the axisCache.
        } else {
            Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(axis) + soBounds_[1].col(axis))
                .matrix().normalized();

            // assert(inSoBounds(keyVol_, 0, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 1, soBounds_, key_));
            // assert(inSoBounds(keyVol_, 2, soBounds_, key_));
                
            Scalar dot = mp[0]*key_[3] + mp[1]*key_[axis];
            if ((c = p->children_[childNo = (dot > 0)]) == nullptr) {
                return &p->children_[childNo];
            }
            
            soBounds_[1-childNo].col(axis) = mp;
            ++soDepth_;
            return adder(c, depth+1);
        }
    }

    template <typename _Remover, typename _T>
    bool removeImpl(_Remover& remover, unsigned axis, KDNode<_T>* p, unsigned depth) {
        KDNode<_T>* c;
        if (soDepth_ < 3) {
            if ((c = p->children_[keyVol_ & 1]) == nullptr)
                return false;

            if (remover.remove(c))
                return true;

            p = c;
            if ((c = p->children_[keyVol_ >> 1]) == nullptr)
                return false;

            ++soDepth_;
            return remover(c, depth+1); // this should be depth+2, but that doesn't work with the axisCache.
        } else {
            int childNo;
            Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(axis) + soBounds_[1].col(axis))
                .matrix().normalized();
                
            Scalar dot = mp[0]*key_[3] + mp[1]*key_[axis];
            if ((c = p->children_[childNo = (dot > 0)]) == nullptr)
                return false;
            
            soBounds_[1-childNo].col(axis) = mp;
            ++soDepth_;
            return remover(c, depth+1);            
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

    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return KDAddTraversal<_Space>::keyDistance(q) * _num / _den;
    }

    inline Distance maxAxis(unsigned *axis) {
        return KDAddTraversal<_Space>::maxAxis(axis) * _num / _den;
    }

    // template <typename _Adder, typename _T>
    // unsigned addImpl(
    //     _Adder& adder,
    //     unsigned axis,
    //     KDNode<_T>* p,
    //     KDNode<_T>* n,
    //     unsigned depth)
    // {
    //     return KDAddTraversal<_Space>::addImpl(
    //         adder, axis, p, n, depth);
    // }
};

template <typename _Space>
struct KDAddTraversal<WeightedSpace<_Space>>
    : KDAddTraversal<_Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;

    KDAddTraversal(const Space& space, const State& key)
        : KDAddTraversal<_Space>(space, key),
          weight_(space.weight())
    {
    }

    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return KDAddTraversal<_Space>::keyDistance(q) * weight_;
    }

    inline Distance maxAxis(unsigned *axis) {
        return KDAddTraversal<_Space>::maxAxis(axis) * weight_;
    }

    // template <typename _Adder, typename _T>
    // unsigned addImpl(
    //     _Adder& adder,
    //     unsigned axis,
    //     KDNode<_T>* p,
    //     KDNode<_T>* n,
    //     unsigned depth)
    // {
    //     return KDAddTraversal<_Space>::addImpl(
    //         adder, axis, p, n, depth);
    // }
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

// // Compute the number of dimensions in the subspaces before the specified index.
// template <int _index, typename ... _Spaces>
// struct DimensionsBefore {
//     static constexpr int value = DimensionsBefore<_index-1, _Spaces...>::value +
//         std::tuple_element<_index-1, std::tuple<_Spaces...>>::type::dimensions;
// };
// // Base case, there are 0 dimensions before the subspace 0.
// template <typename ... _Spaces>
// struct DimensionsBefore<0, _Spaces...> { static constexpr int value = 0; };

template <int _index, typename ... _Spaces>
struct CompoundMaxAxis {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance maxAxis(
        _TraversalsTuple& traversals,
        unsigned dimBefore,
        typename Space::Distance bestDist,
        unsigned *bestAxis)
    {
        unsigned axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        // assert((0 <= axis && axis < std::tuple_element<_index, std::tuple<_Spaces...>>::type::dimensions));
        if (d > bestDist) {
            *bestAxis = dimBefore + axis;
            bestDist = d;
        }
        return CompoundMaxAxis<_index+1, _Spaces...>::maxAxis(
            traversals, dimBefore + std::get<_index>(traversals).dimensions(), bestDist, bestAxis);
    }
};

template <typename ... _Spaces>
struct CompoundMaxAxis<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;

    template <typename _TraversalsTuple>
    inline static typename Space::Distance maxAxis(
        _TraversalsTuple& traversals,
        unsigned dimBefore,
        typename Space::Distance bestDist,
        unsigned *bestAxis)
    {
        return bestDist;
    }
};

template <int _index, typename ... _Spaces>
struct CompoundAddImpl {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    // static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;

    template <typename _Traversals, typename _Adder, typename _T>
    static inline KDNode<_T>** addImpl(
        _Traversals& traversals,
        _Adder& adder,
        unsigned dimBefore,
        unsigned axis,
        KDNode<_T>* p,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            return std::get<_index>(traversals).addImpl(
                adder, axis - dimBefore, p, depth);
        } else {
            return CompoundAddImpl<_index+1, _Spaces...>::addImpl(
                traversals, adder, dimAfter, axis, p, depth);
        }
    }

    template <typename _Traversals, typename _Remover, typename _T>
    static inline bool removeImpl(
        _Traversals& traversals,
        _Remover& remover,
        unsigned dimBefore,
        unsigned axis,
        KDNode<_T>* p,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            return std::get<_index>(traversals).removeImpl(
                remover, axis - dimBefore, p, depth);
        } else {
            return CompoundAddImpl<_index+1, _Spaces...>::removeImpl(
                traversals, remover, dimAfter, axis, p, depth);
        }
    }

};

template <typename ... _Spaces>
struct CompoundAddImpl<sizeof...(_Spaces)-1, _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    // static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;

    template <typename _Traversals, typename _Adder, typename _T>
    static inline KDNode<_T>** addImpl(
        _Traversals& traversals,
        _Adder& adder,
        unsigned dimBefore,
        unsigned axis,
        KDNode<_T>* p,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        // assert(axis < dimBefore + CurrentSpace::dimensions);
        return std::get<_index>(traversals).addImpl(
            adder, axis - dimBefore, p, depth);
    }

    template <typename _Traversals, typename _Remover, typename _T>
    static inline bool removeImpl(
        _Traversals& traversals,
        _Remover& remover,
        unsigned dimBefore,
        unsigned axis,
        KDNode<_T>* p,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        // assert(axis < dimBefore + CurrentSpace::dimensions);
        return std::get<_index>(traversals).removeImpl(
            remover, axis - dimBefore, p, depth);
    }

};

template <typename ... _Spaces>
struct KDAddTraversal<CompoundSpace<_Spaces...>> {
    typedef std::tuple<_Spaces...> SpaceTuple;
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    const Space& space_;
    std::tuple<KDAddTraversal<_Spaces>...> traversals_;

    template <std::size_t ... I>
    KDAddTraversal(const Space& space, const State& key, std::index_sequence<I...>)
        : space_(space),
          traversals_(KDAddTraversal<typename std::tuple_element<I, SpaceTuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }
    
    KDAddTraversal(const Space& space, const State& key)
        : KDAddTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }
    
    inline Distance maxAxis(unsigned *axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        // assert((0 <= *axis && *axis < std::tuple_element<0, SpaceTuple>::type::dimensions));
        return CompoundMaxAxis<1, _Spaces...>::maxAxis(traversals_, std::get<0>(space_).dimensions(), d, axis);
    }

    template <typename _Adder, typename _T>
    KDNode<_T>** addImpl(_Adder& adder, unsigned axis, KDNode<_T>* p, unsigned depth) {
        return CompoundAddImpl<0, _Spaces...>::addImpl(traversals_, adder, 0, axis, p, depth);
    }

    template <typename _Remover, typename _T>
    bool removeImpl(_Remover& remover, unsigned axis, KDNode<_T>* p, unsigned depth) {
        return CompoundAddImpl<0, _Spaces...>::removeImpl(traversals_, remover, 0, axis, p, depth);
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
        : KDBoundedL2Traversal<_Scalar, _dimensions>(space, key),
          regionDeltas_(space.dimensions(), 1)
    {
        regionDeltas_.setZero();
    }

    inline Distance distToRegion() {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Nearest, typename _T>
    inline void traverse(_Nearest& nearest, const KDNode<_T>* n, unsigned axis, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
        _Scalar delta = (split - key_[axis]);
        int childNo = delta < 0;

        if (const KDNode<_T>* c = n->children_[childNo]) {
            std::swap(bounds_(axis, 1-childNo), split);
            nearest(c, depth+1);
            std::swap(bounds_(axis, 1-childNo), split);
        }

        nearest.update(n);

        if (const KDNode<_T>* c = n->children_[1-childNo]) {
            Scalar oldDelta = regionDeltas_[axis];
            regionDeltas_[axis] = delta*delta;
            if (nearest.distToRegion() <= nearest.dist()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c, depth+1);
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
    
    Scalar distToRegionCache_ = 0;

    State origKey_;

    KDNearestTraversal(const Space& space, const State& key)
        : KDSO3Traversal<_Scalar>(space, key),
          origKey_(key)
    {
        key_ = rotateCoeffs(key_, keyVol_ + 1);
        if (key_[3] < 0)
            key_ = -key_;
    }

    template <typename _Derived>
    _Scalar keyDistance(const Eigen::QuaternionBase<_Derived>& q) {
        _Scalar dot = std::abs(origKey_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    template <typename _Derived>
    inline Scalar dotBounds(int b, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        // assert(b == 0 || b == 1);
        // assert(0 <= axis && axis < 3);
            
        return soBounds_[b](0, axis)*q[3]
            +  soBounds_[b](1, axis)*q[axis];
    }

    inline Scalar distToRegion() {
        return distToRegionCache_;
    }

    inline Scalar computeDistToRegion() {
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
    inline void traverse(_Nearest& nearest, const KDNode<_T>* n, unsigned axis, unsigned depth) {
        // std::cout << n->value_.name_ << " " << soDepth_ << std::endl;
        if (soDepth_ < 3) {
            ++soDepth_;
            if (const KDNode<_T> *c = n->children_[keyVol_ & 1]) {
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const KDNode<_T> *g = c->children_[keyVol_ >> 1]) {
                    // assert(std::abs(origKey_.coeffs()[keyVol_]) == key_[3]);
                    nearest(g, depth+1);
                }
                // TODO: can we gain so efficiency by exploring the
                // nearest of the remaining 3 volumes first?
                nearest.update(c);
                if (const KDNode<_T> *g = c->children_[1 - (keyVol_ >> 1)]) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 2) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 2]) == key_[3]);
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g, depth+1);
                }
            }
            nearest.update(n);
            if (const KDNode<_T> *c = n->children_[1 - (keyVol_ & 1)]) {
                // std::cout << c->value_.name_ << " " << soDepth_ << ".5" << std::endl;
                if (const KDNode<_T> *g = c->children_[keyVol_ >> 1]) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 1) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 1]) == key_[3]);
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g, depth+1);
                }
                nearest.update(c);
                if (const KDNode<_T> *g = c->children_[1 - (keyVol_ >> 1)]) {
                    key_ = rotateCoeffs(origKey_.coeffs(), (keyVol_ ^ 3) + 1);
                    if (key_[3] < 0)
                        key_ = -key_;
                    // assert(std::abs(origKey_.coeffs()[keyVol_ ^ 3]) == key_[3]);
                    distToRegionCache_ = computeDistToRegion();
                    if (nearest.distToRegion() <= nearest.dist())
                        nearest(g, depth+1);
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
            if (const KDNode<_T> *c = n->children_[childNo]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(axis);
                soBounds_[1-childNo].col(axis) = mp;
// #ifdef KD_PEDANTIC
//                 Scalar soBoundsDistNow = soBoundsDist();
//                 if (soBoundsDistNow + rvBoundsDistCache_ <= dist_) {
//                     std::swap(soBoundsDistNow, soBoundsDistCache_);
// #endif
                nearest(c, depth+1);
// #ifdef KD_PEDANTIC
//                     soBoundsDistCache_ = soBoundsDistNow;
//                 }
// #endif
                soBounds_[1-childNo].col(axis) = tmp;
            }
            nearest.update(n);
            if (const KDNode<_T> *c = n->children_[1-childNo]) {
                Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(axis);
                soBounds_[childNo].col(axis) = mp;
                Scalar oldDistToRegion = distToRegionCache_;
                distToRegionCache_ = computeDistToRegion();
                if (nearest.distToRegion() <= nearest.dist())
                    nearest(c, depth+1);
                distToRegionCache_ = oldDistToRegion;
                soBounds_[childNo].col(axis) = tmp;
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
    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return KDNearestTraversal<_Space>::keyDistance(q) * _num / _den;
    }

    inline Distance maxAxis(unsigned *axis) {
        return KDNearestTraversal<_Space>::maxAxis(axis) * _num / _den;
    }
    
    inline Distance distToRegion() {
        return KDNearestTraversal<_Space>::distToRegion() * _num / _den;
    }

    template <typename _Nearest, typename _T>
    inline void traverse(_Nearest& nearest, const KDNode<_T>* n, unsigned axis, unsigned depth) {
        KDNearestTraversal<_Space>::traverse(nearest, n, axis, depth);
    }
};

template <typename _Space>
struct KDNearestTraversal<WeightedSpace<_Space>>
    : KDNearestTraversal<_Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;
    
    KDNearestTraversal(const Space& space, const State& key)
        : KDNearestTraversal<_Space>(space, key),
          weight_(space.weight())
    {
    }

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with KDAddTraversal.  Would be nice to merge them.
    template <typename _State>
    Distance keyDistance(const _State& q) {
        return KDNearestTraversal<_Space>::keyDistance(q) * weight_;
    }

    Distance maxAxis(unsigned *axis) {
        return KDNearestTraversal<_Space>::maxAxis(axis) * weight_;
    }
    
    Distance distToRegion() {
        return KDNearestTraversal<_Space>::distToRegion() * weight_;
    }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, unsigned axis, unsigned depth) {
        KDNearestTraversal<_Space>::traverse(nearest, n, axis, depth);
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
    // static constexpr unsigned dimBefore = DimensionsBefore<_index, _Spaces...>::value;
    
    template <typename _Traversals, typename _Nearest, typename _T>
    static inline void traverse(
        _Traversals& traversals,
        _Nearest& nearest,
        const KDNode<_T>* n,
        unsigned dimBefore,
        unsigned axis,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore, depth);
        } else {
            CompoundTraverse<_index+1, _Spaces...>::traverse(
                traversals, nearest, n, dimAfter, axis, depth);
        }
    }
};
template <typename ... _Spaces>
struct CompoundTraverse<sizeof...(_Spaces)-1, _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type CurrentSpace;
    // static constexpr int dimBefore = DimensionsBefore<_index, _Spaces...>::value;
    
    template <typename _Traversals, typename _Nearest, typename _T>
    static inline void traverse(
        _Traversals& traversals,
        _Nearest& nearest,
        const KDNode<_T>* n,
        unsigned dimBefore,
        unsigned axis,
        unsigned depth)
    {
        // assert(axis >= dimBefore);
        // assert(axis - dimBefore < CurrentSpace::dimensions);
        std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore, depth);
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

    template <typename _State>
    Distance keyDistance(const _State& q) {
        return CompoundKeyDistance<1, _Spaces...>::accum(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    Distance maxAxis(unsigned *axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        // assert((0 <= *axis && *axis < std::tuple_element<0, SpaceTuple>::type::dimensions));
        return CompoundMaxAxis<1, _Spaces...>::maxAxis(traversals_, std::get<0>(traversals_).dimensions(), d, axis);
    }

    Distance distToRegion() {
        // if (true) return std::get<0>(traversals_).distToRegion();
        // return std::get<0>(traversals_).distToRegion()
        //     + std::get<1>(traversals_).distToRegion();
        return CompoundDistToRegion<1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _Nearest, typename _T>
    inline void traverse(_Nearest& nearest, const KDNode<_T>* n, unsigned axis, unsigned depth) {
        CompoundTraverse<0, _Spaces...>::traverse(traversals_, nearest, n, 0, axis, depth);
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDAdder {
    typedef typename _Space::Distance Distance;

    KDTreeBase<_T, _Space, _TtoKey>& tree_;
    KDAddTraversal<_Space> traversal_;
    
    KDAdder(KDTreeBase<_T, _Space, _TtoKey>& tree,
            const typename _Space::State& key)
        : tree_(tree),
          traversal_(tree.space_, key)
    {
    }

    inline KDNode<_T>** operator() (KDNode<_T>*& p, unsigned depth) {
        auto it = tree_.removedSet_.find(p);
        if (it != tree_.removedSet_.end()) {
            tree_.removedSet_.erase(it);
            return &p;
        }
        
        if (depth >= tree_.axisCache_.size()) {
            unsigned axis;
            traversal_.maxAxis(&axis);
            tree_.axisCache_.push_back(axis);
        }
        
        return traversal_.addImpl(*this, tree_.axisCache_[depth], p, depth);
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDRemover {
    KDTreeBase<_T, _Space, _TtoKey>& tree_;
    KDAddTraversal<_Space> traversal_;
    const _T& valueToRemove_;

    KDRemover(
        KDTreeBase<_T, _Space, _TtoKey>& tree,
        const _T& value)
        : tree_(tree),
          traversal_(tree.space_, tree.tToKey_(value)),
          valueToRemove_(value)
    {
    }

    inline bool remove(KDNode<_T>* n) {
        // check if already removed or the value is not the correct
        // value to remove, if so, return false.
        if (tree_.removedSet_.count(n) || n->value_ != valueToRemove_)
            return false;

        // otherwise remove it and return true
        tree_.removedSet_.insert(n);
        --tree_.size_;
        return true;
    }
    
    inline bool operator() (KDNode<_T>* n, unsigned depth) {
        return remove(n)
            || ((n->children_[0] != n->children_[1])
                && traversal_.removeImpl(*this, tree_.axisCache_[depth], n, depth));
    }
};

template <typename _Derived, typename _T, typename _Space, typename _TtoKey>
struct KDNearestBase {
    typedef typename _Space::Distance Distance;

    const KDTreeBase<_T, _Space, _TtoKey>& tree_;
    KDNearestTraversal<_Space> traversal_;
    Distance dist_;
    
    inline KDNearestBase(
        const KDTreeBase<_T, _Space, _TtoKey>& tree,
        const typename _Space::State& key,
        Distance dist = std::numeric_limits<Distance>::infinity())
        : tree_(tree),
          traversal_(tree.space_, key),
          dist_(dist)
    {
    }

    inline Distance dist() {
        return dist_;
    }

    inline Distance distToRegion() {
        Distance d = traversal_.distToRegion();
        // assert(d == d); // not NaN
        return d;
    }

    void operator() (const KDNode<_T>* n, unsigned depth) {
        if (n->children_[0] == n->children_[1]) {
            // only possible when both children are null
            update(n);
        } else {
            traversal_.traverse(static_cast<_Derived&>(*this), n, tree_.axisCache_[depth], depth);
        }
    }

    void update(const KDNode<_T>* n) {
        if (tree_.removedSet_.size() > 0 && tree_.removedSet_.count(n))
            return;
        
        Distance d = traversal_.keyDistance(tree_.tToKey_(n->value_));
        // assert(d == space_.distance(this->tToKey_(n->value_), key_));
        // std::cout << "update " << n->value_.name_ << " " << d << std::endl;
        if (d < dist_) {
            static_cast<_Derived*>(this)->updateImpl(d, n);
        }
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDNearest1 : KDNearestBase<KDNearest1<_T, _Space, _TtoKey>, _T, _Space, _TtoKey> {
    typedef typename _Space::Distance Distance;

    const KDNode<_T>* nearest_;

    inline KDNearest1(
        const KDTreeBase<_T, _Space, _TtoKey>& tree,
        const typename _Space::State& key)
        : KDNearestBase<KDNearest1, _T, _Space, _TtoKey>(tree, key),
          nearest_(nullptr)
    {
    }
    
    inline void updateImpl(Distance d, const KDNode<_T> *n) {
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

template <typename _T, typename _Space, typename _TtoKey>
struct KDNearestK : KDNearestBase<KDNearestK<_T, _Space, _TtoKey>, _T, _Space, _TtoKey> {
    typedef typename _Space::Distance Distance;

    std::size_t k_;
    std::vector<std::pair<Distance, _T>>& nearest_;

    inline KDNearestK(
        const KDTreeBase<_T, _Space, _TtoKey>& tree,
        std::vector<std::pair<Distance, _T>>& nearest,
        std::size_t k,
        Distance r,
        const typename _Space::State& key)
        : KDNearestBase<KDNearestK, _T, _Space, _TtoKey>(tree, key, r),
          k_(k),
          nearest_(nearest)
    {
        // update() will not work unless the following to initial criteria hold:
        // assert(k > 0);
        // assert(nearest.size() <= k);
    }

    void updateImpl(Distance d, const KDNode<_T>* n) {
        if (nearest_.size() == k_) {
            std::pop_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
            nearest_.pop_back();
        }
        nearest_.emplace_back(d, n->value_);
        std::push_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
        if (nearest_.size() == k_)
            this->dist_ = nearest_[0].first;
    }
};

    
} // namespace unc::robotics::kdtree::detail

template <typename _T, typename _Space, typename _TtoKey>
class KDTree : private detail::KDTreeBase<_T, _Space, _TtoKey> {
    typedef detail::KDNode<_T> Node;
    typedef _Space Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    Node *root_ = nullptr;

public:
    KDTree(_TtoKey tToKey, const Space& space = _Space())
        : detail::KDTreeBase<_T, _Space, _TtoKey>(tToKey, space)
    {
    }

    // copy constructor disabled since it requires a deep clone of the tree
    KDTree(const KDTree&) = delete;

    // moves are cheap though
    KDTree(KDTree&& other)
        : detail::KDTreeBase<_T, _Space, _TtoKey>(std::move(other)),
          root_(other.root_)
    {
        other.root_ = nullptr;
    }

    ~KDTree() {
        delete root_;
    }

    inline std::size_t size() const {
        return this->size_;
    }

    inline bool empty() const {
        return size() == 0;
    }

    inline unsigned depth() const {
        return this->axisCache_.size() + 1;
    }

    void clear() {
        this->removedSet_.clear();
        this->size_ = 0;
        delete root_;
    }

    bool remove(const _T& value) {
        if (!root_)
            return false;
        
        detail::KDRemover<_T, _Space, _TtoKey> remover(*this, value);
        return remover(root_, 0);
    }

    void add(const _T& value) {
        if (root_ == nullptr) {
            root_ = new Node(value);
        } else {
            const Key& key = this->tToKey_(value);
            detail::KDAdder<_T, _Space, _TtoKey> adder(*this, key);
            Node** ptr = adder(root_, 0);
            if (*ptr) {
                // this node was previously removed, we reuse it now.
                (*ptr)->value_ = value;
            } else {
                *ptr = new Node(value);
            }
        }
        
        ++this->size_;
    }

    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        if (!root_)
            return nullptr;
        
        detail::KDNearest1<_T, _Space, _TtoKey> nearest(*this, key);
        nearest(root_, 0);
        if (distOut)
            *distOut = nearest.dist_;
        
        return &nearest.nearest_->value_;
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

        detail::KDNearestK<_T, _Space, _TtoKey> nearestK(*this, result, k, maxRadius, key);
        nearestK(root_, 0);
        std::sort_heap(result.begin(), result.end(), detail::DistValuePairCompare());
    }
};

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_KDTREE_HPP
