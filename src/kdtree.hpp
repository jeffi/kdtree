#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_HPP

#include "spaces.hpp"
#include <array>

namespace unc {
namespace robotics {
namespace kdtree {

namespace detail {
template <typename _T>
struct KDNode {
    _T value_;
    std::array<KDNode*, 2> children_;

    KDNode(const _T& v)
        : value_(v),
          children_{{nullptr, nullptr}}
    {
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
        _Scalar split = (bounds_(axis, 0) + bounds_(axis,1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T>* c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return adder(c, n, depth+1);
        } else {
            p->children_[childNo] = n;
            return depth;
        }
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
    Eigen::Array<_Scalar, _dimensions, 1> deltas_;

    KDNearestTraversal(const Space& space, const State& key)
        : KDBoundedL2Traversal<_Scalar, _dimensions>(space, key)
    {
        deltas_.setZero();
    }

    Distance distToRegion() {
        return std::sqrt(deltas_.sum());
    }

    template <typename _Nearest, typename _T>
    void traverse(_Nearest& nearest, const KDNode<_T>* n, int axis, typename _Nearest::Distance d) {
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
            Scalar oldDelta = deltas_[axis];
            deltas_[axis] = delta*delta;
            if (nearest.distToRegion() <= nearest.dist()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c);
                std::swap(bounds_(axis, childNo), split);
            }
            deltas_[axis] = oldDelta;
        }

    }
};

template <typename _Space>
struct KDAdder : KDAddTraversal<_Space> {
    typedef typename _Space::Distance Distance;
    
    KDAdder(const _Space& space, const typename _Space::State& key)
        : KDAddTraversal<_Space>(space, key)
    {
    }

    template <typename _T>
    unsigned operator() (KDNode<_T>* p, KDNode<_T>*n, unsigned depth) {
        int axis;
        Distance d = this->maxAxis(&axis);
        return this->addImpl(*this, axis, d, p, n, depth);
    }
};

template <typename _Derived, typename _Space, typename _T, typename _TtoKey>
struct KDNearestBase : KDNearestTraversal<_Space> {
    typedef typename _Space::Distance Distance;

    _TtoKey tToKey_;
    Distance dist_;

    KDNearestBase(
        const _Space& space,
        const typename _Space::State& key, _TtoKey tToKey,
        Distance dist = std::numeric_limits<Distance>::infinity())
        : KDNearestTraversal<_Space>(space, key),
          tToKey_(tToKey),
          dist_(dist)
    {
    }

    Distance dist() {
        return dist_;
    }

    void operator() (const KDNode<_T>* n) {
        int axis;
        Distance d = this->maxAxis(&axis);
        this->traverse(static_cast<_Derived&>(*this), n, axis, d);
    }

    void update(const KDNode<_T>* n) {
        Distance d = this->keyDistance(this->tToKey_(n->value_));
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
        if (nearest_.size() == k_) {
            std::pop_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
            nearest_.pop_back();
        }
        nearest_.emplace_back(d, n->value_);
        std::push_heap(nearest_.begin(), nearest_.end(), DistValuePairCompare());
        if (nearest_.size() == k_)
            this->dist_ = nearest_.front().first;
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
