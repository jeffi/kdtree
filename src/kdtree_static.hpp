#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_STATIC_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_STATIC_HPP

#include <limits>
#include <array>
#include "spaces.hpp"
#include "_bits.hpp"
#include "_so3space.hpp"
#include "_wtspace.hpp"
#include "_compoundspace.hpp"

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {


// Static KD Trees store nodes in contiguous (or at least
// RandomAccess) container, typically std::vector.  Currently nodes
// and values (of type _T) are stored in a single element, though a
// parallel structure may be more appropriate to implement later
// (e.g. keep the original std::vector<_T> and create another
// std::vector<std::pair<AxisType, DistanceType>>.
//
// Sub-trees are referenced by iterator pairs: [begin, end).  For most
// sub-tree types, the nodes are organized s.t. *begin is the root of
// the sub-tree and (begin, mid), [mid, end) are the contained
// subtrees:
//
// [begin,                   end)
//        (begin, mid) [mid, end)
//
// Where
//   mid = begin + std::distance(begin,end)/2;
//
// For the SO(3) root node, there is no split value, but there is a
// split-by-volume:
//
// [begin,                                        end)
//       [q0,              q2)  [q2,              end)
//           (q0, q1) [q1, q2)      (q2, q3) [q3, end)
//
// Where:
//   q0 = begin+1
//   q2 = begin + begin->offset_;
//   q1 = q0 + q0->offset_;
//   q3 = q2 + q2->offset_;
//
// Some care is required since [q0, q2) and/or [q2,end) may be an
// empty range.

// The above strategy produces an indexing relationship in which the
// parent and the first child root are next to each other, as in the
// following:
//
// 0 [1 .. 99)
//    1  [ 2 .. 50)
//         2 [ 3 .. 25)
//        25 [26 .. 50)
//    50 [51 .. 99)
//        51 [52 .. 75)
//        75 [76 .. 99)
//
// A possibly better cache friendly possibility:
//   _Iter root, _Iter begin, _Iter end
// recur(begin + 0, begin+2, mid)
// recur(begin + 1, mid, end);
//
// Causes roots of subtrees to be next to each other.
//
// 0 [1 ... 99)
//    1 [ 3 .. 50)
//        3 [ 5 .. 25)
//            5 [ 7 .. 12)
//            6 [12 .. 25)
//        4 [25 .. 50)
//           25 [27 .. 32)
//           26 [32 .. 50)
//    2 [50 .. 99)
//       50 [52 .. 75)
//           52
//           53
//       51 [75 .. 99)
//           75
//           76
//



template <typename _Scalar, int _dimensions>
struct KDStaticTraversal<L2Space<_Scalar, _dimensions>> {
    typedef L2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    const State& key_;

    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;
    
    KDStaticTraversal(const Space& space, const State& key)
        : key_(key)
    {
        regionDeltas_.setZero();
    }

    inline Distance distToRegion() {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Derived>
    inline Distance keyDistance(const Eigen::MatrixBase<_Derived>& q) {
        return (key_ - q).norm();
    }
    
    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter min, _Iter max) {
        const auto& n = *min++;
        std::array<_Iter, 3> iters{{min, min + std::distance(min, max)/2, max}};
        Distance delta = n.split_ - key_[axis];
        int childNo = delta < 0;
        nearest(iters[childNo], iters[childNo+1]);
        nearest.update(n);
        delta *= delta;
        std::swap(regionDeltas_[axis], delta);
        if (nearest.distToRegion() <= nearest.dist())
            nearest(iters[1-childNo], iters[2-childNo]);
        regionDeltas_[axis] = delta;
    }
};


template <typename _Scalar, int _dimensions>
struct KDStaticTraversal<BoundedL2Space<_Scalar, _dimensions>>
    : KDStaticTraversal<L2Space<_Scalar, _dimensions>>
{
    using KDStaticTraversal<L2Space<_Scalar, _dimensions>>::KDStaticTraversal;
};



// Build


// 0
// 1     8
// 2  5  9  C
// 34 67 AB DE

    
template <typename _Scalar, int _dimensions>
struct KDStaticAccum<L2Space<_Scalar, _dimensions>> {
    typedef L2Space<_Scalar, _dimensions> Space;

    Eigen::Array<_Scalar, _dimensions, 1> min_;
    Eigen::Array<_Scalar, _dimensions, 1> max_;
    
    unsigned dimensions_; // TODO: this is fixed when _dimensions is not Eigen::Dynamic

    inline KDStaticAccum(const Space& space)
        : dimensions_(space.dimensions())
    {
    }

    inline unsigned dimensions() {
        return dimensions_;
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

    _Scalar maxAxis(unsigned *axis) {
        return (max_ - min_).maxCoeff(axis);
    }

    template <typename _Builder, typename _Iter, typename _ToKey>
    void partition(_Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        _Iter mid = begin + (std::distance(begin, end)-1)/2;
        std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
            return toKey(a)[axis] < toKey(b)[axis];
        });
        std::swap(*begin, *mid);
        begin->split_ = toKey(*begin)[axis];
        builder(++begin, ++mid);
        builder(mid, end);
    }
};


template <typename _Scalar, int _dimensions>
struct KDStaticAccum<BoundedL2Space<_Scalar, _dimensions>>
    : KDStaticAccum<L2Space<_Scalar, _dimensions>>
{
    using KDStaticAccum<L2Space<_Scalar, _dimensions>>::KDStaticAccum;
};



template <typename _T, typename _Space, typename _TtoKey>
struct KDStaticBuilder {
    KDStaticAccum<_Space> accum_;
    _TtoKey tToKey_;
    // int indent_ = 0;
    
    KDStaticBuilder(const _Space& space, const _TtoKey& tToKey)
        : accum_(space),
          tToKey_(tToKey)
    {
    }
    
    template <typename _Iter>
    void operator() (_Iter begin, _Iter end) {
        if (begin == end)
            return;

        // for (int i=0 ; i<indent_ ; ++i)
        //     std::cout << "  ";
        // std::cout << begin->value_.name_ << std::endl;
    
        _Iter it = begin;
        accum_.init(tToKey_(it->value_));
        while (++it != end)
            accum_.accum(tToKey_(it->value_));

        // ++indent_;
        unsigned axis;
        accum_.maxAxis(&axis);
        accum_.partition(*this, axis, begin, end, [&] (auto& t) -> auto& { return tToKey_(t.value_); });
        begin->axis_ = axis;
        // --indent_;
    }
};

template <typename _Derived, typename _T, typename _Space, typename _TtoKey>
struct KDStaticNearestBase {
    typedef typename _Space::Distance Distance;

    const KDStaticTreeBase<_T, _Space, _TtoKey>& tree_;
    KDStaticTraversal<_Space> traversal_;
    Distance dist_;

    inline KDStaticNearestBase(
        const KDStaticTreeBase<_T, _Space, _TtoKey>& tree,
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
        return traversal_.distToRegion();
    }

    template <typename Iter>
    void operator() (Iter begin, Iter end) {
        if (begin != end) {
            traversal_.traverse(static_cast<_Derived&>(*this), begin->axis_, begin, end);
        }
    }

    template <typename Node>
    void update(const Node& n) {
        Distance d = traversal_.keyDistance(tree_.tToKey_(n.value_));
        if (d < dist_)
            static_cast<_Derived*>(this)->updateImpl(d, &n);
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDStaticNearest1
    : KDStaticNearestBase<KDStaticNearest1<_T, _Space, _TtoKey>, _T, _Space, _TtoKey>
{
    typedef typename _Space::Distance Distance;
    
    const KDStaticNode<_T, Distance, std::ptrdiff_t>* nearest_ = nullptr;

    inline KDStaticNearest1(
        const KDStaticTreeBase<_T, _Space, _TtoKey>& tree,
        const typename _Space::State& key)
        : KDStaticNearestBase<KDStaticNearest1, _T, _Space, _TtoKey>(tree, key)
    {
    }

    template <typename Node>
    inline void updateImpl(Distance d, const Node *n) {
        this->dist_ = d;
        nearest_ = n;
    }
};

template <typename _T, typename _Space, typename _TtoKey>
struct KDStaticNearestK {
    typedef typename _Space::Distance Distance;
    
    std::size_t k_;
    std::vector<std::pair<Distance, _T>>& nearest_;

    inline KDStaticNearestK(
        const KDStaticTreeBase<_T, _Space, _TtoKey>& tree,
        std::vector<std::pair<Distance, _T>>& nearest,
        std::size_t k,
        Distance r,
        const typename _Space::State& key)
        : KDStaticNearestBase<KDStaticNearestK, _T, _Space, _TtoKey>(tree, key, r),
          k_(k),
          nearest_(nearest)
    {
    }

    template <typename Node>
    void updateImpl(Distance d, const Node* n) {
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


// Node build(_Iter begin, _Iter end) {
//     assert(begin != end);
    
//     _Iter it = begin;
//     Eigen::Array<_Scalar, _dim, 1> min_ = std::get<1>(tToKey_(it->value_));
//     Eigen::Array<_Scalar, _dim, 1> max_ = min_;
//     Eigen::Array<_Scalar, 2, 3> soMin, soMax;

//     // TODO: initialize to min/max of first elements
//     soMin << 1,1,1, 0,0,0;
//     soMax = -soMin;
    
//     while (++it != end) {
//         min_ = min_.cwiseMin(std::get<1>(tToKey_(it->value_)));
//         max_ = max_.cwiseMax(std::get<1>(tToKey_(it->value_)));

//         if (vol_ != -1) {
//             for (unsigned soAxis = 0 ; soAxis<3 ; ++soAxis) {
//                 Eigen::Vector2d split = projectToAxis(std::get<0>(tToKey_(it->value_)), vol_, soAxis);
//                 if (split[0] < soMin_(0, axis))
//                     soMin_.col(soAxis) = split;
//                 if (split[0] > soMin_(0, axis))
//                     soMax_.col(soAxis) = split;
//             }
//         }
//     }

//     unsigned rvAxis;
//     _Scalar rvDist = (max_ - min_).maxCoeff(&rvAxis);
//     unsigned soAxis;
//     _Scalar soDist = (soMin_ * soMax_).colwise().sum().minCoeff(&soAxis);

//     _Iter sub = begin;
//     ++sub;
//     _Iter mid = sub + std::distance(sub, end)/2;

//     if (rvAxis > soAxis) {
//         std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
//             return tToKey_(a)[axis] < tToKey_(b)[axis];
//         });
//         std::swap(*begin, *mid);
//         build(sub, ++mid);
//         build(mid, end);
//     } else if (vol_ == -1) {
//         // TODO: build quadrants
//         // nested build uses different accumulator
//         build(++begin, q1);
//         build(q1, q2);
//         build(q2, q3);
//         build(q3, end);
//     } else {
//         std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
//             Eigen::Matrix<Scalar, 2, 1> aSplit = projectToAxis(tToKey_(a), vol_, soAxis);
//             Eigen::Matrix<Scalar, 2, 1> bSplit = projectToAxis(tToKey_(b), vol_, soAxis);
//             return aSplit[0] < bSplit[0];
//         });
//         std::swap(*begin, *mid);
//         build(sub, ++mid);
//         build(mid, end);
//     }

// }


} // namespace unc::robotics::kdtree::detail

template <typename _T, typename _Space, typename _TtoKey>
class KDStaticTree : detail::KDStaticTreeBase<_T, _Space, _TtoKey> {

    typedef _Space Space;
    typedef typename _Space::State State;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;
    typedef detail::KDStaticNode<_T, Distance, std::ptrdiff_t> Node;
    
    std::vector<Node> nodes_;
        
public:
    template <typename _It>
    KDStaticTree(const _TtoKey& tToKey, const Space& space, _It begin, _It end)
        : detail::KDStaticTreeBase<_T, _Space, _TtoKey>(tToKey, space)
    {
        std::transform(begin, end, std::back_inserter(nodes_), [&](auto& t) {
            return Node(t);
        });
        
        detail::KDStaticBuilder<_T, _Space, _TtoKey> builder(space, tToKey);
        builder(nodes_.begin(), nodes_.end());
    }

    constexpr std::size_t size() const {
        return nodes_.size();
    }

    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        if (size() == 0)
            return nullptr;

        detail::KDStaticNearest1<_T, _Space, _TtoKey> nearest(*this, key);
        nearest(nodes_.begin(), nodes_.end());
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

        detail::KDStaticNearestK<_T, _Space, _TtoKey> nearest(*this, result, k, maxRadius, key);
        nearest(nodes_.begin(), nodes_.end());
        std::sort_heap(result.begin(), result.end(), detail::DistValuePairCompare());
    }
};

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_KDTREE_STATIC_HPP