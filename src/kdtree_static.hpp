#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_STATIC_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_STATIC_HPP

#include <limits>
#include <array>
#include "spaces.hpp"
#include "_bits.hpp"
#include "_l2space.hpp"
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





// Build


// 0
// 1     8
// 2  5  9  C
// 34 67 AB DE

    

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
struct KDStaticNearestK
    : KDStaticNearestBase<KDStaticNearestK<_T, _Space, _TtoKey>, _T, _Space, _TtoKey>
{
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
