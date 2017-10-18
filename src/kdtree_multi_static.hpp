#pragma once
#ifndef UNC_ROBOTICS_KDTREE_STATIC_TO_DYNAMIC_HPP
#define UNC_ROBOTICS_KDTREE_STATIC_TO_DYNAMIC_HPP

#include "kdtree_static.hpp"

namespace unc {
namespace robotics {
namespace kdtree {

template <typename _T, typename _Space, typename _TtoKey>
class KDMultiStaticTree : detail::KDStaticTreeBase<_T, _Space, _TtoKey> {
    
    // must be a power of 2
    static constexpr std::size_t minStaticTreeSize_ = 2;

    typedef _Space Space;
    typedef typename _Space::State State;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;
    typedef detail::KDStaticNode<_T, Distance, std::ptrdiff_t> Node;
    typedef std::vector<Node> Nodes;
    typedef typename Nodes::iterator Iter;
    typedef typename Nodes::const_iterator ConstIter;

    // nodes_ is organized into trees of decreasing power-of-two
    // sizes, the sum of which totals to the number of points in the
    // tree.
    //
    // size: trees
    // 1: 1       [0]
    // 2: 2       [0 1]
    // 3: 2 1     [0 1|2]
    // 4: 4       [0 1 2 3]
    // 5: 4 1     [0 1 2 3|4]
    // 6: 4 2     [0 1 2 3|4 5]
    // 7: 4 2 1   [0 1 2 3|4 5|6]
    // 8: 8       [0 1 2 3 4 5 6 7]
    // ...
    // 20: 16 4   [0 ... 15|16 17 18 19]
    // 28: 16 8 4 [0 ... 15|16 17 18 19 20 21 22 23|24 25 26 27]
    //
    // We implicitly store the trees by making use of bitwise
    // operations on the tree size.

    Nodes nodes_;
    detail::KDStaticBuilder<_T, _Space, _TtoKey> builder_;

    template <typename _Nearest>
    inline void scanTrees(_Nearest& nearest) const {
        ConstIter it = nodes_.begin();
        for (std::size_t remaining = size() ; remaining >= minStaticTreeSize_ ; ) {
            std::size_t treeSize = detail::log2(remaining);
            nearest(it, it + treeSize);
            it += treeSize;
            remaining &= ~(std::size_t(1) << treeSize);
        }

        for ( ; it != nodes_.end() ; ++it)
            nearest.update(*it);
    }
    
public:
    KDMultiStaticTree(const _TtoKey& tToKey, const Space& space)
        : detail::KDStaticTreeBase<_T, _Space, _TtoKey>(tToKey, space),
          builder_(space, tToKey)
    {
    }

    constexpr std::size_t size() const {
        return nodes_.size();
    }

    constexpr bool empty() const {
        return nodes_.empty();
    }

    void add(const _T& value) {
        nodes_.emplace_back(value);
        std::size_t s = nodes_.size();

        // determine the size of the new subtree to construct.
        // E.g., when the number of nodes becomes 20, the new treeSize is 4. (16+4 = 20)
        std::size_t newTreeSize = ((s^(s-1)) + 1) >> 1;

        // Overwrite the tree structures with the new tree structure
        // E.g., when the number of nodes becomes 20, newTreeSize is
        // 4, the subtrees will go from (16, 2, 1) to (16, 4), noting
        // that the trees (2, 1) and the new element that follows will
        // be overwritten with a subtree of size 4.
        if (newTreeSize >= minStaticTreeSize_)
            builder_(nodes_.end() - newTreeSize, nodes_.end());
    }

    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        if (size() == 0)
            return nullptr;

        detail::KDStaticNearest1<_T, _Space, _TtoKey> nearest(*this, key);
        scanTrees(nearest);
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
        scanTrees(nearest);
        std::sort_heap(result.begin(), result.end(), detail::DistValuePairCompare());
    }
};


} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_STATIC_TO_DYNAMIC_HPP
