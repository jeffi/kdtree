#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP

#include "_spaces.hpp"

namespace unc { namespace robotics { namespace kdtree {

struct MidpointSplit {};
struct MedianSplit {};

template <
    typename _T,
    typename _Space,
    typename _GetKey,
    typename _SplitStrategy,
    bool _dynamic = true,
    bool _lockfree = false>
struct KDTree;

namespace detail {
template <typename _Node, typename _Space>
struct MidpointAddTraversal;
template <typename _Node, typename _Space>
struct MidpointNearestTraversal;
}

}}}

#include "_l2space.hpp"
#include "_so3space.hpp"
#include "_wtspace.hpp"
#include "_compoundspace.hpp"

#endif // UNC_ROBOTICS_KDTREE_KDTREE_BASE_HPP
