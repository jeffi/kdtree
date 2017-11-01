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
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_MEDIAN_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_MEDIAN_HPP

namespace unc { namespace robotics { namespace kdtree {

template <typename _Node, typename _Distance>
struct MedianSplitNodeMember {
    union {
        _Distance split_;
        // TODO: consider making offset_'s type a template parameter
        std::ptrdiff_t offset_;
    };
    unsigned axis_;
};

namespace detail {

template <typename _T, typename _Distance>
struct MedianSplitNode {
    MedianSplitNodeMember<MedianSplitNode<_T, _Distance>, _Distance> hook_;
    _T data_;
    
    // TODO: delete?
    // MedianSplitNode(const MedianSplitNode&) = delete;
    // MedianSplitNode(MedianSplitNode&&) = delete;
    
    MedianSplitNode(const _T& value) : data_(value) {}
    template <typename ... _Args>
    MedianSplitNode(_Args&& ... args) : _T(std::forward<_Args>(args)...) {}
};

template <typename _T, typename _Distance, typename _GetKey>
struct MedianSplitNodeKey : _GetKey {
    inline MedianSplitNodeKey(const _GetKey& getKey) : _GetKey(getKey) {}

    constexpr decltype(auto) operator() (const MedianSplitNode<_T, _Distance>& node) const {
        return _GetKey::operator()(node.data_);
    }
    constexpr decltype(auto) operator() (MedianSplitNode<_T, _Distance>& node) {
        return _GetKey::operator()(node.data_);
    }
};

template <typename _Node, typename _Space, typename _GetKey,
          MedianSplitNodeMember<_Node, typename _Space::Distance> _Node::* _member>
struct MedianBuilder {
    typedef _Space Space;
    typedef _Node Node;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef MedianSplitNodeMember<Node, Distance> Member;
        
    MedianAccum<_Space> accum_;
    _GetKey getKey_;

    MedianBuilder(const _Space& space, const _GetKey& getKey)
        : accum_(space),
          getKey_(getKey)
    {
    }

    static void setSplit(Node& node, Distance split) {
        (node.*_member).split_ = split;
    }

    static void setOffset(Node& node, std::ptrdiff_t offset) {
        (node.*_member).offset_ = offset;
    }

    template <typename _Iter>
    void operator() (_Iter begin, _Iter end) {
        if (begin == end)
            return;

        _Iter it = begin;
        accum_.init(getKey_(*it));
        while (++it != end)
            accum_.accum(getKey_(*it));

        unsigned axis;
        accum_.maxAxis(&axis);
        accum_.partition(*this, axis, begin, end, getKey_);
        ((*begin).*_member).axis_ = axis;
    }
};

template <typename _Derived, typename _Tree>
struct MedianNearest {
    typedef _Tree Tree;

    typedef typename Tree::Space Space;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef typename Tree::Node Node;

    const Tree& tree_;
    MedianNearestTraversal<Space> traversal_;
    Distance dist_;
    
    MedianNearest(const Tree& tree, const Key& key, Distance dist)
        : tree_(tree), traversal_(tree.space(), key), dist_(dist)
    {
    }
    
    constexpr bool shouldTraverse() const {
        return traversal_.distToRegion() <= dist_;
    }

    static constexpr Distance split(const Node& n) {
        return (n.*_Tree::member_).split_;
    }

    static constexpr std::ptrdiff_t offset(const Node& n) {
        return (n.*_Tree::member_).offset_;
    }
    
    template <typename _Iter>
    inline void operator() (_Iter begin, _Iter end) {
        assert(begin <= end);
        if (begin != end)
            traversal_.traverse(*this, ((*begin).*_Tree::member_).axis_, begin, end);
    }

    template <typename Node>
    void updateX(const Node& n) { update(n); }

    template <typename Node>
    void update(const Node& n) {
        Distance d = traversal_.keyDistance(tree_.getNodeKey_(n));
        if (d <= dist_)
            static_cast<_Derived*>(this)->update(d, n);
    }    
};

template <typename _Tree>
struct MedianNearest1
    : MedianNearest<MedianNearest1<_Tree>, _Tree>
{
    typedef MedianNearest<MedianNearest1<_Tree>, _Tree> Base;
    typedef typename _Tree::Space Space;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef typename Base::Node Node;
    
    using MedianNearest<MedianNearest1<_Tree>, _Tree>::MedianNearest;
    using Base::dist_;

    const Node* nearest_ = nullptr;

    // MedianNearest1(const _Tree& tree, const Key& key)
    //     : Base(tree, key)
    // {
    //     std::cout << "nearest = " << nearest_ << std::endl;
    // }

    inline void update(Distance d, const Node& n) {
        dist_ = d;
        nearest_ = &n;
        // std::cout << d << std::endl;
        // std::cout << nearest_ << std::endl;
    }
};

template <typename _Tree, typename _Allocator, typename _Value, typename _NodeValueFn>
struct MedianNearestK
    : MedianNearest<MedianNearestK<_Tree, _Allocator, _Value, _NodeValueFn>, _Tree>
{
    typedef _Tree Tree;
    typedef _Value Value;
    typedef _NodeValueFn NodeValueFn;
    typedef MedianNearest<MedianNearestK<Tree, _Allocator, Value, NodeValueFn>, Tree> Base;
    typedef typename _Tree::Space Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef typename Base::Node Node;
    
    using Base::dist_;

    std::size_t k_;
    std::vector<std::pair<Distance, Value>, _Allocator>& nearest_;
    NodeValueFn nodeValueFn_;

    MedianNearestK(
        const Tree& tree,
        const Key& key,
        Distance dist,
        std::size_t k,
        std::vector<std::pair<Distance, Value>, _Allocator>& nearest,
        NodeValueFn&& nodeValueFn)
        : Base(tree, key, dist), k_(k), nearest_(nearest), nodeValueFn_(nodeValueFn)
    {
    }
    
    inline void update(Distance d, const Node& n) {
        if (nearest_.size() == k_) {
            std::pop_heap(nearest_.begin(), nearest_.end(), CompareFirst());
            nearest_.pop_back();
        }
        
        nearest_.emplace_back(d, nodeValueFn_(n));
        std::push_heap(nearest_.begin(), nearest_.end(), CompareFirst());

        if (nearest_.size() == k_)
            dist_ = nearest_[0].first;
    }
};



} // namespace detail

// Static Median-Split Tree (lock-free operation is not supported)
template <
    typename _T,
    typename _Space,
    typename _GetKey>
struct KDTree<_T, _Space, _GetKey, MedianSplit, StaticBuild, SingleThread> {
    typedef _Space Space;

private:
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef _GetKey GetKey;
    
    typedef detail::MedianSplitNode<_T, Distance> Node;
    typedef detail::MedianSplitNodeKey<_T, Distance, _GetKey> GetNodeKey;
    
    static constexpr auto member_ = &Node::hook_;
    
    Space space_;
    GetNodeKey getNodeKey_;
    
    std::vector<Node> nodes_;

    template<typename, typename> friend struct detail::MedianNearest;
    
public:
    KDTree(const Space& space, const _GetKey& getKey = _GetKey())
        : space_(space),
          getNodeKey_(getKey)
    {
    }

    template <typename _Iter>
    KDTree(const Space& space, const _GetKey& getKey, _Iter begin, _Iter end)
        : KDTree(space, getKey)
    {
        build(begin, end);
    }

    template <typename _Container>
    KDTree(const Space& space, const _GetKey& getKey, const _Container& container)
        : KDTree(space, getKey)
    {
        build(container);
    }

    template <typename _Iter>
    void build(_Iter begin, _Iter end) {
        nodes_.clear();
        nodes_.reserve(std::distance(begin, end));
        std::transform(begin, end, std::back_inserter(nodes_), [&](auto& v) { return Node(v); });
        
        detail::MedianBuilder<Node, Space, GetNodeKey, &Node::hook_> builder(space_, getNodeKey_);
        builder(nodes_.begin(), nodes_.end());
    }

    template <typename _Container>
    void build(const _Container& container) {
        build(container.begin(), container.end());
    }

    constexpr const Space& space() const {
        return space_;
    }

    constexpr std::size_t size() const {
        return nodes_.size();
    }

    // TODO: non-const version returning non-const result?
    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        if (nodes_.size() == 0)
            return nullptr;

        detail::MedianNearest1<KDTree> nearest(
            *this, key, std::numeric_limits<Distance>::infinity());
        nearest(nodes_.begin(), nodes_.end());
        if (distOut)
            *distOut = nearest.dist_;

        return &(nearest.nearest_->data_);
    }

    template <typename _Allocator, typename _Value, typename _NodeValueFn>
    void nearest(
        std::vector<std::pair<Distance, _Value>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxRadius,
        _NodeValueFn&& nodeValueFn) const
    {
        result.clear();
        if (k == 0)
            return;

        detail::MedianNearestK<KDTree, _Allocator, _Value, _NodeValueFn> nearest(
            *this, key, maxRadius, k, result, std::forward<_NodeValueFn>(nodeValueFn));
        
        nearest(nodes_.begin(), nodes_.end());
        std::sort_heap(result.begin(), result.end(), detail::CompareFirst());
    }

    template <typename _Allocator>
    void nearest(
        std::vector<std::pair<Distance, _T>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
    {
        nearest(result, key, k, maxRadius, [] (const Node& n) -> const auto& { return n.data_; });
    }
};

// Dynamically balanced tree
template <
    typename _T,
    typename _Space,
    typename _GetKey>
struct KDTree<_T, _Space, _GetKey, MedianSplit, DynamicBuild, SingleThread> {
    typedef _Space Space;

private:
    // must be a power of 2
    static constexpr std::size_t minStaticTreeSize_ = 2;

    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef _GetKey GetKey;

    typedef detail::MedianSplitNode<_T, Distance> Node;
    typedef detail::MedianSplitNodeKey<_T, Distance, _GetKey> GetNodeKey;

    typedef std::vector<Node> Nodes;
    typedef typename Nodes::iterator Iter;
    typedef typename Nodes::const_iterator ConstIter;
    
    static constexpr auto member_ = &Node::hook_;

    Space space_;
    Nodes nodes_;
    detail::MedianBuilder<Node, Space, GetNodeKey, &Node::hook_> builder_;

    template <typename, typename> friend struct detail::MedianNearest;
    
    template <typename _Nearest>
    inline void scanTrees(_Nearest& nearest) const {
        ConstIter it = nodes_.begin();
        for (std::size_t remaining = size() ; remaining >= minStaticTreeSize_ ; ) {
            std::size_t treeSize = 1 << detail::log2(remaining);
            // std::cout << "scan " << remaining << " -> " << treeSize << std::endl;
            nearest(it, it + treeSize);
            it += treeSize;
            remaining &= ~treeSize;
        }

        for ( ; it != nodes_.end() ; ++it)
            nearest.updateX(*it);
    }

    constexpr const Key& getNodeKey_(const Node& node) const {
        return builder_.getKey_(node);
    }

public:
    KDTree(const Space& space, const _GetKey& getKey = _GetKey())
        : space_(space),
          builder_(space, getKey)
    {
    }

    constexpr const Space& space() const {
        return space_;
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

        std::size_t newTreeSize = ((s^(s-1)) + 1) >> 1;

        if (newTreeSize >= minStaticTreeSize_)
            builder_(nodes_.end() - newTreeSize, nodes_.end());
    }
    
    const _T* nearest(const Key& key, Distance *distOut = nullptr) const {
        if (empty())
            return nullptr;

        detail::MedianNearest1<KDTree> nearest(
            *this, key, std::numeric_limits<Distance>::infinity());
        scanTrees(nearest);
        if (distOut)
            *distOut = nearest.dist_;
        return &(nearest.nearest_->data_);
    }

    template <typename _Allocator, typename _Value, typename _NodeValueFn>
    void nearest(
        std::vector<std::pair<Distance, _Value>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxRadius,
        _NodeValueFn&& nodeValueFn) const
    {
        result.clear();
        if (k == 0)
            return;

        detail::MedianNearestK<KDTree, _Allocator, _Value, _NodeValueFn> nearest(
            *this, key, maxRadius, k, result, std::forward<_NodeValueFn>(nodeValueFn));
        scanTrees(nearest);
        std::sort_heap(result.begin(), result.end(), detail::CompareFirst());
    }

    template <typename _Allocator>
    void nearest(
        std::vector<std::pair<Distance, _T>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxRadius = std::numeric_limits<Distance>::infinity()) const
    {
        nearest(result, key, k, maxRadius, [] (const Node& n) -> const auto& { return n.data_; });
    }


};


}}}

#endif // UNC_ROBOTICS_KDTREE_KDTREE_MEDIAN_HPP
