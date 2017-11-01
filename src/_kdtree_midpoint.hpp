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
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP

#include "_spaces.hpp"

#include <array>

namespace unc { namespace robotics { namespace kdtree {

// When using the intrusive version of the KDTree, the caller must
// provide a member node.
template <typename _Node, bool _destructorDeletes, typename _Locking>
struct MidpointSplitNodeMember;

template <typename _Node, bool _destructorDeletes>
struct MidpointSplitNodeMember<_Node, _destructorDeletes, SingleThread> {
    typedef SingleThread Locking;
        
    std::array<_Node*, 2> children_{};

    MidpointSplitNodeMember(const MidpointSplitNodeMember&) = delete;
    
    ~MidpointSplitNodeMember() {
        if (_destructorDeletes) {
            delete children_[0];
            delete children_[1];
        }
    }

    constexpr _Node* child(int no) { return children_[no]; }
    constexpr const _Node* child(int no) const { return children_[no]; }
    inline constexpr bool hasChild() const { return children_[0] != children_[1]; }
    inline bool update(int no, _Node*, _Node* n) {
        children_[no] = n;
        return true;
    }
};

template <typename _Node, bool _destructorDeletes>
struct MidpointSplitNodeMember<_Node, _destructorDeletes, MultiThread> {
    typedef MultiThread Locking;
    
    std::array<std::atomic<_Node*>, 2> children_{};

    MidpointSplitNodeMember(const MidpointSplitNodeMember&) = delete;
    
    ~MidpointSplitNodeMember() {
        if (_destructorDeletes) {
            delete children_[0].load(std::memory_order_acquire);
            delete children_[1].load(std::memory_order_acquire);
        }
    }

    constexpr _Node* child(int no) { return children_[no].load(std::memory_order_acquire); }
    constexpr const _Node* child(int no) const {
        return children_[no].load(std::memory_order_relaxed);
    }
    inline constexpr bool hasChild() const {
        return children_[0].load(std::memory_order_relaxed) != children_[1].load(std::memory_order_relaxed);
    }
    inline bool update(int no, _Node*& c, _Node* n) {
        return children_[no].compare_exchange_weak(
            c, n, std::memory_order_release, std::memory_order_relaxed);
    }
};

namespace detail {

// MidpointSplitNode is used in the default non-intrusive KDTree
// implementation with MidpointSplits.  It extends the value type and
// adds the required intrusive KDTree child members.
template <typename _T, typename _Locking>
struct MidpointSplitNode : _T {
    MidpointSplitNodeMember<MidpointSplitNode<_T, _Locking>, true, _Locking> children_{};

    MidpointSplitNode(const MidpointSplitNode&) = delete;
    MidpointSplitNode(MidpointSplitNode&&) = delete;
    
    MidpointSplitNode(const _T& value) : _T(value) {}
    template <typename ... _Args>
    MidpointSplitNode(_Args&& ... args) : _T(std::forward<_Args>(args)...) {}
};

// This class is usually not require, and _GetKey could be used
// directly instead.  However, there is a chance the caller could
// provide a _GetKey that would unexpectectedly handle the derived
// class of _T that we use in the default implementation.
template <typename _T, typename _Locking, typename _GetKey>
struct MidpointSplitNodeKey : _GetKey {
    inline MidpointSplitNodeKey(const _GetKey& getKey) : _GetKey(getKey) {}

    constexpr decltype(auto) operator() (const MidpointSplitNode<_T, _Locking>& node) const {
        return _GetKey::operator()(static_cast<const _T&>(node));
    }

    constexpr decltype(auto) operator() (MidpointSplitNode<_T, _Locking>& node) const {
        return _GetKey::operator()(static_cast<_T&>(node));
    }
};

template <typename _Node, typename _Locking>
struct MidpointSplitRoot;

template <typename _Node>
struct MidpointSplitRoot<_Node, SingleThread> {
    _Node *root_ = nullptr;
    std::size_t size_ = 0;

    ~MidpointSplitRoot() {
        // TODO: this is used by the intrusive version to hold the
        // root, but the caller is responsible for allocation and thus
        // should be responsible for the delete.
        delete root_;
    }

    constexpr const _Node* get() const {
        return root_;
    }

    inline _Node* update(_Node *node) {
        _Node *root = root_;
        if (root_ == nullptr)
            root_ = node;
        return root;
    }
};

template <typename _Node>
struct MidpointSplitRoot<_Node, MultiThread> {
    std::atomic<_Node*> root_{};
    std::atomic<std::size_t> size_{};

    ~MidpointSplitRoot() {
        // TODO: this is used by the intrusive version to hold the
        // root, but the caller is responsible for allocation and thus
        // should be responsible for the delete.
        delete root_.load();
    }

    constexpr const _Node* get() const {
        return root_.load(std::memory_order_relaxed);
    }

    inline _Node* update(_Node *node) {
        _Node *root = root_.load(std::memory_order_acquire);
        while (root == nullptr)
            if (root_.compare_exchange_weak(root, node, std::memory_order_release, std::memory_order_relaxed))
                return nullptr;
        return root;
    }
};

template <typename _Locking>
struct MidpointAxisCache;

template <>
struct MidpointAxisCache<SingleThread> {
    unsigned axis_;
    MidpointAxisCache* next_;

    MidpointAxisCache(unsigned axis) : axis_(axis), next_(nullptr) {}
    ~MidpointAxisCache() {
        delete next_;
    }

    constexpr MidpointAxisCache* next() { return next_; }
    constexpr const MidpointAxisCache* next() const { return next_; }
    inline MidpointAxisCache* next(unsigned axis) {
        return next_ = new MidpointAxisCache(axis);
    }
};

template <>
struct MidpointAxisCache<MultiThread> {
    unsigned axis_;
    std::atomic<MidpointAxisCache*> next_{};

    MidpointAxisCache(unsigned axis) : axis_(axis) {}
    ~MidpointAxisCache() {
        delete next_.load();
    }

    constexpr MidpointAxisCache* next() { return next_.load(std::memory_order_acquire); }
    constexpr const MidpointAxisCache* next() const { return next_.load(std::memory_order_acquire); }
    
    MidpointAxisCache* next(unsigned axis) {
        MidpointAxisCache* next = new MidpointAxisCache(axis);
        MidpointAxisCache* prev = nullptr;
        if (next_.compare_exchange_strong(prev, next))
            return next;
        
        // other thread beat this thread to the update.
        assert(prev->axis_ == axis);
        delete next;
        return prev;
    }
};

template <
    typename _Node,
    typename _Space,
    typename _GetKey,
    bool _destructorDeletes,
    typename _Locking,
    MidpointSplitNodeMember<_Node, _destructorDeletes, _Locking> _Node::* _member>
struct KDTreeMidpointSplitIntrusiveImpl
{
    typedef _Space Space;
    typedef _Node Node;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef MidpointSplitNodeMember<Node, _destructorDeletes, _Locking> Member;

    Space space_;
    _GetKey getKey_;

    detail::MidpointSplitRoot<Node, _Locking> root_;
    MidpointAxisCache<_Locking> axisCache_;

    struct Adder {
        MidpointAddTraversal<Node, _Space> traversal_;
        MidpointAxisCache<_Locking>* axisCache_;
    
        Adder(KDTreeMidpointSplitIntrusiveImpl& tree, const Key& key)
            : traversal_(tree.space_, key),
              axisCache_(&tree.axisCache_)
        {
        }

        static constexpr _Node* child(_Node *p, int childNo) {
            return (p->*_member).child(childNo);
        }

        static constexpr bool update(Node *p, int childNo, Node*& c, Node* n) {
            return (p->*_member).update(childNo, c, n);
        }
    
        void operator() (Node* p, Node* n) {
            MidpointAxisCache<_Locking>* nextAxis = axisCache_->next();
            if (nextAxis == nullptr) {
                unsigned axis;
                traversal_.maxAxis(&axis);
                nextAxis = axisCache_->next(axis);
            }
            axisCache_ = nextAxis;
            traversal_.addImpl(*this, axisCache_->axis_, p, n);
        }
    };

    template <typename _Derived>
    struct Nearest {
        MidpointNearestTraversal<_Node, _Space> traversal_;
        const KDTreeMidpointSplitIntrusiveImpl& tree_;
        Distance dist_;
        const MidpointAxisCache<_Locking>* axisCache_;

        Nearest(
            const KDTreeMidpointSplitIntrusiveImpl& tree,
            const Key& key,
            Distance dist = std::numeric_limits<Distance>::infinity())
            : traversal_(tree.space_, key),
              tree_(tree),
              dist_(dist),
              axisCache_(&tree.axisCache_)
        {
        }

        constexpr bool shouldTraverse() const {
            return traversal_.distToRegion() <= dist_;
        }

        static constexpr const _Node* child(const _Node* n, int no) {
            return (n->*_member).child(no);
        }

        inline void update(const _Node* n) {
            Distance d = traversal_.keyDistance(tree_.getKey_(*n));
            if (d <= dist_) {
                static_cast<_Derived*>(this)->update(d, n);
            }
        }

        inline void operator() (const _Node* n) {
            if ((n->*_member).hasChild()) {
                const MidpointAxisCache<_Locking> *oldCache = axisCache_;
                axisCache_ = axisCache_->next();
                traversal_.traverse(*this, n, axisCache_->axis_);
                axisCache_ = oldCache;
            } else {
                update(n);
            }
        }
    };

    struct Nearest1 : Nearest<Nearest1> {
        const _Node *nearest_ = nullptr;
        
        using Nearest<Nearest1>::Nearest;
        using Nearest<Nearest1>::dist_;
        
        inline void update(Distance d, const _Node* n) {
            dist_ = d;
            nearest_ = n;
        }
    };

    template <typename _Value, typename _Allocator, typename _NodeValueFn>
    struct NearestK : Nearest<NearestK<_Value, _Allocator, _NodeValueFn>> {
        std::vector<std::pair<Distance, _Value>, _Allocator>& nearest_;
        std::size_t k_;
        _NodeValueFn nodeValueFn_;

        using Nearest<NearestK>::dist_;
        
        NearestK(
            const KDTreeMidpointSplitIntrusiveImpl& tree,
            std::vector<std::pair<Distance, _Value>, _Allocator>& result,
            const Key& key,
            std::size_t k,
            Distance dist,
            const _NodeValueFn& nodeValueFn)
            : Nearest<NearestK>(tree, key, dist),
              nearest_(result),
              k_(k),
              nodeValueFn_(nodeValueFn)
        {
        }

        void update(Distance d, const _Node* n) {
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

    
    KDTreeMidpointSplitIntrusiveImpl(const Space& space, const _GetKey& getKey)
        : space_(space),
          getKey_(getKey),
          axisCache_(~0)
    {
    }

    constexpr std::size_t size() const {
        return root_.size_;
    }
    
    void add(Node* node) {
        if (Node* root = root_.update(node)) {
            Adder adder(*this, getKey_(*node));
            adder(root, node);
        }
        ++root_.size_;
    }

    // TODO: support non-const return on non-const call?
    const _Node* nearest(const Key& key, Distance* distOut = nullptr) const {
        if (const Node* root = root_.get()) {
            Nearest1 nearest(*this, key);
            nearest(root);
            if (distOut)
                *distOut = nearest.dist_;
            return nearest.nearest_;
        }
        return nullptr;
    }

    template <typename _Value, typename _Allocator, typename _NodeValueFn>
    void nearest(
        std::vector<std::pair<Distance, _Value>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxDist,
        _NodeValueFn&& nodeValue) const
    {
        result.clear();
        if (k == 0)
            return;

        // std::cout << "nearest = " << &result << std::endl;
        // result.size();
        if (const Node* root = root_.get()) {
            NearestK<_Value, _Allocator, _NodeValueFn> nearest(
                *this, result, key, k, maxDist, std::forward<_NodeValueFn>(nodeValue));
            nearest(root);
            std::sort_heap(result.begin(), result.end(), CompareFirst());
        }
    }
};

} // namespace detail

template <
    typename _T,
    typename _Space,
    typename _GetKey,
    typename _Locking>
struct KDTree<_T, _Space, _GetKey, MidpointSplit, DynamicBuild, _Locking>
    : private detail::KDTreeMidpointSplitIntrusiveImpl<
         detail::MidpointSplitNode<_T, _Locking>,
         _Space,
         detail::MidpointSplitNodeKey<_T, _Locking, _GetKey>,
         true,
         _Locking,
         &detail::MidpointSplitNode<_T, _Locking>::children_>
{    
    typedef detail::KDTreeMidpointSplitIntrusiveImpl<
        detail::MidpointSplitNode<_T, _Locking>,
        _Space,
        detail::MidpointSplitNodeKey<_T, _Locking, _GetKey>,
        true,
        _Locking,
        &detail::MidpointSplitNode<_T, _Locking>::children_> Base;
    
    typedef _Space Space;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef detail::MidpointSplitNode<_T, _Locking> Node;
    
public:
    KDTree(const Space& space, const _GetKey& getKey = _GetKey())
        : Base(space, detail::MidpointSplitNodeKey<_T, _Locking, _GetKey>(getKey))
    {
    }
    
    void add(const _T& arg) {
        Base::add(new Node(arg));
    }

    using Base::size;
    
    template <typename ... _Args>
    void emplace(_Args&& ... args) {
        Base::add(new Node(std::forward<_Args>(args)...));
    }

    const _T* nearest(const Key& key, Distance* distOut = nullptr) const {
        const Node* n = Base::nearest(key, distOut);
        return n == nullptr ? nullptr : n; // ->value_;
    }

    template <typename _Allocator>
    void nearest(
        std::vector<std::pair<Distance, _T>, _Allocator>& result,
        const Key& key,
        std::size_t k,
        Distance maxDist = std::numeric_limits<Distance>::infinity()) const
    {
        Base::nearest(result, key, k, maxDist, [](const Node* n) -> const auto& { return *n; });
    }
};


}}}

#endif // UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP

