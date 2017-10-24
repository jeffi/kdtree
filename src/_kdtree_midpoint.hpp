#pragma once
#ifndef UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP
#define UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP

#include "_spaces.hpp"

#include <array>

namespace unc { namespace robotics { namespace kdtree {

// When using the intrusive version of the KDTree, the caller must
// provide a member node.
template <typename _Node, bool _destructorDeletes, bool _lockfree>
struct MidpointSplitNodeMember;

template <typename _Node, bool _destructorDeletes>
struct MidpointSplitNodeMember<_Node, _destructorDeletes, false> {
    static constexpr bool lockfree = false;
        
    std::array<_Node*, 2> children_{};
    
    ~MidpointSplitNodeMember() {
        if (_destructorDeletes) {
            delete children_[0];
            delete children_[1];
        }
    }

    inline _Node* child(int no) { return children_[no]; }
    inline const _Node* child(int no) const { return children_[no]; }
    inline bool hasChild() const { return children_[0] != children_[1]; }
    inline bool update(int no, _Node*, _Node* n) {
        children_[no] = n;
        return true;
    }
};

template <typename _Node, bool _destructorDeletes>
struct MidpointSplitNodeMember<_Node, _destructorDeletes, true> {
    static constexpr bool lockfree = true;
    
    std::array<std::atomic<_Node*>, 2> children_{};
    
    ~MidpointSplitNodeMember() {
        if (_destructorDeletes) {
            delete children_[0].load(std::memory_order_acquire);
            delete children_[1].load(std::memory_order_acquire);
        }
    }

    inline _Node* child(int no) { return children_[no].load(std::memory_order_acquire); }
    inline const _Node* child(int no) const {
        return children_[no].load(std::memory_order_relaxed);
    }
    inline bool hasChild() const {
        return children_[0].load(std::memory_order_relaxed) != children_[1].load(std::memory_order_relaxed);
    }
    inline bool update(int no, _Node*& c, _Node* n) {
        return children_[no].compare_exchange_weak(
            c, n, std::memory_order_release, std::memory_order_relaxed);
    }
};

namespace detail {

struct CompareFirst {
    template <typename _First, typename _Second>
    inline bool operator() (const std::pair<_First,_Second>& a, const std::pair<_First,_Second>& b) {
        return a.first < b.first;
    }
};

template <typename _T, bool _lockfree>
struct MidpointSplitNode {
    _T value_;
    MidpointSplitNodeMember<MidpointSplitNode<_T, _lockfree>, true, _lockfree> children_;

    MidpointSplitNode(const MidpointSplitNode&) = delete;
    MidpointSplitNode(MidpointSplitNode&&) = delete;
    
    MidpointSplitNode(const _T& value)
        : value_(value)
    {
    }
    
    template <typename ... _Args>
    MidpointSplitNode(_Args&& ... args)
        : value_(std::forward<_Args>(args)...)
    {
    }
};

template <typename _T, bool _lockfree, typename _GetKey>
struct MidpointSplitNodeKey : _GetKey {
    inline MidpointSplitNodeKey(const _GetKey& getKey) : _GetKey(getKey) {}

    constexpr decltype(auto) operator() (const MidpointSplitNode<_T, _lockfree>& node) const {
        return _GetKey::operator()(node.value_);
    }

    constexpr decltype(auto) operator() (MidpointSplitNode<_T, _lockfree>& node) const {
        return _GetKey::operator()(node.value_);
    }
};


template <typename _Node, bool _lockfree>
struct MidpointSplitRoot;

template <typename _Node>
struct MidpointSplitRoot<_Node, false> {
    _Node *root_ = nullptr;
    std::size_t size_ = 0;

    ~MidpointSplitRoot() {
        // TODO: this is used by the intrusive version to hold the
        // root, but the caller is responsible for allocation and thus
        // should be responsible for the delete.
        delete root_;
    }

    inline const _Node* get() const {
        return root_;
    }

    _Node* update(_Node *node) {
        _Node *root = root_;
        if (root_ == nullptr)
            root_ = node;
        return root;
    }
};

template <typename _Node>
struct MidpointSplitRoot<_Node, true> {
    std::atomic<_Node*> root_{};
    std::atomic<std::size_t> size_{};

    ~MidpointSplitRoot() {
        // TODO: this is used by the intrusive version to hold the
        // root, but the caller is responsible for allocation and thus
        // should be responsible for the delete.
        delete root_.load();
    }

    inline const _Node* get() const {
        return root_.load(std::memory_order_relaxed);
    }

    _Node* update(_Node *node) {
        _Node *root = root_.load(std::memory_order_acquire);
        while (root == nullptr)
            if (root_.compare_exchange_weak(root, node, std::memory_order_release, std::memory_order_relaxed))
                return nullptr;
        return root;
    }
};

template <bool _lockfree>
struct MidpointAxisCache;

template <>
struct MidpointAxisCache<false> {
    unsigned axis_;
    MidpointAxisCache* next_;

    MidpointAxisCache(unsigned axis) : axis_(axis), next_(nullptr) {}
    ~MidpointAxisCache() {
        delete next_;
    }

    MidpointAxisCache* next() { return next_; }
    const MidpointAxisCache* next() const { return next_; }
    MidpointAxisCache* next(unsigned axis) {
        return next_ = new MidpointAxisCache(axis);
    }
};

template <>
struct MidpointAxisCache<true> {
    unsigned axis_;
    std::atomic<MidpointAxisCache*> next_{};

    MidpointAxisCache(unsigned axis) : axis_(axis) {}
    ~MidpointAxisCache() {
        delete next_.load();
    }

    MidpointAxisCache* next() { return next_.load(std::memory_order_acquire); }
    const MidpointAxisCache* next() const { return next_.load(std::memory_order_acquire); }
    
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
    bool _lockfree,
    MidpointSplitNodeMember<_Node, _destructorDeletes, _lockfree> _Node::* _member>
struct KDTreeMidpointSplitIntrusiveImpl
{
    typedef _Space Space;
    typedef _Node Node;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef MidpointSplitNodeMember<Node, _destructorDeletes, _lockfree> Member;

    Space space_;
    _GetKey getKey_;

    detail::MidpointSplitRoot<Node, _lockfree> root_;
    MidpointAxisCache<_lockfree> axisCache_;

    struct Adder {
        MidpointAddTraversal<Node, _Space> traversal_;
        MidpointAxisCache<_lockfree>* axisCache_;
    
        Adder(KDTreeMidpointSplitIntrusiveImpl& tree, const Key& key)
            : traversal_(tree.space_, key),
              axisCache_(&tree.axisCache_)
        {
        }

        static inline _Node* child(_Node *p, int childNo) {
            return (p->*_member).child(childNo);
        }

        static inline bool update(Node *p, int childNo, Node*& c, Node* n) {
            return (p->*_member).update(childNo, c, n);
        }
    
        void operator() (Node* p, Node* n) {
            MidpointAxisCache<_lockfree>* nextAxis = axisCache_->next();
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
        const KDTreeMidpointSplitIntrusiveImpl& tree_;
        MidpointNearestTraversal<_Node, _Space> traversal_;
        const MidpointAxisCache<_lockfree>* axisCache_;
        Distance dist_;

        Nearest(
            const KDTreeMidpointSplitIntrusiveImpl& tree,
            const Key& key,
            Distance dist = std::numeric_limits<Distance>::infinity())
            : tree_(tree),
              traversal_(tree.space_, key),
              axisCache_(&tree.axisCache_),
              dist_(dist)
        {
        }

        inline bool shouldTraverse() {
            return traversal_.distToRegion() <= dist_;
        }

        static inline const _Node* child(const _Node* n, int no) {
            return (n->*_member).child(no);
        }

        void update(const _Node* n) {
            const auto& q = tree_.getKey_(*n);
            Distance d = traversal_.keyDistance(q);
            if (d <= dist_) {
                static_cast<_Derived*>(this)->update(d, n);
            }
        }

        void operator() (const _Node* n) {
            update(n);
            
            const Member& m = n->*_member;
            if (m.hasChild()) {
                const MidpointAxisCache<_lockfree> *oldCache = axisCache_;
                axisCache_ = axisCache_->next();
                traversal_.traverse(*this, n, axisCache_->axis_);
                axisCache_ = oldCache;
            }
        }
    };

    struct Nearest1 : Nearest<Nearest1> {
        const _Node *nearest_ = nullptr;
        
        using Nearest<Nearest1>::Nearest;
        
        void update(Distance d, const _Node* n) {
            this->dist_ = d;
            nearest_ = n;
        }
    };

    template <typename _Value, typename _NodeValueFn>
    struct NearestK : Nearest<NearestK<_Value, _NodeValueFn>> {
        _NodeValueFn nodeValueFn_;
        std::vector<std::pair<Distance, _Value>>& nearest_;
        std::size_t k_;

        NearestK(
            const KDTreeMidpointSplitIntrusiveImpl& tree,
            std::vector<std::pair<Distance, _Value>>& result,
            const Key& key,
            std::size_t k,
            Distance dist,
            const _NodeValueFn& nodeValueFn)
            : Nearest<NearestK>(tree, key, dist),
              nodeValueFn_(nodeValueFn),
              nearest_(result),
              k_(k)
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
                this->dist_ = nearest_[0].first;
        }        
    };

    
    KDTreeMidpointSplitIntrusiveImpl(const Space& space, const _GetKey& getKey)
        : space_(space),
          getKey_(getKey),
          axisCache_(~0)
    {
    }

    std::size_t size() const {
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

    template <typename _Value, typename _NodeValueFn>
    void nearest(
        std::vector<std::pair<Distance, _Value>>& result,
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
            NearestK<_Value, _NodeValueFn> nearest(*this, result, key, k, maxDist, std::forward<_NodeValueFn>(nodeValue));
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
    bool _lockfree>
struct KDTree<_T, _Space, _GetKey, ::unc::robotics::kdtree::MidpointSplit, true, _lockfree>
    : private detail::KDTreeMidpointSplitIntrusiveImpl<
         detail::MidpointSplitNode<_T, _lockfree>,
         _Space,
         detail::MidpointSplitNodeKey<_T, _lockfree, _GetKey>,
         true,
         _lockfree,
         &detail::MidpointSplitNode<_T, _lockfree>::children_>
{    
    typedef detail::KDTreeMidpointSplitIntrusiveImpl<
        detail::MidpointSplitNode<_T, _lockfree>,
        _Space,
        detail::MidpointSplitNodeKey<_T, _lockfree, _GetKey>,
        true,
        _lockfree,
        &detail::MidpointSplitNode<_T, _lockfree>::children_> Base;
    
    typedef _Space Space;
    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    typedef detail::MidpointSplitNode<_T, _lockfree> Node;
    
public:
    KDTree(const Space& space, const _GetKey& getKey = _GetKey())
        : Base(space, detail::MidpointSplitNodeKey<_T, _lockfree, _GetKey>(getKey))
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
        return n == nullptr ? nullptr : &n->value_;
    }

    void nearest(
        std::vector<std::pair<Distance, _T>>& result,
        const Key& key,
        std::size_t k,
        Distance maxDist = std::numeric_limits<Distance>::infinity()) const
    {
        Base::nearest(result, key, k, maxDist, [](const Node* n) { return n->value_; });
    }
};


}}}

#endif // UNC_ROBOTICS_KDTREE_KDTREE_MIDPOINT_HPP
