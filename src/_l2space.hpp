#pragma once
#ifndef UNC_ROBOTICS_KDTREE_L2SPACE_HPP
#define UNC_ROBOTICS_KDTREE_L2SPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Scalar, int _dimensions>
struct MidpointBoundedL2TraversalBase {
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;
    const Key& key_;
    int dimensions_; // TODO: remove when fixed, keep when dynamic

    MidpointBoundedL2TraversalBase(const Space& space, const Key& key)
        : bounds_(space.bounds()),
          key_(key),
          dimensions_(space.dimensions())
    {
    }

    inline constexpr unsigned dimensions() {
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

template <typename _Node, typename _Scalar, int _dimensions>
struct MidpointAddTraversal<_Node, BoundedL2Space<_Scalar, _dimensions>>
    : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::MidpointBoundedL2TraversalBase;

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node *n) {
        _Scalar split = (this->bounds_(axis, 0) + this->bounds_(axis, 1)) * 0.5;
        int childNo = (split - this->key_[axis]) < 0;
        _Node* c = _Adder::child(p, childNo);
        while (c == nullptr)
            if (_Adder::update(p, childNo, c, n))
                return;

        this->bounds_(axis, 1-childNo) = split;
        adder(c, n);
    }
};

template <typename _Node, typename _Scalar, int _dimensions>
struct MidpointNearestTraversal<_Node, BoundedL2Space<_Scalar, _dimensions>>
    : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>(space, key)
    {
        regionDeltas_.setZero();
    }

    inline Distance distToRegion() {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Nearest>
    void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        _Scalar split = (this->bounds_(axis, 0) + this->bounds_(axis, 1)) * 0.5;
        _Scalar delta = (split - this->key_[axis]);
        int childNo = delta < 0;

        if (const _Node* c = _Nearest::child(n, childNo)) {
            std::swap(this->bounds_(axis, 1-childNo), split);
            nearest(c);
            std::swap(this->bounds_(axis, 1-childNo), split);            
        }

        nearest.update(n);

        if (const _Node* c = _Nearest::child(n, 1-childNo)) {
            Distance oldDelta = regionDeltas_[axis];
            regionDeltas_[axis] = delta*delta;
            if (nearest.shouldTraverse()) {
                std::swap(this->bounds_(axis, childNo), split);
                nearest(c);
                std::swap(this->bounds_(axis, childNo), split);
            }
            regionDeltas_[axis] = oldDelta;
        }
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_L2SPACE_HPP
