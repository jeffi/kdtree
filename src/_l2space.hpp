#pragma once
#ifndef UNC_ROBOTICS_KDTREE_L2SPACE_HPP
#define UNC_ROBOTICS_KDTREE_L2SPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Scalar, int _dimensions>
struct MidpointBoundedL2TraversalBase {
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;

    const Key& key_;
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;

    inline MidpointBoundedL2TraversalBase(const Space& space, const Key& key)
        : key_(key), bounds_(space.bounds())
    {
    }

    constexpr unsigned dimensions() const {
        return _dimensions;
    }
};

template <typename _Scalar>
struct MidpointBoundedL2TraversalBase<_Scalar, Eigen::Dynamic> {
    typedef BoundedL2Space<_Scalar, Eigen::Dynamic> Space;
    typedef typename Space::State Key;

    const Key& key_;
    Eigen::Array<_Scalar, Eigen::Dynamic, 2> bounds_;
    unsigned dimensions_;
    
    inline MidpointBoundedL2TraversalBase(const Space& space, const Key& key)
        : key_(key),
          bounds_(space.bounds()),
          dimensions_(space.dimensions())
    {
    }

    constexpr unsigned dimensions() const {
        return dimensions_;
    }
};


template <typename _Node, typename _Scalar, int _dimensions>
struct MidpointAddTraversal<_Node, BoundedL2Space<_Scalar, _dimensions>>
    : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>
{
    typedef BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State Key;
    
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::bounds_;
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::MidpointBoundedL2TraversalBase;

    constexpr _Scalar maxAxis(unsigned *axis) {
        return (this->bounds_.col(1) - this->bounds_.col(0)).maxCoeff(axis);
    }

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node *n) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;
        int childNo = (split - this->key_[axis]) < 0;
        _Node* c = _Adder::child(p, childNo);
        while (c == nullptr)
            if (_Adder::update(p, childNo, c, n))
                return;

        bounds_(axis, 1-childNo) = split;
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

    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::key_;
    using MidpointBoundedL2TraversalBase<_Scalar, _dimensions>::bounds_;

    _Scalar distToRegionCache_ = 0;
    _Scalar distToRegionSum_ = 0;
    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointBoundedL2TraversalBase<_Scalar, _dimensions>(space, key),
          regionDeltas_(space.dimensions(), 1)
    {
        regionDeltas_.setZero();
    }

    template <typename _Derived>
    constexpr _Scalar keyDistance(const Eigen::MatrixBase<_Derived>& q) {
        return (this->key_ - q).norm();
    }

    constexpr Distance distToRegion() const {
        return distToRegionCache_; // std::sqrt(regionDeltas_.sum());
    }

    template <typename _Nearest>
    inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * 0.5;
        _Scalar delta = (split - key_[axis]);
        int childNo = delta < 0;

        if (const _Node* c = _Nearest::child(n, childNo)) {
            std::swap(bounds_(axis, 1-childNo), split);
            nearest(c);
            std::swap(bounds_(axis, 1-childNo), split);            
        }

        nearest.update(n);

        if (const _Node* c = _Nearest::child(n, 1-childNo)) {
            Distance oldDelta = regionDeltas_[axis];
            Distance oldSum = distToRegionSum_;
            Distance oldDist = distToRegionCache_;
            delta *= delta;
            regionDeltas_[axis] = delta;
            distToRegionSum_ = distToRegionSum_ - oldDelta + delta;
            distToRegionCache_ = std::sqrt(distToRegionSum_);
            if (nearest.shouldTraverse()) {
                std::swap(bounds_(axis, childNo), split);
                nearest(c);
                std::swap(bounds_(axis, childNo), split);
            }
            regionDeltas_[axis] = oldDelta;
            distToRegionSum_ = oldSum;
            distToRegionCache_ = oldDist;
        }
    }
};

}}}}

#endif // UNC_ROBOTICS_KDTREE_L2SPACE_HPP
