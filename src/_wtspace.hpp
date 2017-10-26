#pragma once
#ifndef UNC_ROBOTICS_KDTREE_WTSPACE_HPP
#define UNC_ROBOTICS_KDTREE_WTSPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Node, typename _Space, typename _Ratio>
struct MidpointAddTraversal<_Node, RatioWeightedSpace<_Space, _Ratio>>
    : MidpointAddTraversal<_Node, _Space>
{
    typedef RatioWeightedSpace<_Space, _Ratio> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    // inherit constructor
    using MidpointAddTraversal<_Node, _Space>::MidpointAddTraversal;

    template <typename _State>
    inline Distance keyDistance(const _State& q) const {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    inline Distance maxAxis(unsigned *axis) const {
        return MidpointAddTraversal<_Node, _Space>::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Node, typename _Space>
struct MidpointAddTraversal<_Node, WeightedSpace<_Space>>
    : MidpointAddTraversal<_Node, _Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;

    MidpointAddTraversal(const Space& space, const State& key)
        : MidpointAddTraversal<_Node, _Space>(space, key),
          weight_(space.weight())
    {
    }

    template <typename _State>
    inline Distance keyDistance(const _State& q) const {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    inline Distance maxAxis(unsigned *axis) const {
        return MidpointAddTraversal<_Node, _Space>::maxAxis(axis) * weight_;
    }
};

template <typename _Node, typename _Space, typename _Ratio>
struct MidpointNearestTraversal<_Node, RatioWeightedSpace<_Space, _Ratio>>
    : MidpointNearestTraversal<_Node, _Space>
{
    typedef RatioWeightedSpace<_Space, _Ratio> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    // inherit constructor
    using MidpointNearestTraversal<_Node, _Space>::MidpointNearestTraversal;

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with MidpointAddTraversal.  Would be nice to merge them.
    template <typename _State>
    inline Distance keyDistance(const _State& q) const {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    inline Distance maxAxis(unsigned *axis) const {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
    
    inline Distance distToRegion() const {
        return MidpointNearestTraversal<_Node, _Space>::distToRegion() * _Ratio::num / _Ratio::den;
    }

    // template <typename _Nearest>
    // inline void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
    //     MidpointNearestTraversal<_Node, _Space>::traverse(nearest, n, axis);
    // }
};

template <typename _Node, typename _Space>
struct MidpointNearestTraversal<_Node, WeightedSpace<_Space>>
    : MidpointNearestTraversal<_Node, _Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;
    
    MidpointNearestTraversal(const Space& space, const State& key)
        : MidpointNearestTraversal<_Node, _Space>(space, key),
          weight_(space.weight())
    {
    }

    // TODO: keyDistance and maxAxis implementations are duplicated
    // with MidpointAddTraversal.  Would be nice to merge them.
    template <typename _State>
    Distance keyDistance(const _State& q) const {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    Distance maxAxis(unsigned *axis) const {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * weight_;
    }
    
    Distance distToRegion() const {
        return MidpointNearestTraversal<_Node, _Space>::distToRegion() * weight_;
    }

    // template <typename _Nearest>
    // void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
    //     MidpointNearestTraversal<_Node, _Space>::traverse(nearest, n, axis);
    // }
};

template <typename _Space, typename _Ratio>
struct MedianAccum<RatioWeightedSpace<_Space, _Ratio>>
    : MedianAccum<_Space>
{
    typedef MedianAccum<_Space> Base;
    typedef typename _Space::Distance Distance;
    
    using Base::Base;
    
    Distance maxAxis(unsigned *axis) const {
        return Base::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Space, typename _Ratio>
struct MedianNearestTraversal<RatioWeightedSpace<_Space, _Ratio>>
    : MedianNearestTraversal<_Space>
{
    typedef MedianNearestTraversal<_Space> Base;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;

    using Base::Base;
    
    Distance distToRegion() const {
        return Base::distToRegion() * _Ratio::num / _Ratio::den;
    }

    template <typename _Key>
    Distance keyDistance(const _Key& key) const {
        return Base::keyDistance(key) * _Ratio::num / _Ratio::den;
    }
};

template <typename _Space>
struct MedianAccum<WeightedSpace<_Space>>
    : MedianAccum<_Space>
{
    typedef MedianAccum<_Space> Base;
    typedef typename _Space::Distance Distance;

    Distance weight_;
    
    MedianAccum(const WeightedSpace<_Space>& space)
        : Base(space),
          weight_(space.weight())
    {
    }
    
    Distance maxAxis(unsigned *axis) const {
        return Base::maxAxis(axis) * weight_;
    }
};

template <typename _Space>
struct MedianNearestTraversal<WeightedSpace<_Space>>
    : MedianNearestTraversal<_Space>
{
    typedef MedianNearestTraversal<_Space> Base;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;

    Distance weight_;
    
    MedianNearestTraversal(
        const WeightedSpace<_Space>& space,
        const Key& key)
        : Base(space, key),
          weight_(space.weight())
    {
    }

    Distance distToRegion() const {
        return Base::distToRegion() * weight_;
    }

    template <typename _Key>
    Distance keyDistance(const _Key& key) const {
        return Base::keyDistance(key) * weight_;
    }
};
    


}}}}

#endif // UNC_ROBOTICS_KDTREE_WTSPACE_HPP
