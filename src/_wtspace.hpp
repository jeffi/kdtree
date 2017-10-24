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
    inline Distance keyDistance(const _State& q) {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    inline Distance maxAxis(unsigned *axis) {
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
    inline Distance keyDistance(const _State& q) {
        return MidpointAddTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    inline Distance maxAxis(unsigned *axis) {
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
    inline Distance keyDistance(const _State& q) {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * _Ratio::num / _Ratio::den;
    }

    inline Distance maxAxis(unsigned *axis) {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * _Ratio::num / _Ratio::den;
    }
    
    inline Distance distToRegion() {
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
    Distance keyDistance(const _State& q) {
        return MidpointNearestTraversal<_Node, _Space>::keyDistance(q) * weight_;
    }

    Distance maxAxis(unsigned *axis) {
        return MidpointNearestTraversal<_Node, _Space>::maxAxis(axis) * weight_;
    }
    
    Distance distToRegion() {
        return MidpointNearestTraversal<_Node, _Space>::distToRegion() * weight_;
    }

    // template <typename _Nearest>
    // void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
    //     MidpointNearestTraversal<_Node, _Space>::traverse(nearest, n, axis);
    // }
};




}}}}

#endif // UNC_ROBOTICS_KDTREE_WTSPACE_HPP
