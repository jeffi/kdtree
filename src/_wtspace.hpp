#pragma once
#ifndef UNC_ROBOTICS_KDTREE_WTSPACE_HPP
#define UNC_ROBOTICS_KDTREE_WTSPACE_HPP

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDStaticAccum<RatioWeightedSpace<_Space, _num, _den>>
    : KDStaticAccum<_Space>
{
    using KDStaticAccum<_Space>::KDStaticAccum;
};

template <typename _Space>
struct KDStaticAccum<WeightedSpace<_Space>>
    : KDStaticAccum<_Space>
{
    typename _Space::Distance weight_;

    inline KDStaticAccum(const WeightedSpace<_Space>& space)
        : KDStaticAccum<_Space>(space),
          weight_(space.weight())
    {
    }
    
    decltype(auto) maxAxis(unsigned *axis) {
        return KDStaticAccum<_Space>::maxAxis(axis) * weight_;
    }        
};


template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDStaticTraversal<RatioWeightedSpace<_Space, _num, _den>>
    : KDStaticTraversal<_Space>
{
    typedef typename _Space::Distance Distance;
    
    using KDStaticTraversal<_Space>::KDStaticTraversal;

    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return KDStaticTraversal<_Space>::keyDistance(q) * _num / _den;
    }

    inline Distance distToRegion() {
        return KDStaticTraversal<_Space>::distToRegion() * _num / _den;
    }
};

template <typename _Space>
struct KDStaticTraversal<WeightedSpace<_Space>>
    : KDStaticTraversal<_Space>
{
    typedef WeightedSpace<_Space> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Distance weight_;
    
    KDStaticTraversal(const Space& space, const State& key)
        : KDStaticTraversal<_Space>(space, key),
          weight_(space.weight())
    {
    }

    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return KDStaticTraversal<_Space>::keyDistance(q) * weight_;
    }

    inline Distance distToRegion() {
        return KDStaticTraversal<_Space>::distToRegion() * weight_;
    }
};


} // namespace unc::robotics::kdtree::detail
} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc


#endif // UNC_ROBOTICS_KDTREE_WTSPACE_HPP
