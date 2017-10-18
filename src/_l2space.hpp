#pragma once
#ifndef UNC_ROBOTICS_KDTREE_LSSPACE_HPP
#define UNC_ROBOTICS_KDTREE_LSSPACE_HPP

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

template <typename _Scalar, int _dimensions>
struct KDStaticAccum<L2Space<_Scalar, _dimensions>> {
    typedef L2Space<_Scalar, _dimensions> Space;

    Eigen::Array<_Scalar, _dimensions, 1> min_;
    Eigen::Array<_Scalar, _dimensions, 1> max_;
    
    unsigned dimensions_; // TODO: this is fixed when _dimensions is not Eigen::Dynamic

    inline KDStaticAccum(const Space& space)
        : dimensions_(space.dimensions())
    {
    }

    inline unsigned dimensions() {
        return dimensions_;
    }
    
    template <typename _Derived>
    void init(const Eigen::MatrixBase<_Derived>& q) {
        min_ = q;
        max_ = q;
    }

    template <typename _Derived>
    void accum(const Eigen::MatrixBase<_Derived>& q) {
        min_ = min_.min(q.array());
        max_ = max_.max(q.array());
    }

    _Scalar maxAxis(unsigned *axis) {
        return (max_ - min_).maxCoeff(axis);
    }

    template <typename _Builder, typename _Iter, typename _ToKey>
    void partition(_Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        _Iter mid = begin + (std::distance(begin, end)-1)/2;
        std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
            return toKey(a)[axis] < toKey(b)[axis];
        });
        std::swap(*begin, *mid);
        begin->split_ = toKey(*begin)[axis];
        builder(++begin, ++mid);
        builder(mid, end);
    }
};


template <typename _Scalar, int _dimensions>
struct KDStaticAccum<BoundedL2Space<_Scalar, _dimensions>>
    : KDStaticAccum<L2Space<_Scalar, _dimensions>>
{
    using KDStaticAccum<L2Space<_Scalar, _dimensions>>::KDStaticAccum;
};



template <typename _Scalar, int _dimensions>
struct KDStaticTraversal<L2Space<_Scalar, _dimensions>> {
    typedef L2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    const State& key_;

    Eigen::Array<_Scalar, _dimensions, 1> regionDeltas_;
    
    KDStaticTraversal(const Space& space, const State& key)
        : key_(key)
    {
        regionDeltas_.setZero();
    }

    inline Distance distToRegion() {
        return std::sqrt(regionDeltas_.sum());
    }

    template <typename _Derived>
    inline Distance keyDistance(const Eigen::MatrixBase<_Derived>& q) {
        return (key_ - q).norm();
    }
    
    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter min, _Iter max) {
        const auto& n = *min++;
        std::array<_Iter, 3> iters{{min, min + std::distance(min, max)/2, max}};
        Distance delta = n.split_ - key_[axis];
        int childNo = delta < 0;
        nearest(iters[childNo], iters[childNo+1]);
        nearest.update(n);
        delta *= delta;
        std::swap(regionDeltas_[axis], delta);
        if (nearest.distToRegion() <= nearest.dist())
            nearest(iters[1-childNo], iters[2-childNo]);
        regionDeltas_[axis] = delta;
    }
};


template <typename _Scalar, int _dimensions>
struct KDStaticTraversal<BoundedL2Space<_Scalar, _dimensions>>
    : KDStaticTraversal<L2Space<_Scalar, _dimensions>>
{
    using KDStaticTraversal<L2Space<_Scalar, _dimensions>>::KDStaticTraversal;
};



} // namespace unc::robotics::kdtree::detail
} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_L2SPACE_HPP
