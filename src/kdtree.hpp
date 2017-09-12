#pragma once
#ifndef UNC_KDTREE_HPP
#define UNC_KDTREE_HPP

#include <ostream>
#include <array>
#include <atomic>
#include <Eigen/Dense>

namespace unc {
namespace robotics {
namespace kdtree {

// TODO: change `_rows` to `_dimensions` everywhere

template <typename _Scalar, int _rows>
class EuclideanSpace {
public:
    typedef _Scalar Distance;
    typedef Eigen::Matrix<_Scalar, _rows, 1> State;
    static constexpr int axes = _rows;

    inline Distance distance(const State& a, const State& b) const {
        return (a - b).norm();
    }
};

template <typename _Scalar, int _rows>
class BoundedEuclideanSpace : public EuclideanSpace<_Scalar, _rows> {
    Eigen::Array<_Scalar, _rows, 2> bounds_;

public:
    BoundedEuclideanSpace(const Eigen::Array<_Scalar, _rows, 2>& bounds)
        : bounds_(bounds)
    {
    }

    BoundedEuclideanSpace(
        const Eigen::Array<_Scalar, _rows, 1>& min,
        const Eigen::Array<_Scalar, _rows, 1>& max)
    {
        bounds_.col(0) = min;
        bounds_.col(1) = max;
    }

    inline const Eigen::Array<_Scalar, _rows, 2>& bounds() const {
        return bounds_;
    }

    inline _Scalar bounds(int r, int c) const {
        return bounds_(r, c);
    }
};

template <typename _Scalar>
class SO3Space {
public:
    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;
    static constexpr int axes = 3;

    inline Distance distance(const State& a, const State& b) const {
        return std::acos(std::abs(a.coeffs().matrix().dot(b.coeffs().matrix())));
    }
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
class RatioWeightedSpace : public _Space {
public:
    // inherit constructor
    using _Space::_Space;
    
    RatioWeightedSpace(const _Space& space)
        : _Space(space)
    {
    }

    inline typename _Space::Distance distance(
        const typename _Space::State& a,
        const typename _Space::State& b) const {
        return _Space::distance(a, b) * _num / _den;
    }
};

template <typename ... _States>
class CompoundState {
    std::tuple<_States...> states_;
    
public:
    template <typename ... _Args>
    CompoundState(_Args&& ... args)
        : states_(_States(std::forward<_Args>(args))...)
    {
    }

    template <std::size_t _index>
    const typename std::tuple_element<_index, std::tuple<_States...>>::type& substate() const {
        return std::get<_index>(states_);
    }
};

namespace detail {
template <typename ... _Scalars>
struct ScalarResult {};
template <typename _Scalar>
struct ScalarResult<_Scalar> { typedef _Scalar type; };
template <typename ... _Scalars>
struct ScalarResult<long double, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<long double, double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<long double, float, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, double, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, float, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, double, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, float, _Scalars...> : ScalarResult<float, _Scalars...> {};

template <int ... values>
struct sum { static constexpr int value = 0; };
template <int first, int ... rest>
struct sum<first, rest...> { static constexpr int value = first + sum<rest...>::value; };

template <int _index, typename ... _Spaces>
struct CompoundDistance {
    typedef CompoundState<typename _Spaces::State...> State;
    typedef typename detail::ScalarResult<typename _Spaces::Distance...>::type Distance;

    inline static Distance accum(const std::tuple<_Spaces...>& spaces, const State& a, const State& b, Distance x) {
        return CompoundDistance<_index + 1, _Spaces...>::accum(
            spaces,
            a, b,
            x + std::get<_index>(spaces).distance(
                a.template substate<_index>(),
                b.template substate<_index>()));
    }
};
// base case, x contains accumulated distance
template <typename ... _Spaces>
struct CompoundDistance<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundState<typename _Spaces::State...> State;
    typedef typename detail::ScalarResult<typename _Spaces::Distance...>::type Distance;

    inline static Distance accum(const std::tuple<_Spaces...>& spaces, const State& a, const State& b, Distance x) {
        return x;
    }
};

} // namespace detail

template <typename ... _Spaces>
class CompoundSpace {
    std::tuple<_Spaces...> spaces_;

public:
    typedef CompoundState<typename _Spaces::State...> State;

    static constexpr std::size_t size = sizeof...(_Spaces);
    static constexpr int axes = detail::sum<_Spaces::axes...>::value;

    typedef typename detail::ScalarResult<typename _Spaces::Distance...>::type Distance;

    template <typename ... _Args>
    CompoundSpace(_Args&& ... args)
        : spaces_(std::forward<_Args>(args)...)
    {
    }

    template <std::size_t _index>
    const typename std::tuple_element<_index, std::tuple<_Spaces...>>::type& subspace() const {
        return std::get<_index>(spaces_);
    }

    Distance distance(const State& a, const State& b) const {
        return detail::CompoundDistance<1, _Spaces...>::accum(
            spaces_,
            a, b,
            std::get<0>(spaces_).distance(
                a.template substate<0>(),
                b.template substate<0>()));
    }
};

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using SE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<EuclideanSpace<_Scalar, 3>, _tWeight, 1>>;

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using BoundedSE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<BoundedEuclideanSpace<_Scalar, 3>, _tWeight, 1>>;

namespace detail {

template <typename _T>
struct KDNode {
    _T value_;
    std::array<KDNode*, 2> children_;
    KDNode(const _T& v) : value_(v), children_{{nullptr, nullptr}} {}
    ~KDNode() {
        for (int i=0 ; i<2 ; ++i)
            if (children_[i])
                delete children_[i];
    }

    template <typename _Fn>
    void visit(_Fn& fn, unsigned depth) const {
        fn(value_, depth++);
        for (int i=0 ; i<2 ; ++i)
            if (children_[i])
                children_[i]->visit(fn, depth);
    }
};

template <typename _Space>
struct KDAdder;

template <typename _Space>
struct KDWalker;

template <typename _Space>
struct KDAdderBase {
    typedef KDAdder<_Space> Derived;
    
    template <typename _T>
    unsigned operator() (KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        int axis;
        typename _Space::Distance d = static_cast<Derived*>(this)->maxAxis(&axis);
        return static_cast<Derived*>(this)->doAdd(*this, axis, d, p, n, depth);
    }
};

template <typename _Scalar, int _rows>
struct KDAdder<BoundedEuclideanSpace<_Scalar, _rows>> : KDAdderBase<BoundedEuclideanSpace<_Scalar, _rows>> {
    typedef BoundedEuclideanSpace<_Scalar, _rows> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    const State& key_;
    Eigen::Array<_Scalar, _rows, 2> bounds_;
    
    KDAdder(const State& key, const Space& space)
        : key_(key),
          bounds_(space.bounds())
    {
    }

    Distance maxAxis(int* axis) {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
    }

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T> *c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return adder(c, n, depth+1);
        } else {
            p->children_[childNo] = n;
            return depth;
        }
    }
};

template <typename _Scalar, int _rows>
struct KDWalker<BoundedEuclideanSpace<_Scalar, _rows>> : KDAdder<BoundedEuclideanSpace<_Scalar, _rows>> {
    typedef KDAdder<BoundedEuclideanSpace<_Scalar, _rows>> Base;
    typedef BoundedEuclideanSpace<_Scalar, _rows> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    Eigen::Array<_Scalar, _rows, 1> deltas_;
    
    KDWalker(const State& key, const Space& space)
        : Base(key, space)
    {
        deltas_.setZero();
    }

    using Base::key_;
    using Base::bounds_;

    template <typename _Nearest, typename _T, typename _MinDist>
    void traverse(_Nearest& t, const KDNode<_T> *n, int axis, Distance dist, _MinDist minDist, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
        _Scalar delta = (split - key_[axis]);
        int childNo = delta < 0;

        if (const KDNode<_T>* c = n->children_[childNo]) {
            std::swap(bounds_(axis, 1-childNo), split);
            t.traverse(c, minDist, depth);
            std::swap(bounds_(axis, 1-childNo), split);
        }

        t.update(n);

        if (const KDNode<_T>* c = n->children_[1-childNo]) {
            _Scalar newDelta = delta*delta;
            _Scalar oldDelta = deltas_[axis];
            minDist = minDist - oldDelta + newDelta;
            if (minDist <= t.minDistSquared()) {
                std::swap(bounds_(axis, childNo), split);
                deltas_[axis] = newDelta;
                t.traverse(c, minDist, depth);
                deltas_[axis] = oldDelta;
                bounds_(axis, childNo) = split;
            }
        }
    }
};

template <typename _Scalar>
int volumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    int index;
    q.coeffs().array().abs().maxCoeff(&index);
    return index;
}

// Compute a midpoint between the bounds along an SO(3) quaternion
// axis.
template <typename _DerivedMin, typename _DerivedMax>
Eigen::Array<typename _DerivedMin::Scalar, 2, 1> axisMidPoint(
    const Eigen::ArrayBase<_DerivedMin>& min,
    const Eigen::ArrayBase<_DerivedMax>& max,
    unsigned depth)
{
    // TODO: the computation of s0 can be replaced with a lookup table
    // based upon depth.
    typedef typename _DerivedMin::Scalar Scalar;
    Scalar dq = std::abs(min.matrix().dot(max.matrix()));
    // Scalar theta = acos(dq);
    // Scalar d = (1 / sin(theta));
    // Scalar s0 = sin(theta / 2) * d;
    Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));

    // std::cout << "depth=" << depth << ", s0=" << std::setprecision(15) << s0 << std::endl;
    return (min + max) * s0;
}

template <typename _Scalar>
struct KDAdder<SO3Space<_Scalar>> : KDAdderBase<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    int vol_;
    int depth_;
    State key_;
    
    std::array<Eigen::Array<Scalar, 2, 3>, 2> bounds_;

    KDAdder(const State& key, const Space& space)
        : vol_(volumeIndex(key)),
          depth_(1),
          key_(key.coeffs()[vol_] < 0 ? -key.coeffs() : key.coeffs())
    {
        static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
        bounds_[0] = rt;
        bounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
    }

    Distance maxAxis(int* axis) {
        *axis = depth_ % 3;
#if 0
        // Computes the distance between the min and max bounds.  This
        // follows the standard distance function of acos(dot(x,y)).
        return std::acos(
            std::abs(bounds_[0].col(*axis).matrix().dot(
                         bounds_[1].col(*axis).matrix())));
#else
        // Since we're doing midpoint splits, the distance along an
        // axis halves every time it revisits an axis.  We can save on
        // the costly acos here.
#if 1
        return (depth_ <= 3) ? M_PI_2 :  M_PI/(1 << (depth_ / 3));
#else
        // ldexp does exactly the above, but it is slower in
        // benchmarks than the above.
        return (depth_ <= 3) ? M_PI_2 : std::ldexp(static_cast<Distance>(M_PI), -(depth_/3));
#endif
#endif
    }

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        int childNo;
        switch (++depth_) {
        case 2:
            if (KDNode<_T>* c = p->children_[childNo = vol_ < 2])
                return adder(c, n, depth+1); // tail recur
            break;
        case 3:
            if (KDNode<_T>* c = p->children_[childNo = vol_ & 1])
                return adder(c, n, depth+1); // tail recur
            break;
        default:
#if 0
            Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (std::cos(d) + 1));
#else
            Scalar dq = std::abs(bounds_[0].col(axis).matrix().dot(
                                     bounds_[1].col(axis).matrix()));
            Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));
#endif

            Eigen::Matrix<Scalar, 2, 1> mp =
                (bounds_[0].col(axis) + bounds_[1].col(axis)) * s0;
            Scalar dot = mp[0]*key_.coeffs()[vol_] + mp[1]*key_.coeffs()[(vol_ + axis + 1) % 4];

            // std::cout << "SO3::doAdd " << vol_ << ", " << depth << ", " << depth_
            //           << ", " << d << ", " << s0 << ", " << axis << ", " << dot << std::endl;
            
            if (KDNode<_T>* c = p->children_[childNo = (dot > 0)]) {
                bounds_[1-childNo].col(axis) = mp;
                return adder(c, n, depth+1); // tail recur
            }
            break;
        }

        assert(p->children_[childNo] == nullptr);
        p->children_[childNo] = n;
        return depth;        
    }
};

template <typename _Scalar>
inline bool asinXlessThanY(_Scalar x, _Scalar y) {
    constexpr _Scalar PI0_5 = 1.5707963267948966192313216916397514420985846996875529L;
    constexpr _Scalar ASIN_BOUNDED_MAX = 0.33067408756426286;
        
    return x <= y // sufficient
        || (x * PI0_5 - ASIN_BOUNDED_MAX <= y // necessary
            && std::asin(x) <= y); // exact
}

template <typename _Scalar>
struct KDWalker<SO3Space<_Scalar>> : KDAdder<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef KDAdder<Space> Base;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    int tVol_;

    using Base::bounds_;
    using Base::depth_;
    using Base::key_;
    using Base::vol_;
    
    KDWalker(const State& key, const Space& space)
        : Base(key, space)
    {
    }

    template <typename _Nearest, typename _T, typename _MinDist>
    void traverse(_Nearest& t, const KDNode<_T>* n, int axis, Distance dist, _MinDist minDist, unsigned depth) {
        int childNo;
        assert(vol_ == volumeIndex(key_));
        switch (++depth_) {
        case 2:
            if (KDNode<_T>* c = n->children_[childNo = vol_ < 2]) {
                tVol_ = vol_ & 2;
                t.traverse(c, minDist, depth);
            }
            t.update(n);
            if (KDNode<_T>* c = n->children_[1-childNo]) {
                tVol_ = (vol_ & 2) ^ 2;
                t.traverse(c, minDist, depth);
            }
            break;

        case 3:
            if (KDNode<_T>* c = n->children_[childNo = vol_ & 1]) {
                tVol_ = (tVol_ & 2) | (vol_ & 1);
                if (key_.coeffs()[tVol_] < 0)
                    key_.coeffs() = -key_.coeffs();
                t.traverse(c, minDist, depth);
            }
            t.update(n);
            if (KDNode<_T>* c = n->children_[1-childNo]) {
                tVol_ = (tVol_ & 2) | ((vol_ & 1) ^ 1);
                if (key_.coeffs()[tVol_] < 0)
                    key_.coeffs() = -key_.coeffs();
                t.traverse(c, minDist, depth);
            }
            break;

        default:
            Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (std::cos(dist) + 1));
            Eigen::Matrix<Scalar, 2, 1> mp =
                (bounds_[0].col(axis) + bounds_[1].col(axis)) * s0;
            Scalar dot = mp[0]*key_.coeffs()[tVol_] + mp[1]*key_.coeffs()[(tVol_ + axis + 1) % 4];

            if (KDNode<_T>* c = n->children_[childNo = (dot > 0)]) {
                Eigen::Matrix<Scalar, 2, 1> tmp(bounds_[1-childNo].col(axis));
                bounds_[1-childNo].col(axis) = mp;
                t.traverse(c, minDist, depth);
                bounds_[1-childNo].col(axis) = tmp;
            }

            t.update(n);

            if (KDNode<_T>* c = n->children_[1-childNo]) {
                Scalar df =
                    bounds_[childNo](0, axis) * key_.coeffs()[tVol_] +
                    bounds_[childNo](1, axis) * key_.coeffs()[(tVol_ + axis + 1) % 4];
                df = std::min(std::abs(dot), std::abs(df));
                //if (asinXlessThanY(df, t.minDist())) {
                if (std::asin(df) < t.minDist()) {
                    Eigen::Matrix<Scalar, 2, 1> tmp(bounds_[childNo].col(axis));
                    bounds_[childNo].col(axis) = mp;
                    t.traverse(c, minDist, depth);
                    bounds_[childNo].col(axis) = tmp;
                }
            }
        }
        --depth_;
    }
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDAdder<RatioWeightedSpace<_Space, _num, _den>> : KDAdder<_Space> {
    typedef typename _Space::State State;

    // inherit constructor
    using KDAdder<_Space>::KDAdder;

    typename _Space::Distance maxAxis(int *axis) {
        return KDAdder<_Space>::maxAxis(axis) * _num / _den;
    }
};

template <typename _Nearest, std::intmax_t _num, std::intmax_t _den>
struct RatioWeightedNearest {
    typedef typename _Nearest::Distance Distance;

    _Nearest& nearest_;
    
    RatioWeightedNearest(_Nearest& nearest)
        : nearest_(nearest)
    {
    }

    Distance minDistSquared() const {
        // called by the contained (unweighted) space, the traversal
        // uses the weighted metric so the return value needs to be
        // unweighted.
        return nearest_.minDistSquared() * (_den * _den) / (_num * _num);
    }

    Distance minDist() const {
        // See comment in minDistSquared.
        return nearest_.minDist() * _den / _num;
    }

    template <typename _T>
    void traverse(const KDNode<_T>* n, Distance minDist, unsigned depth) {
        // The unweighted space calls this traversal method to
        // traverse to a child node.  The `minDist` value thus must be
        // weighted again.
        nearest_.traverse(n, minDist * _num / _den, depth);
    }

    template <typename _T>
    void update(const KDNode<_T>* n) {
        nearest_.update(n);
    }
};


template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDWalker<RatioWeightedSpace<_Space, _num, _den>>
    : KDWalker<_Space>
{
    typedef typename _Space::State State;
    typedef typename _Space::Distance Distance;

    // inherit constructor
    using KDWalker<_Space>::KDWalker;

    typename _Space::Distance maxAxis(int *axis) {
        return KDWalker<_Space>::maxAxis(axis) * _num / _den;
    }

    template <typename _Nearest, typename _T, typename _MinDist>
    void traverse(_Nearest& t, const KDNode<_T>* n, int axis, Distance dist, _MinDist minDist, unsigned depth) {
        // from a the super-space's perpective, all the distances
        // returned by this space are scaled by _num/_den.  Thus,
        // `dist` and `minDist` need to be scaled by _den/_num to
        // present a consistent distance to the contained space.
        // Similarly, the callbacks to the `_Nearest` object need to
        // be scaled.
        RatioWeightedNearest<_Nearest, _num, _den> wrap(t);
        KDWalker<_Space>::traverse(
            wrap,
            n, axis,
            dist * _den / _num,
            minDist * _den / _num,
            depth);
    }
};

// Helper template for computing the max axis in a compound space it
// tail recurses on incremented space indexes, tracking the best
// distance and axis as it goes.  Recursion ends once all subspaces
// are checked, and it returns the best distance found.  The axis is
// computed as the number of axes before the space + the axis within
// the subspace.
template <int _index, template <typename> class _KDAdder, typename ... _Spaces>
struct CompoundMaxAxis {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    
    static Distance maxAxis(std::tuple<_KDAdder<_Spaces>...>& adders, Distance bestDist, int *bestAxis, int axesBefore) {
        typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;
        int axis;
        Distance dist = std::get<_index>(adders).maxAxis(&axis);
        if (dist > bestDist) {
            *bestAxis = axesBefore + axis;
            bestDist = dist;
        }
        return CompoundMaxAxis<_index+1, _KDAdder, _Spaces...>::maxAxis(
            adders, bestDist, bestAxis, axesBefore + Subspace::axes);
    }
};
// Base case
template <template <typename> class _KDAdder, typename ... _Spaces>
struct CompoundMaxAxis<sizeof...(_Spaces), _KDAdder, _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;

    static Distance maxAxis(std::tuple<_KDAdder<_Spaces>...>& adders, Distance bestDist, int *bestAxis, int axesBefore) {
        return bestDist;
    }
};

// Helper template for adding a node in a compound space.  It tail
// recurses until it finds the axis that is in the range of the
// subspace.  This essentially performs a linear search through the
// subspaces (again).
template <int _index, typename ... _Spaces>
struct CompoundDoAdd {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;
    
    template <typename _RootAdder, typename _T>
    static unsigned doAdd(std::tuple<KDAdder<_Spaces>...>& adders, _RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        if (axis < Subspace::axes)
            return std::get<_index>(adders).doAdd(adder, axis, static_cast<typename Subspace::Distance>(d), p, n, depth);
        return CompoundDoAdd<_index+1, _Spaces...>::doAdd(adders, adder, axis - Subspace::axes, d, p, n, depth);
    }
};

// Base case.  The main difference is that it does not need to test if
// the axis is in range (though it does assert it).
template <typename ... _Spaces>
struct CompoundDoAdd<sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    static constexpr int N = sizeof...(_Spaces);
    typedef typename std::tuple_element<N-1, std::tuple<_Spaces...>>::type Subspace;
    
    template <typename _RootAdder, typename _T>
    static unsigned doAdd(std::tuple<KDAdder<_Spaces>...>& adders, _RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        assert(axis < Subspace::axes);
        return std::get<N-1>(adders).doAdd(adder, axis, static_cast<typename Subspace::Distance>(d), p, n, depth);
    }
};

template <int _index, typename ... _Spaces>
struct CompoundTraverse {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;

    template <typename _Nearest, typename _T, typename _MinDist>
    static void traverse(
        std::tuple<KDWalker<_Spaces>...>& walkers,
        _Nearest& t, const KDNode<_T>* n, int axis, Distance d,
        _MinDist minDist, unsigned depth)
    {
        assert(axis >= 0);
        if (axis < Subspace::axes)
            std::get<_index>(walkers).traverse(
                t, n, axis, static_cast<typename Subspace::Distance>(d), minDist, depth);
        else
            CompoundTraverse<_index+1, _Spaces...>::traverse(
                walkers, t, n, axis - Subspace::axes, d, minDist, depth);
    }
};

template <typename ... _Spaces>
struct CompoundTraverse<sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    static constexpr int N = sizeof...(_Spaces);
    typedef typename std::tuple_element<N-1, std::tuple<_Spaces...>>::type Subspace;

    template <typename _Nearest, typename _T, typename _MinDist>
    static void traverse(
        std::tuple<KDWalker<_Spaces>...>& walkers,
        _Nearest& t, const KDNode<_T>* n, int axis, Distance d,
        _MinDist minDist, unsigned depth)
    {
        assert(0 <= axis && axis < Subspace::axes);
        std::get<N-1>(walkers).traverse(
            t, n, axis, d, minDist, depth);
    }
};


template <template <typename> class _KDAdder, typename ... _Spaces>
struct KDCompoundTraversal : KDAdderBase<CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    std::tuple<_KDAdder<_Spaces>...> adders_;
    
    typedef std::make_index_sequence<Space::size> _Indexes;
    
    template <std::size_t ... I>
    static auto make(const State& key, const Space& space, std::index_sequence<I...>) {
        return std::make_tuple(
            _KDAdder<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>(
                key.template substate<I>(),
                space.template subspace<I>())...);
    }
    
    inline KDCompoundTraversal(const State& key, const CompoundSpace<_Spaces...>& space)
        : adders_(make(key, space, _Indexes{}))
    {
    }

    Distance maxAxis(int *axis) {
        typedef typename std::tuple_element<0, std::tuple<_Spaces...>>::type Subspace0;
        Distance dist = std::get<0>(adders_).maxAxis(axis);
        return CompoundMaxAxis<1, _KDAdder, _Spaces...>::maxAxis(adders_, dist, axis, Subspace0::axes);
    }    

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        return CompoundDoAdd<0, _Spaces...>::doAdd(adders_, adder, axis, d, p, n, depth);
    }
};

template <typename ... _Spaces>
struct KDAdder<CompoundSpace<_Spaces...>> : KDCompoundTraversal<detail::KDAdder, _Spaces...> {
    using KDCompoundTraversal<detail::KDAdder, _Spaces...>::KDCompoundTraversal;
};

template <typename ... _Spaces>
struct KDWalker<CompoundSpace<_Spaces...>> : KDCompoundTraversal<detail::KDWalker, _Spaces...> {
    typedef KDCompoundTraversal<detail::KDWalker, _Spaces...> Base;

    // inherit constructor
    using KDCompoundTraversal<detail::KDWalker, _Spaces...>::KDCompoundTraversal;
    
    using typename Base::Distance;

    template <typename _Nearest, typename _T, typename _MinDist>
    void traverse(
        _Nearest& t, const KDNode<_T>* n,
        int axis, Distance dist,
        _MinDist minDist, unsigned depth)
    {
        CompoundTraverse<0, _Spaces...>::template traverse(
            this->adders_,
            t, n, axis, dist, minDist, depth);
    }
};

// template <typename ... _Spaces>
// struct KDWalker<CompoundSpace<_Spaces...>> {
//     typedef CompoundSpace<_Spaces...> Space;
//     typedef typename Space::State State;
//     typedef typename Space::Distance Distance;

//     std::tuple<KDWalker<_Spaces>...> walkers_;

//     typedef std::make_index_sequence<Space::size> _Indexes;

//     KDWalker(const State& key, const Space& space)
//         : KDWalker(key, space, _Indexes{})
//     {
//     }

//     template <std::size_t ... I>
//     KDWalker(const State& key, const Space& space, std::index_sequence<I...>)
//         : walkers_(KDWalker<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>(
//                        key.template substate<I>(),
//                        space.template substate<I>())...);
//     {
//     }

//     Distance maxAxis(int *axis) {
//         typedef typename std::tuple_element<0, std::tuple<_Spaces...>>::type Subspace0;
//         Distance dist = std::get<0>(walkers_).maxAxis(axis);
//         return CompoundMaxAxis<1, KDWalker, _Spaces...>::maxAxis(walkers_, dist, axis, Subspace0::axes);
//     }

//     template <typename _Nearest, typename _T, typename _MinDist>
//     void traverse(_Nearest& t, const KDNode<_T> *n, int axis, Distance dist, _MinDist minDist, unsigned depth) {
//         // CompoundTraverse<0, _Spaces...>::traverse(
            
//     }
    
// };

template <typename _Space, typename _T, typename _TtoKey>
struct KDNearest {
    typedef typename _Space::Distance Distance;

    // TODO: update template to use member pointers take tree instead
    // (and maybe use member pointers, or just standard locations.)
    // (Will need a friend in the tree.)
    const _Space& space_;
    const _TtoKey& tToKey_;

    const typename _Space::State& key_;
    
    KDWalker<_Space> walker_;
    const KDNode<_T>* nearest_;
    Distance dist_;

    KDNearest(
        const _Space& space,
        const _TtoKey& tToKey,
        const typename _Space::State& key)
        : space_(space),
          tToKey_(tToKey),
          key_(key),
          walker_(key, space),
          nearest_(nullptr),
          dist_(std::numeric_limits<Distance>::infinity())
    {
        // std::cout << "===== " << std::endl;
    }

    Distance minDistSquared() const {
        return dist_*dist_;
    }

    Distance minDist() const {
        return dist_;
    }
    
    void traverse(const KDNode<_T>* n, Distance minDist, unsigned depth) {
        int axis;
        Distance dist = walker_.maxAxis(&axis);
        walker_.traverse(*this, n, axis, dist, minDist, depth+1);
    }

    void update(const KDNode<_T>* n) {
        Distance d = space_.distance(tToKey_(n->value_), key_);
        // std::cout << "dist " << d << " < " << dist_ << std::endl;
        if (d < dist_) {
            dist_ = d;
            nearest_ = n;
        }
    }
};


} // namespace detail



template <typename _T, typename _Space, typename _TtoKey>
class KDTree {
    typedef detail::KDNode<_T> Node;
    typedef typename _Space::State Key;
    typedef typename _Space::Distance Distance;

    _Space space_;
    Node *root_;
    std::size_t size_;
    unsigned depth_;

    _TtoKey tToKey_;

public:
    KDTree(_TtoKey tToKey, const _Space& space = _Space())
        : space_(space),
          root_(nullptr),
          size_(0),
          depth_(0),
          tToKey_(tToKey)
    {
    }

    ~KDTree() {
        if (root_)
            delete root_;
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    unsigned depth() const {
        return depth_;
    }

    void add(const _T& value) {
        Node* n = new Node(value);
        Node* p = root_;
        unsigned depth = 1;
        
        if (p == nullptr) {
            root_ = n;
        } else {
            const typename _Space::State& key = tToKey_(value);
            detail::KDAdder<_Space> adder(key, space_);
            depth = adder(p, n, 2);
        }
        
        ++size_;
        if (depth_ < depth)
            depth_ = depth;
    }

    // Returns a pointer to the nearest _T in the tree, or `nullptr`
    // if the tree is empty.
    const _T* nearest(const Key& key, Distance* dist = nullptr) {
        if (root_ == nullptr)
            return nullptr;
        
        detail::KDNearest<_Space, _T, _TtoKey> nearest(space_, tToKey_, key);
        nearest.traverse(root_, 0, 0);
        if (dist)
            *dist = nearest.dist_;
        return &nearest.nearest_->value_;
    }

    template <typename _Fn>
    void visit(const _Fn& fn) const {
        if (root_)
            root_->visit(fn, 0);
    }

    // // Returns the `k` nearest neighbors of `key`.  The nearest
    // // neighbors can optionally be bounded to a maximum radius.
    // const void nearest(
    //     std::vector<std::pair<Distance,_T>>& results,
    //     const Key& key,
    //     std::size_t k,
    //     Distance maxRadius = std::numeric_limits<Distance>::infinity())
    // {
    //     results.clear();
    //     if (!root_)
    //         return;

    //     detail::KDNearestK<_Space, _T, _TtoKey> nearest(space_, tToKey_, key, k, maxRadius, results);
    //     nearest.traverse(root_, 0, 0);
    //     std::sort_heap(results.begin(), results.end(), detail::ComparePairFirst());
    // }
};

} // namespace kdtree
} // namespace robotics
} // namespace unc

#endif // UNC_KDTREE_HPP
