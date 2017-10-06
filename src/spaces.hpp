#pragma once
#ifndef UNC_ROBOTICS_KDTREE_SPACES_HPP
#define UNC_ROBOTICS_KDTREE_SPACES_HPP

#include <tuple>
#include <Eigen/Dense>

namespace unc {
namespace robotics {
namespace kdtree {


namespace detail {

// Template that computes the sum of its template arguments.  This is
// also the base case for when _args is empty.
template <typename _T, _T _a, _T ... _args>
struct Sum { static constexpr _T value = _a; };

// Recursion: compute the sum of the first two arguments then recurse.
template <typename _T, _T _a, _T _b, _T ... _rest>
struct Sum<_T, _a, _b, _rest...> { static constexpr _T value = Sum<_T, _a + _b, _rest...>::value; };


// Template to compute the result type of a sumation of arbitrary
// scalars.  This follows standard C++ type promotion rules,
// e.g. float+float+double => double, etc...
template <typename _T, typename ... _Rest>
struct SumResult { typedef _T type; };
template <typename _T, typename _U, typename ... _Rest>
struct SumResult<_T, _U, _Rest...> { typedef typename SumResult<decltype(_T() + _U()), _Rest...>::type type; };

// Helper template for computing the distance in CompoundSpace.  The
// distance is the sum of distances for each substate using the
// distance function of the associated subspace.
template <int _index, typename ... _Spaces>
struct CompoundDistance {
    typedef std::tuple<typename _Spaces::State...> State;
    typedef typename SumResult<typename _Spaces::Distance...>::type Distance;

    inline static Distance accum(
        const std::tuple<_Spaces...>& spaces,
        const State& a,
        const State& b,
        Distance sum)
    {
        // tail recursion
        return CompoundDistance<_index + 1, _Spaces...>::accum(
            spaces, a, b,
            sum + std::get<_index>(spaces).distance(std::get<_index>(a), std::get<_index>(b)));
    }
};
// Base case
template <typename ... _Spaces>
struct CompoundDistance<sizeof...(_Spaces), _Spaces...> {
    typedef std::tuple<typename _Spaces::State...> State;
    typedef typename SumResult<typename _Spaces::Distance...>::type Distance;

    inline static Distance accum(
        const std::tuple<_Spaces...>& spaces,
        const State& a,
        const State& b,
        Distance sum)
    {
        return sum;
    }
};

} // namespace unc::robotics::kdtree::detail


// Euclidean metric space.  The state is a vector of the specified
// dimensions and of the associated scalar type.
template <typename _Scalar, int _dimensions>
class L2Space {
public:
    static_assert(std::is_floating_point<_Scalar>::value, "scalar type must be a floating point type");
    // this next assert also rules out using Eigen::Dynamic
    static_assert(_dimensions > 0, "dimensions must be positive");
    
    typedef _Scalar Distance;
    typedef Eigen::Matrix<_Scalar, _dimensions, 1> State;
    static constexpr int dimensions = _dimensions;

    inline Distance distance(const State& a, const State& b) const {
        return (a - b).norm();
    }
};

template <typename _Scalar, int _dimensions>
class BoundedL2Space : public L2Space<_Scalar, _dimensions> {
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;

    void checkBounds() {
        // check for NaN
        assert((bounds_ == bounds_).all());
        // make sure that min bounds < max bounds
        assert((bounds_.col(0) < bounds_.col(1)).all());
        // make sure that all bounds are finite (and thus bounded)
        assert(((bounds_.col(1) - bounds_.col(0)) < std::numeric_limits<_Scalar>::infinity()).all());
    }

public:
    template <typename _Derived>
    BoundedL2Space(const Eigen::DenseBase<_Derived>& bounds)
        : bounds_(bounds)
    {
        checkBounds();
    }

    template <typename _DerivedMin, typename _DerivedMax>
    BoundedL2Space(
        const Eigen::DenseBase<_DerivedMin>& min,
        const Eigen::DenseBase<_DerivedMax>& max)
    {
        bounds_.col(0) = min;
        bounds_.col(1) = max;
        checkBounds();
    }

    const Eigen::Array<_Scalar, _dimensions, 2>& bounds() const {
        return bounds_;
    }

    template <typename _Index>
    _Scalar bounds(_Index dim, _Index j) const {
        return bounds_(dim, j);
    }
};

template <typename _Scalar>
class SO3Space {
public:
    static_assert(std::is_floating_point<_Scalar>::value, "scalar type must be a floating point type");

    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;
    static constexpr int dimensions = 3;

    inline Distance distance(const State& a, const State& b) const {
        Distance dot = std::abs(a.coeffs().matrix().dot(b.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
class RatioWeightedSpace : public _Space {
public:
    typedef std::ratio<_num, _den> Ratio;

    // Define num and den in terms of the std::ratio members instead
    // of template arguments, since it provides a GCD and sign
    // reduction.
    static constexpr std::intmax_t num = Ratio::num;
    static constexpr std::intmax_t den = Ratio::den;
    
    using _Space::_Space;

    static_assert(num > 0, "ratio weight must be positive");
    
    RatioWeightedSpace(const _Space& space)
        : _Space(space)
    {
    }

    inline typename _Space::Distance distance(
        const typename _Space::State& a,
        const typename _Space::State& b) const
    {
        return _Space::distance(a, b) * num / den;
    }
};

template <typename _Space>
class WeightedSpace : public _Space {
    typename _Space::Distance weight_;
    
public:
    using _Space::_Space;

    WeightedSpace(const _Space& space, typename _Space::Distance weight)
        : _Space(space), weight_(weight)
    {
        assert(weight == weight); // no NaNs
        assert(weight >= 0); // non-negative
        assert(weight != std::numeric_limits<typename _Space::Distance>::infinity()); // finite
    }

    inline typename _Space::Distance distance(
        const typename _Space::State& a,
        const typename _Space::State& b) const
    {
        return _Space::distance(a, b) * weight_;
    }
};

template <typename ... _Spaces>
class CompoundSpace {
    typedef std::tuple<_Spaces...> Spaces;
    
    Spaces spaces_;

    // the implementation currently supports 1 subspace (probably),
    // but making the minimum 2 allows us to make use of that
    // assumption if needed.  The implementation does not support 0
    // subspaces.
    static_assert(sizeof...(_Spaces) > 1, "compound space must have two or more subspaces");

public:
    typedef std::tuple<typename _Spaces::State...> State;
    static constexpr int dimensions = detail::Sum<int, _Spaces::dimensions...>::value;
    typedef typename detail::SumResult<typename _Spaces::Distance...>::type Distance;

    Distance distance(const State& a, const State& b) const {
        return detail::CompoundDistance<1, _Spaces...>::accum(
            spaces_, a, b,
            std::get<0>(spaces_).distance(std::get<0>(a), std::get<0>(b)));
    }

    template <std::size_t I>
    constexpr typename std::tuple_element<I, Spaces>::type& get() {
        return std::get<I>(spaces_);
    }

    template <std::size_t I>
    constexpr typename std::tuple_element<I, Spaces>::type const& get() const {
        return std::get<I>(spaces_);
    }
};

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using SE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<L2Space<_Scalar, 3>, _tWeight, 1>>;

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using BoundedSE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<BoundedL2Space<_Scalar, 3>, _tWeight, 1>>;


} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

namespace std {
// provide implementations of std::get for CompoundSpace.  The
// implementations also shows the reason we provide
// std::get<I>(space), because the alternative is to use
// space.template get<I>() (i.e., avoid the otherwise required extra
// 'template' keyword)
template <std::size_t I, class ... _Spaces>
constexpr typename std::tuple_element<I, std::tuple<_Spaces...>>::type&
get(unc::robotics::kdtree::CompoundSpace<_Spaces...>& space)
{
    return space.template get<I>();
}

template <std::size_t I, class ... _Spaces>
constexpr typename std::tuple_element<I, std::tuple<_Spaces...>>::type const&
get(const unc::robotics::kdtree::CompoundSpace<_Spaces...>& space)
{
    return space.template get<I>();
}

} // namespace std

#endif // UNC_ROBOTICS_KDTREE_SPACES_HPP
