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
#ifndef UNC_ROBOTICS_KDTREE_SPACES_HPP
#define UNC_ROBOTICS_KDTREE_SPACES_HPP

#include <Eigen/Dense>
#include <tuple>
#include <functional>
#include <utility>
#include <type_traits>
#include <ratio>
#include "_tuple.hpp"

namespace unc {
namespace robotics {
namespace kdtree {

namespace detail {
template <typename _Scalar, int _dimensions>
class L2SpaceBase {
public:
    typedef _Scalar Distance;
    typedef Eigen::Matrix<_Scalar, _dimensions, 1> State;

    template <typename _Derived>
    bool isValid(const Eigen::MatrixBase<_Derived>& q) const {
        return q.rows() == _dimensions && q.cols() == 1 && q.allFinite();
    }

    template <typename _DerivedA, typename _DerivedB>
    constexpr Distance distance(
        const Eigen::MatrixBase<_DerivedA>& a,
        const Eigen::MatrixBase<_DerivedB>& b) const
    {
        return (a - b).norm();
    }

    template <typename _DerivedA, typename _DerivedB>
    constexpr State interpolate(
        const Eigen::MatrixBase<_DerivedA>& from,
        const Eigen::MatrixBase<_DerivedB>& to,
        Distance t) const
    {
        return from + (to - from) * t;
    }
};
}

template <typename _Scalar, int _dimensions>
class L2Space : public detail::L2SpaceBase<_Scalar, _dimensions> {
public:
    L2Space(unsigned dimensions = _dimensions) {
        assert(dimensions == _dimensions);
    }

    constexpr unsigned dimensions() const { return _dimensions; }

    template <typename _Derived>
    bool isValid(const Eigen::MatrixBase<_Derived>& q) const {
        return q.rows() == _dimensions && detail::L2SpaceBase<_Scalar, _dimensions>::isValid(q);
    }
};

template <typename _Scalar>
class L2Space<_Scalar, Eigen::Dynamic> : public detail::L2SpaceBase<_Scalar, Eigen::Dynamic> {
    unsigned dimensions_;
    
public:
    L2Space(unsigned dimensions)
        : dimensions_(dimensions)
    {
    }

    constexpr unsigned dimensions() const {
        return dimensions_;
    }

    template <typename _Derived>
    bool isValid(const Eigen::MatrixBase<_Derived>& q) const {
        return q.rows() == dimensions_ && detail::L2SpaceBase<_Scalar, Eigen::Dynamic>::isValid(q);
    }
};

template <typename _Scalar, int _dimensions>
class BoundedL2Space : public L2Space<_Scalar, _dimensions> {
    Eigen::Array<_Scalar, _dimensions, 2> bounds_;
    
    typedef typename Eigen::Array<_Scalar, _dimensions, 2>::Index Index;

    void checkBounds() {
        assert((bounds_.col(0) < bounds_.col(1)).all());
        assert((bounds_.col(1) - bounds_.col(0)).allFinite());
    }

public:
    using typename L2Space<_Scalar, _dimensions>::State;

    template <typename _Derived>
    BoundedL2Space(const Eigen::DenseBase<_Derived>& bounds)
        : L2Space<_Scalar, _dimensions>(bounds.rows()),
          bounds_(bounds)
    {
        checkBounds();
    }

    template <typename _DerivedMin, typename _DerivedMax>
    BoundedL2Space(
        const Eigen::DenseBase<_DerivedMin>& min,
        const Eigen::DenseBase<_DerivedMax>& max)
        : L2Space<_Scalar, _dimensions>(min.rows())
    {
        bounds_.col(0) = min;
        bounds_.col(1) = max;
    }

    template <typename _Derived>
    bool isValid(const Eigen::MatrixBase<_Derived>& q) const {
        return L2Space<_Scalar, _dimensions>::isValid(q)
            && (bounds_.col(0) <= q).all()
            && (bounds_.col(1) >= q).all();
    }

    const Eigen::Array<_Scalar, _dimensions, 2>& bounds() const {
        return bounds_;
    }

    _Scalar bounds(Index dim, Index j) const {
        return bounds_(dim, j);
    }
};

template <typename _Scalar>
class SO3Space {
public:
    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;

    constexpr unsigned dimensions() const { return 3; }

    template <typename _Derived>
    bool isValid(const Eigen::QuaternionBase<_Derived>& q) const {
        return std::abs(1 - q.coeffs().squaredNorm()) <= 1e-5;
    }

    template <typename _DerivedA, typename _DerivedB>
    inline Distance distance(
        const Eigen::QuaternionBase<_DerivedA>& a,
        const Eigen::QuaternionBase<_DerivedB>& b) const
    {
        Distance dot = std::abs(a.coeffs().matrix().dot(b.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    template <typename _DerivedA, typename _DerivedB>
    constexpr State interpolate(
        const Eigen::QuaternionBase<_DerivedA>& from,
        const Eigen::QuaternionBase<_DerivedB>& to,
        Distance t) const
    {
        Distance dq = from.coeffs().matrix().dot(to.coeffs().matrix());
        if (std::abs(dq) >= 1)
            return from;
        
        Distance theta = std::acos(std::abs(dq));            
        Distance d = 1 / std::sin(theta);
        Distance s0 = std::sin((1 - t) * theta);
        Distance s1 = std::sin(t * theta);

        if (dq < 0)
            s1 = -s1;

        return State(d * (from.coeffs() * s0 + to.coeffs() * s1));
    }
};

template <typename _Scalar>
class SO3AltSpace {
public:
    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;

    constexpr unsigned dimensions() const { return 3; }

    template <typename _Derived>
    bool isValid(const Eigen::QuaternionBase<_Derived>& q) const {
        return std::abs(1 - q.coeffs().squaredNorm()) <= 1e-5;
    }

    template <typename _DerivedA, typename _DerivedB>
    inline Distance distance(
        const Eigen::QuaternionBase<_DerivedA>& a,
        const Eigen::QuaternionBase<_DerivedB>& b) const
    {
        Distance dot = std::abs(a.coeffs().matrix().dot(b.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }
};

template <typename _Scalar>
class SO3RLSpace {
public:
    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;

    constexpr unsigned dimensions() const { return 4; }

    template <typename _Derived>
    bool isValid(const Eigen::QuaternionBase<_Derived>& q) const {
        return std::abs(1 - q.coeffs().squaredNorm()) <= 1e-5;
    }

    template <typename _DerivedA, typename _DerivedB>
    inline Distance distance(
        const Eigen::QuaternionBase<_DerivedA>& a,
        const Eigen::QuaternionBase<_DerivedB>& b) const
    {
        Distance dot = std::abs(a.coeffs().matrix().dot(b.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }
};

template <typename _Space, typename _Ratio = std::ratio<1>>
class RatioWeightedSpace : public _Space {
public:
    typedef _Ratio Ratio;

    static constexpr std::intmax_t num = Ratio::num;
    static constexpr std::intmax_t den = Ratio::den;

    // inherit constructor
    using _Space::_Space;

    // RatioWeightedSpace() {}

    RatioWeightedSpace(const _Space& space)
        : _Space(space)
    {
    }

    RatioWeightedSpace(_Space&& space)
        : _Space(std::forward<_Space>(space))
    {
    }

    template <typename _A, typename _B>
    inline typename _Space::Distance distance(const _A& a, const _B& b) const {
        return _Space::distance(a, b) * num / den;
    }
};

template <std::intmax_t num, std::intmax_t den = 1, typename _Space>
auto makeRatioWeightedSpace(_Space&& space) {
    return RatioWeightedSpace<_Space, std::ratio<num, den>>(std::forward<_Space>(space));
}

template <typename _Space>
class WeightedSpace : public _Space {
public:
    using typename _Space::Distance;

private:
    Distance weight_;

public:
    WeightedSpace(Distance weight, const _Space& space = _Space())
        : _Space(space), weight_(weight)
    {
    }

    WeightedSpace(Distance weight, _Space&& space)
        : _Space(std::forward<_Space>(space)), weight_(weight)
    {
    }

    template <typename ... _Args>
    WeightedSpace(Distance weight, _Args&& ... args)
        : _Space(std::forward<_Args>(args)...),
          weight_(weight)
    {
    }

    constexpr Distance weight() const {
        return weight_;
    }

    template <typename _A, typename _B>
    constexpr Distance distance(const _A& a, const _B& b) const {
        return _Space::distance(a, b) * weight_;
    }    
};

template <typename ... _Spaces>
class CompoundSpace {
    typedef std::tuple<_Spaces...> Spaces;

    Spaces spaces_;

    static_assert(sizeof...(_Spaces) > 1, "compound space must have two or more subspaces");

public:
    typedef std::tuple<typename _Spaces::State...> State;
    typedef typename detail::SumResultType<typename _Spaces::Distance...>::type Distance;

    explicit CompoundSpace() {
    }
    
    explicit CompoundSpace(const _Spaces& ... args)
        : spaces_(args...)
    {
    }

    template <typename ... _Args>
    explicit CompoundSpace(_Args&& ... args)
        : spaces_(std::forward<_Args>(args)...)
    {
    }

    template <std::size_t I>
    constexpr typename std::tuple_element<I, Spaces>::type& get() {
        return std::get<I>(spaces_);
    }

    template <std::size_t I>
    constexpr typename std::tuple_element<I, Spaces>::type const& get() const {
        return std::get<I>(spaces_);
    }

    inline constexpr unsigned dimensions() const {
        using namespace detail;
        return sum(map([](const auto& space) { return space.dimensions(); }, spaces_));
    }

    template <typename _State>
    bool isValid(_State&& q) const {
        using namespace detail;
        // TODO: return reduce(std::logical_and<bool>(), zip([](auto&& subs, auto&& subq) { return subs.isValid(subq); }, spaces_, q));
        assert(false);
        return false;
    }

    template <typename _StateA, typename _StateB>
    inline Distance distance(_StateA&& a, _StateB&& b) const {
        using namespace detail;
        return sum(zip([](auto&& subs, auto&& suba, auto&& subb) {
                    return subs.distance(suba, subb);
                }, spaces_, a, b));
    }

private:
    template <typename _StateA, typename _StateB, std::size_t ... I>
    constexpr State interpolate(
        const _StateA& from,
        const _StateB& to,
        Distance t,
        std::index_sequence<I...>) const
    {
        return State(std::get<I>(spaces_).interpolate(std::get<I>(from), std::get<I>(to), t)...);
    }

public:
    template <typename _StateA, typename _StateB>
    constexpr State interpolate(
        const _StateA& from,
        const _StateB& to,
        Distance t) const
    {
        return interpolate(from, to, t, std::make_index_sequence<sizeof...(_Spaces)>{});
    }

};

template <typename ... _Spaces>
constexpr auto makeCompoundSpace(_Spaces&&... args) {
    return CompoundSpace<typename std::decay<_Spaces>::type...>(std::forward<_Spaces>(args)...);
}

template <typename _Space>
constexpr auto makeWeightedSpace(
    typename _Space::Distance weight,
    _Space&& space = _Space())
{
    return WeightedSpace<_Space>(weight, std::forward<_Space>(space));
}

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using SE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, std::ratio<_qWeight>>,
    RatioWeightedSpace<L2Space<_Scalar, 3>, std::ratio<_tWeight>>>;

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using BoundedSE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, std::ratio<_qWeight>>,
    RatioWeightedSpace<BoundedL2Space<_Scalar, 3>, std::ratio<_tWeight>>>;

}
}
}

namespace std {

template <std::size_t I, class ... _Spaces>
class tuple_element<I, unc::robotics::kdtree::CompoundSpace<_Spaces...>> {
public:
    typedef typename std::tuple_element<I, std::tuple<_Spaces...>>::type type;
};

template <std::size_t I, class ... _Spaces>
constexpr typename std::tuple_element<I, std::tuple<_Spaces...>>::type&
get(unc::robotics::kdtree::CompoundSpace<_Spaces...>& space) {
    return space.template get<I>();
}

template <std::size_t I, class ... _Spaces>
constexpr typename std::tuple_element<I, std::tuple<_Spaces...>>::type const&
get(const unc::robotics::kdtree::CompoundSpace<_Spaces...>& space) {
    return space.template get<I>();
}


}

#endif // UNC_ROBOTICS_KDTREE_SPACES_HPP
