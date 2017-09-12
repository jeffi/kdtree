#pragma once
#include <random>

template <typename _Space, typename _RNG>
typename _Space::State randomState(const _Space& space, _RNG& rng);


template <typename _Scalar, typename _RNG>
typename unc::robotics::kdtree::SO3Space<_Scalar>::State
randomState(const unc::robotics::kdtree::SO3Space<_Scalar>&, _RNG& rng) {
    std::uniform_real_distribution<_Scalar> dist01(0, 1);
    std::uniform_real_distribution<_Scalar> dist2pi(0, 2*M_PI);
    _Scalar a = dist01(rng);
    _Scalar b = dist2pi(rng);
    _Scalar c = dist2pi(rng);

    return Eigen::Quaternion<_Scalar>(
        std::sqrt(1-a)*std::sin(b),
        std::sqrt(1-a)*std::cos(b),
        std::sqrt(a)*std::sin(c),
        std::sqrt(a)*std::cos(c));
}

template <typename _Scalar, int _dim, typename _RNG>
typename unc::robotics::kdtree::BoundedEuclideanSpace<_Scalar, _dim>::State
randomState(const unc::robotics::kdtree::BoundedEuclideanSpace<_Scalar, _dim>& space, _RNG& rng) {
    Eigen::Matrix<_Scalar, _dim, 1> q;
    
    for (int i=0 ; i<_dim ; ++i) {
        std::uniform_real_distribution<_Scalar> dist(space.bounds(i, 0), space.bounds(i, 1));
        q[i] = dist(rng);
    }

    return q;
}

template <typename _Space, std::intmax_t _num, std::intmax_t _den, typename _RNG>
typename _Space::State
randomState(const unc::robotics::kdtree::RatioWeightedSpace<_Space, _num, _den>& space, _RNG& rng) {
    return randomState(*static_cast<const _Space*>(&space), rng);
}

template <typename _RNG, typename _Space, std::size_t ... _I>
auto randomCompoundState(const _Space& space, _RNG& rng, std::index_sequence<_I...>) {
    return typename _Space::State(randomState(space.template subspace<_I>(), rng)...);
}


template <typename _RNG, typename ... _Spaces>
typename unc::robotics::kdtree::CompoundSpace<_Spaces...>::State
randomState(const unc::robotics::kdtree::CompoundSpace<_Spaces...>& space, _RNG& rng) {
    return randomCompoundState(space, rng, std::make_index_sequence<sizeof...(_Spaces)>{});
}
