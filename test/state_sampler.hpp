#include <random>

template <typename _Space>
struct StateSampler {};

template <typename _Scalar, int _dimensions>
struct StateSampler<unc::robotics::kdtree::BoundedL2Space<_Scalar, _dimensions>> {
    typedef unc::robotics::kdtree::BoundedL2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    
    template <typename _RNG>
    static State randomState(_RNG& rng, const Space& space) {
        State q;
        for (int i=0 ; i<space.dimensions() ; ++i) {
            std::uniform_real_distribution<_Scalar> dist(space.bounds(i, 0), space.bounds(i, 1));
            q[i] = dist(rng);
        }
        return q;
    }
};

template <typename _Scalar, int _dimensions>
struct StateSampler<unc::robotics::kdtree::L2Space<_Scalar, _dimensions>> {
    typedef unc::robotics::kdtree::L2Space<_Scalar, _dimensions> Space;
    typedef typename Space::State State;
    
    template <typename _RNG>
    static State randomState(_RNG& rng, const Space& space) {
        State q;
        std::uniform_real_distribution<_Scalar> dist(-50, 50);
        for (int i=0 ; i<space.dimensions() ; ++i) {
            q[i] = dist(rng);
        }
        return q;
    }
};

template <typename _Scalar>
struct StateSampler<unc::robotics::kdtree::SO3Space<_Scalar>> {
    typedef unc::robotics::kdtree::SO3Space<_Scalar> Space;
    template <typename _RNG, typename _Space>
    static typename Space::State
    randomState(_RNG& rng, const _Space&) {
        typename Space::State q;
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
};

template <typename _Scalar>
struct StateSampler<unc::robotics::kdtree::SO3AltSpace<_Scalar>>
    : StateSampler<unc::robotics::kdtree::SO3Space<_Scalar>>
{};

template <typename _Scalar>
struct StateSampler<unc::robotics::kdtree::SO3RLSpace<_Scalar>>
    : StateSampler<unc::robotics::kdtree::SO3Space<_Scalar>>
{};

template <typename _Space, typename _Ratio>
struct StateSampler<unc::robotics::kdtree::RatioWeightedSpace<_Space, _Ratio>> {
    typedef unc::robotics::kdtree::RatioWeightedSpace<_Space, _Ratio> Space;

    template <typename _RNG>
    static typename Space::State
    randomState(_RNG& rng, const Space& space) {
        return StateSampler<_Space>::randomState(rng, space);
    }
};

template <typename _Space>
struct StateSampler<unc::robotics::kdtree::WeightedSpace<_Space>> {
    typedef unc::robotics::kdtree::WeightedSpace<_Space> Space;

    template <typename _RNG>
    static typename Space::State
    randomState(_RNG& rng, const Space& space) {
        return StateSampler<_Space>::randomState(rng, space);
    }
};

template <typename ... _Spaces>
struct StateSampler<unc::robotics::kdtree::CompoundSpace<_Spaces...>> {
    typedef unc::robotics::kdtree::CompoundSpace<_Spaces...> Space;

    template <typename _RNG, std::size_t ... I>
    static typename Space::State
    compoundRandomState(_RNG& rng, const Space& space, std::index_sequence<I...>)
    {
        return typename Space::State(
            StateSampler<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>
            ::randomState(rng, std::get<I>(space))...);
    }

    template <typename _RNG>
    static typename Space::State
    randomState(_RNG& rng, const Space& space) {
        return compoundRandomState(rng, space, std::make_index_sequence<sizeof...(_Spaces)>{});
    }
};
