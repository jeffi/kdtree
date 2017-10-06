#include "../src/kdtree.hpp"
#include "test.hpp"
#include <random>

template <typename _State>
struct TestNode {
    _State state_;
    int name_;
    
    TestNode(const _State& state, int name)
        : state_(state),
          name_(name)
    {
    }
};

struct TestNodeKey {
    template <typename _State>
    inline const _State& operator() (const TestNode<_State>& node) const {
        return node.state_;
    }
};

template <typename _RNG, typename _Scalar, int _dimensions>
typename unc::robotics::kdtree::BoundedL2Space<_Scalar,_dimensions>::State
randomState(
    _RNG& rng,
    const unc::robotics::kdtree::BoundedL2Space<_Scalar, _dimensions>& space)
{
    typename unc::robotics::kdtree::BoundedL2Space<_Scalar, _dimensions>::State q;
    for (int i=0 ; i<_dimensions ; ++i) {
        std::uniform_real_distribution<_Scalar> dist(space.bounds(i, 0), space.bounds(i, 1));
        q[i] = dist(rng);
    }

    return q;
}

template <typename _RNG, typename _Scalar>
typename unc::robotics::kdtree::SO3Space<_Scalar>::State
randomState(
    _RNG& rng,
    const unc::robotics::kdtree::SO3Space<_Scalar>& state)
{
    typename unc::robotics::kdtree::SO3Space<_Scalar>::State q;
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

template <typename _Scalar, int _dim>
unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim> makeBoundedL2Space() {
    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    return unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim>(bounds);
}

template <typename _Scalar, int _dim, std::intmax_t _num, std::intmax_t _den>
unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim> makeRatioWeightedBoundedL2Space() {
    using namespace unc::robotics::kdtree;
    return RatioWeightedSpace<BoundedL2Space<_Scalar, _dim>, _num, _den>(
        makeBoundedL2Space<_Scalar, _dim>());
}

template <typename Space>
static void testAdd(const Space& space) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    constexpr int N = 1000;

    KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);

    EXPECT(tree.size()) == 0;
    EXPECT(tree.empty()) == true;
    
    std::mt19937_64 rng;
    std::vector<std::pair<Distance, TestNode<State>>> nearestK;
    
    for (int i=0 ; i<N ; ++i) {
        // std::cout << "# " << i << std::endl;
        State q = randomState(rng, space);
        Distance zero = space.distance(q, q);
        tree.add(TestNode<State>(q, i));

        EXPECT(tree.size()) == i+1;
        EXPECT(tree.empty()) == false;
        int minDepth = std::floor(std::log(i+1) / std::log(2) + 1);
        // int maxDepth = std::ceil(minDepth*(1 + std::log(2)));
        EXPECT(tree.depth()) >= minDepth;
        EXPECT(tree.depth()) <= 3 * minDepth;

        Distance dist;
        const TestNode<State>* nearest = tree.nearest(q, &dist);
        EXPECT(nearest) != nullptr;
        EXPECT(nearest->name_) == i;
        EXPECT(dist) == zero;

        tree.nearest(nearestK, q, 1);
        EXPECT(nearestK.size()) == 1;
        EXPECT(nearestK[0].first) == zero;
        EXPECT(nearestK[0].second.name_) == i;
    }
}

TEST_CASE(KDTree_RV3_add_double) {
    testAdd(makeBoundedL2Space<double, 3>());
}

TEST_CASE(KDTree_RV3_add_float) {
    testAdd(makeBoundedL2Space<float, 3>());
}

TEST_CASE(KDTree_RV3_add_long_double) {
    testAdd(makeBoundedL2Space<long double, 3>());
}

TEST_CASE(KDTree_SO3_add_double) {
    testAdd(unc::robotics::kdtree::SO3Space<double>());
}

TEST_CASE(KDTree_SO3_add_float) {
    testAdd(unc::robotics::kdtree::SO3Space<float>());
}

TEST_CASE(KDTree_SO3_add_long_double) {
    testAdd(unc::robotics::kdtree::SO3Space<long double>());
}

TEST_CASE(KDTree_RatioWeightedRV3_add) {
    testAdd(makeRatioWeightedBoundedL2Space<double, 3, 11, 3>());
}

template <typename Space>
static void testKNN(const Space& space, std::size_t N, std::size_t Q, std::size_t k) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);
    
    std::mt19937_64 rng;
    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(k);
    for (std::size_t i=0 ; i<Q ; ++i) {
        auto q = randomState(rng, space);
        tree.nearest(nearest, q, k);

        EXPECT(nearest.size()) == k;
        
        std::partial_sort(nodes.begin(), nodes.begin() + k, nodes.end(), [&q, &space] (auto& a, auto& b) {
            return space.distance(q, a.state_) < space.distance(q, b.state_);
        });

        for (std::size_t j=0 ; j<k ; ++j) {
            EXPECT(nearest[j].second.name_) == nodes[j].name_;
        }
    }
}

TEST_CASE(KDTree_RV3_nearestK_float) {
    testKNN(makeBoundedL2Space<float, 3>(), 5000, 500, 20);
}

TEST_CASE(KDTree_RV3_nearestK_double) {
    testKNN(makeBoundedL2Space<double, 3>(), 5000, 500, 20);
}

TEST_CASE(KDTree_RV3_nearestK_long_double) {
    testKNN(makeBoundedL2Space<long double, 3>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SO3_nearestK_float) {
    testKNN(unc::robotics::kdtree::SO3Space<float>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SO3_nearestK_double) {
    testKNN(unc::robotics::kdtree::SO3Space<double>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SO3_nearestK_long_double) {
    testKNN(unc::robotics::kdtree::SO3Space<long double>(), 5000, 500, 20);
}

TEST_CASE(KDTree_RatioWeightedRV3_nearestK) {
    testKNN(makeRatioWeightedBoundedL2Space<double, 3, 17, 5>(), 5000, 500, 20);
}

template <typename Space>
static void testRNN(const Space& space, std::size_t N, std::size_t Q, typename Space::Distance maxRadius) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);
    
    std::mt19937_64 rng;
    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::size_t total = 0;
    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(N);
    for (std::size_t i=0 ; i<Q ; ++i) {
        auto q = randomState(rng, space);
        tree.nearest(nearest, q, N, maxRadius);

        // maxRadius must be small enough that the following assert does not trigger
        EXPECT(nearest.size()) != N;

        // swap nodes s.t. all nodes in [begin, mid) <= maxRadius
        auto mid = nodes.begin();
        for (auto j = mid ; j != nodes.end() ; ++j)
            if (space.distance(q, j->state_) <= maxRadius)
                std::swap(*mid++, *j);

        EXPECT(mid - nodes.begin()) == nearest.size();
        
        // sort the nodes into order
        std::sort(nodes.begin(), mid, [&q, &space] (auto& a, auto& b) {
            return space.distance(q, a.state_) < space.distance(q, b.state_);
        });

        for (std::size_t j=0 ; j<nearest.size() ; ++j)
            EXPECT(nearest[j].second.name_) == nodes[j].name_;

        total += nearest.size();
    }

    // maxRadius must be large enough that the average is > 10
    EXPECT(total/(double)Q) > 10;
}

TEST_CASE(KDTree_RV3_nearestR_float) {
    testRNN(makeBoundedL2Space<float, 3>(), 5000, 100, 0.5);
}

TEST_CASE(KDTree_RV3_nearestR_double) {
    testRNN(makeBoundedL2Space<double, 3>(), 5000, 100, 0.5);
}

TEST_CASE(KDTree_RV3_nearestR_long_double) {
    testRNN(makeBoundedL2Space<long double, 3>(), 5000, 100, 0.5);
}

TEST_CASE(KDTree_SO3_nearestR_float) {
    testRNN(unc::robotics::kdtree::SO3Space<float>(), 5000, 100, 0.2);
}

TEST_CASE(KDTree_SO3_nearestR_double) {
    testRNN(unc::robotics::kdtree::SO3Space<double>(), 5000, 100, 0.2);
}

TEST_CASE(KDTree_SO3_nearestR_long_double) {
    testRNN(unc::robotics::kdtree::SO3Space<long double>(), 5000, 100, 0.2);
}

TEST_CASE(KDTree_RatioWeightedRV3_nearestR) {
    testRNN(makeRatioWeightedBoundedL2Space<double, 3, 17, 5>(), 5000, 100, 0.5);
}


template <typename Space, typename _Duration>
static std::pair<std::size_t, double>
benchmarkKNN(const Space& space, std::size_t N, std::size_t k, _Duration maxDuration) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);
    
    std::mt19937_64 rng;
    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(k);
    std::chrono::high_resolution_clock::duration maxElapsed = maxDuration;
    std::size_t count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (;;) {
        for (std::size_t i=0 ; i<100 ; ++i) {
            auto q = randomState(rng, space);
            tree.nearest(nearest, q, k);
        }
        count += 100;
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (elapsed > maxElapsed)
            return std::make_pair(count, std::chrono::duration<double>(elapsed).count());
    }
}


template <typename _Scalar, int _dim, typename _Duration>
static std::pair<std::size_t, double>
benchmarkKNNL2(std::size_t N, std::size_t k, _Duration maxDuration) {
    using namespace unc::robotics::kdtree;

    typedef BoundedL2Space<_Scalar, _dim> Space;

    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    Space space((bounds));

    return benchmarkKNN(space, N, k, maxDuration);
}

TEST_CASE(benchmark) {
    using namespace unc::robotics::kdtree;
    using namespace std::literals::chrono_literals;
    
    auto result = benchmarkKNNL2<double, 3>(50000, 20, 1s);
    std::cout << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNN(SO3Space<double>(), 50000, 20, 1s);
    std::cout << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;
}

