#include <iostream>
#include "../src/kdtree_static.hpp"
#include "test.hpp"
#include "state_sampler.hpp"
#include <ratio>
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

    template <typename _State>
    inline const _State& operator() (const TestNode<_State>* node) const {
        return node->state_;
    }
};

template <typename _Scalar, int _dim>
unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim> makeBoundedL2Space() {
    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    return unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim>(bounds);
}

template <typename _Scalar, std::intmax_t _num = 1, std::intmax_t _den = 1>
unc::robotics::kdtree::BoundedSE3Space<_Scalar, _num, _den> makeBoundedSE3Space() {
    using namespace unc::robotics::kdtree;
    return BoundedSE3Space<_Scalar, _num, _den>(
        SO3Space<_Scalar>(),
        makeBoundedL2Space<_Scalar, 3>());
}


template <typename Space>
static void testBuildAndQuery(const Space& space, std::size_t N, std::size_t Q) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    constexpr std::size_t k = 20;

    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    std::mt19937_64 rng;
    for (std::size_t i=0 ; i<N ; ++i)
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);

    KDStaticTree<TestNode<State>, Space, TestNodeKey> kdtree(
        TestNodeKey(), space, nodes.begin(), nodes.end());

    std::vector<std::pair<Distance, TestNode<State>>> result;
    
    EXPECT(kdtree.size()) == N;
    
    for (std::size_t i=0 ; i<Q ; ++i) {
        State q = StateSampler<Space>::randomState(rng, space);
        auto actual = kdtree.nearest(q);
        EXPECT(actual) != nullptr;
        auto expected = std::min_element(nodes.begin(), nodes.end(), [&] (auto& a, auto& b) {
            return space.distance(q, a.state_) < space.distance(q, b.state_);
        });

        EXPECT(actual->name_) == expected->name_;

        std::partial_sort(nodes.begin(), nodes.begin()+k, nodes.end(), [&] (auto& a, auto& b) {
            return space.distance(q, a.state_) < space.distance(q, b.state_);
        });

        kdtree.nearest(result, q, k);
        EXPECT(result.size()) == k;
        for (std::size_t j=0 ; j<k ; ++j) {
            EXPECT(result[j].second.name_) == nodes[j].name_;
        }
    }
}

template <typename Space>
static void testBenchmarkQueries(const std::string& name, const Space& space, std::size_t N) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    // typedef typename Space::Distance Distance;

    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    std::mt19937_64 rng;
    for (std::size_t i=0 ; i<N ; ++i)
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);

    KDStaticTree<TestNode<State>, Space, TestNodeKey> kdtree(
        TestNodeKey(), space, nodes.begin(), nodes.end());

    using namespace std::literals::chrono_literals;

    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point start = Clock::now();
    Clock::duration elapsed;
    Clock::duration minElapsed = 1000ms;
    std::size_t Q = 0;
    do {
        for (std::size_t i=0 ; i<100 ; ++i) {
            State q = StateSampler<Space>::randomState(rng, space);
            kdtree.nearest(q);
        }
        Q += 100;
    } while ((elapsed = Clock::now() - start) < minElapsed);

    std::cout << name << ", N=" << N
              << ": " << std::chrono::duration<double, std::micro>(elapsed).count()/Q << " us/op" << std::endl;
}


TEST_CASE(KDStaticTree_RV3_buildAndQuery_double) {
    testBuildAndQuery(makeBoundedL2Space<double, 3>(), 1000, 1000);
}

TEST_CASE(KDStaticTree_RW3_buildAndQuery_double) {
    using namespace unc::robotics::kdtree;
    testBuildAndQuery(
        RatioWeightedSpace<BoundedL2Space<double, 3>, 5, 17>(
            makeBoundedL2Space<double, 3>()),
        1000, 1000);
}

TEST_CASE(KDStaticTree_WT3_buildAndQuery_double) {
    using namespace unc::robotics::kdtree;
    testBuildAndQuery(
        WeightedSpace<BoundedL2Space<double, 3>>(
            5.0/17.0, makeBoundedL2Space<double, 3>()),
        1000, 1000);
}


TEST_CASE(KDStaticTree_SO3_buildAndQuery_double) {
    using namespace unc::robotics::kdtree;
    testBuildAndQuery(SO3Space<double>(), 1000, 1000);
}

TEST_CASE(KDStaticTree_SE3_buildAndQuery_double) {
    using namespace unc::robotics::kdtree;
    testBuildAndQuery(makeBoundedSE3Space<double>(), 1000, 1000);
}


TEST_CASE(KDStaticTree_SO3_benchmark_double) {
    using namespace unc::robotics::kdtree;

    for (std::size_t N = 1000 ; N <= 1000000 ; N *= 10) {
        testBenchmarkQueries("L2Space", makeBoundedL2Space<double, 3>(), N);
        testBenchmarkQueries("SO3Space", SO3Space<double>(), N);
    }
}
