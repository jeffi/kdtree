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
        State q = randomState(rng, space);
        tree.add(TestNode<State>(q, i));

        EXPECT(tree.size()) == i+1;
        EXPECT(tree.empty()) == false;
        int minDepth = std::floor(std::log(i+1) / std::log(2) + 1);
        int maxDepth = std::ceil(minDepth*(1 + std::log(2)));
        EXPECT(tree.depth()) >= minDepth;
        EXPECT(tree.depth()) <= maxDepth;

        Distance dist;
        const TestNode<State>* nearest = tree.nearest(q, &dist);
        EXPECT(nearest) != nullptr;
        EXPECT(nearest->name_) == i;
        EXPECT(dist) == 0;

        tree.nearest(nearestK, q, 1);
        EXPECT(nearestK.size()) == 1;
        EXPECT(nearestK[0].first) == 0;
        EXPECT(nearestK[0].second.name_) == i;
    }
}

template <typename _Scalar, int _dim>
static void testAddL2() {
    using namespace unc::robotics::kdtree;

    typedef BoundedL2Space<_Scalar, _dim> Space;

    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    Space space((bounds));

    testAdd(space);
}

TEST_CASE(KDTree_add_double) {
    testAddL2<double, 3>();
}

TEST_CASE(KDTree_add_float) {
    testAddL2<float, 3>();
}

TEST_CASE(KDTree_add_long_double) {
    testAddL2<long double, 3>();
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


template <typename _Scalar, int _dim>
static void testKNNL2(std::size_t N, std::size_t Q, std::size_t k) {
    using namespace unc::robotics::kdtree;

    typedef BoundedL2Space<_Scalar, _dim> Space;

    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    Space space((bounds));

    testKNN(space, N, Q, k);
}

TEST_CASE(KDTree_nearestK_float) {
    testKNNL2<float, 3>(5000, 500, 20);
}

TEST_CASE(KDTree_nearestK_double) {
    testKNNL2<double, 3>(5000, 500, 20);
}

TEST_CASE(KDTree_nearestK_long_double) {
    testKNNL2<long double, 3>(5000, 500, 20);
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
    using namespace std::literals::chrono_literals;
    auto result = benchmarkKNNL2<double, 3>(50000, 20, 1s);
    std::cout << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;
}
