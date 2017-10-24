#include <iostream>
#include "../src/kdtree.hpp"
#include "test.hpp"
#include "state_sampler.hpp"

template <typename _State>
struct TestNode {
    _State key_;
    int name_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    TestNode(const _State& key, int name)
        : key_(key), name_(name)
    {
    }

    template <typename _Char, typename _Traits>
    friend std::basic_ostream<_Char, _Traits>& operator << (
        std::basic_ostream<_Char, _Traits>& os, const TestNode& node)
    {
        return os << "Node{" << node.name_ << "}";
    }
};

struct TestNodeKey {
    template <typename _State>
    constexpr const _State& operator() (const TestNode<_State>& state) const {
        return state.key_;
    }
};

template <typename _Scalar, int _dimensions>
unc::robotics::kdtree::BoundedL2Space<_Scalar, _dimensions>
createBoundedL2Space(int dim = _dimensions) {
    using namespace unc::robotics::kdtree;

    Eigen::Array<_Scalar, _dimensions, 2> bounds(dim, 2);
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    
    return BoundedL2Space<_Scalar, _dimensions>(bounds);
}

template <typename _Scalar>
auto createBoundedSE3Space() {
    using namespace unc::robotics::kdtree;
    return makeCompoundSpace(
        SO3Space<_Scalar>(),
        createBoundedL2Space<_Scalar, 3>());
}

template <typename _Scalar, std::intmax_t _q, std::intmax_t _t>
auto createRatioWeightedBoundedSE3Space() {
    using namespace unc::robotics::kdtree;
    return makeCompoundSpace(
        makeRatioWeightedSpace<_q>(SO3Space<_Scalar>()),
        makeRatioWeightedSpace<_t>(createBoundedL2Space<_Scalar, 3>()));
}


template <typename Space>
void testAdd(const Space& space) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;
    
    constexpr std::size_t N = 1000;
   
    KDTree<TestNode<Key>, Space, TestNodeKey, MidpointSplit> tree(space);

    std::vector<std::pair<Distance, TestNode<Key>>> nearest;
    std::mt19937_64 rng;
    std::vector<TestNode<Key>> nodes;
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        EXPECT(tree.size()) == i;
        tree.add(nodes.back());

        tree.nearest(nodes.back().key_);
        tree.nearest(nearest, nodes.back().key_, 1);
    }
    EXPECT(tree.size()) == N;
}

template <typename Space>
void testKNN(const Space& space, std::size_t N, std::size_t Q, std::size_t k) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;

    KDTree<TestNode<Key>, Space, TestNodeKey, MidpointSplit> tree(space);

    std::mt19937_64 rng;
    std::vector<TestNode<Key>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::vector<std::pair<Distance, TestNode<Key>>> nearest;
    nearest.reserve(k);
    for (std::size_t i=0 ; i<Q ; ++i) {
        auto q = StateSampler<Space>::randomState(rng, space);
        tree.nearest(nearest, q, k);

        EXPECT(nearest.size()) == k;

        std::partial_sort(nodes.begin(), nodes.begin() + k, nodes.end(), [&q, &space] (auto& a, auto& b) {
            return space.distance(q, a.key_) < space.distance(q, b.key_);
        });

        for (std::size_t j=0 ; j<k ; ++j) {
            EXPECT(nearest[j].second.name_) == nodes[j].name_;
            if (j) EXPECT(nearest[j-1].first) <= nearest[j].first;
        }
    }    
}

template <typename Space, typename _Duration>
std::pair<double, std::size_t> benchmark(
    const std::string& name,
    const Space& space,
    std::size_t N, std::size_t k,
    _Duration duration)
{
    using namespace unc::robotics::kdtree;

    typedef typename Space::Distance Distance;
    typedef typename Space::State Key;

    KDTree<TestNode<Key>, Space, TestNodeKey, MidpointSplit> tree(space);

    std::mt19937_64 rng;
    std::vector<TestNode<Key>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        tree.add(nodes.back());
    }

    typedef std::pair<Distance, TestNode<Key>> DistNode;
    std::vector<DistNode, Eigen::aligned_allocator<DistNode>> nearest;
    nearest.reserve(k);
    constexpr std::size_t batchSize = 100;
    typedef std::chrono::high_resolution_clock Clock;
    Clock::duration maxElapsed = duration;
    Clock::duration elapsed;
    std::size_t count = 0;
    Clock::time_point start = Clock::now();
    do {
        for (std::size_t i=0 ; i<batchSize ; ++i) {
            auto q = StateSampler<Space>::randomState(rng, space);
            tree.nearest(nearest, q, k);
        }
        count += batchSize;
    } while ((elapsed = Clock::now() - start) < maxElapsed);
    
    double seconds = std::chrono::duration<double>(elapsed).count();
    std::cout << name << ": " << seconds*1e6/count << " us/op" << std::endl;
    
    return std::make_pair(seconds, count);
}

TEST_CASE(Add_BoundedL2) {
    testAdd(createBoundedL2Space<double, 3>());
}
TEST_CASE(KNN_BoundedL2) {
    testKNN(createBoundedL2Space<double, 3>(), 10000, 1000, 20);
}

TEST_CASE(Add_SO3Space) {
    testAdd(unc::robotics::kdtree::SO3Space<double>());
}
TEST_CASE(KNN_SO3Space) {
    testKNN(unc::robotics::kdtree::SO3Space<double>(), 10000, 1000, 20);
}

TEST_CASE(Add_RatioWeightedSpace) {
    using namespace unc::robotics::kdtree;
    testAdd(makeRatioWeightedSpace<17,5>(SO3Space<double>()));
}

TEST_CASE(KNN_RatioWeightedSpace) {
    using namespace unc::robotics::kdtree;
    testKNN(makeRatioWeightedSpace<17,5>(SO3Space<double>()), 10000, 1000, 20);
}

TEST_CASE(Add_WeightedSpace) {
    using namespace unc::robotics::kdtree;
    testAdd(WeightedSpace<SO3Space<double>>(3.21));
}

TEST_CASE(KNN_WeightedSpace) {
    using namespace unc::robotics::kdtree;
    testKNN(WeightedSpace<SO3Space<double>>(3.21), 10000, 1000, 20);
}

TEST_CASE(Add_CompoundSpace_SE3_1to1) {
    testAdd(createBoundedSE3Space<double>());
}

TEST_CASE(KNN_CompoundSpace_SE3_1to1) {
    testKNN(createBoundedSE3Space<double>(), 10000, 1000, 20);
}

TEST_CASE(Add_CompoundSpace_SE3_5to17) {
    testAdd(createRatioWeightedBoundedSE3Space<double, 5, 27>());
}

TEST_CASE(KNN_CompoundSpace_SE3_5to17) {
    testKNN(createRatioWeightedBoundedSE3Space<double, 5, 27>(), 10000, 1000, 20);
}


TEST_CASE(benchmark) {
    using namespace unc::robotics::kdtree;
    using namespace std::literals::chrono_literals;

    std::size_t N = 100000;
    std::size_t k = 1;
    
    benchmark("R^3l2", createBoundedL2Space<double, 3>(), N, k, 1s);
    benchmark("R^6l2", createBoundedL2Space<double, 6>(), N, k, 1s);
    benchmark("SO(3)", SO3Space<double>(), N, k, 1s);
    benchmark("SE(3)", createBoundedSE3Space<double>(), N, k, 1s);
}
