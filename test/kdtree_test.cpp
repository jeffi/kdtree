#include "../src/kdtree.hpp"
#include "test.hpp"
#include "state_sampler.hpp"
#include <random>

namespace test {
template <typename _Scalar>
struct PrintWrap<Eigen::Quaternion<_Scalar>> {
    const Eigen::Quaternion<_Scalar>& value_;
    PrintWrap(const Eigen::Quaternion<_Scalar>& v) : value_(v) {}
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& v) {
        // could also use std::boolalpha
        return os << v.value_.coeffs().transpose();
    }    
};

template <typename _Scalar, int _dim>
struct PrintWrap<Eigen::Matrix<_Scalar, _dim, 1>> {
    const Eigen::Matrix<_Scalar, _dim, 1>& value_;
    PrintWrap(const Eigen::Matrix<_Scalar, _dim, 1>& v) : value_(v) {}
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& v) {
        // could also use std::boolalpha
        return os << v.value_.transpose();
    }    
};
}

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

template <typename _Scalar, int _dim, std::intmax_t _num, std::intmax_t _den>
unc::robotics::kdtree::BoundedL2Space<_Scalar, _dim> makeRatioWeightedBoundedL2Space() {
    using namespace unc::robotics::kdtree;
    return RatioWeightedSpace<BoundedL2Space<_Scalar, _dim>, _num, _den>(
        makeBoundedL2Space<_Scalar, _dim>());
}

// template <typename _Scalar, std::intmax_t _qWeight, std::intmax_t _tWeight>
// unc::robotics::kdtree::BoundedSE3Space<_Scalar, _qWeight, _tWeight> makeBoundedSE3Space() {
//     using namespace unc::robotics::kdtree;
//     return BoundedSE3Space<_Scalar, _qWeight, _tWeight>(
//         SO3Space<_Scalar>(),
//         makeBoundedL2Space<_Scalar, 3>());
// }

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
// unc::robotics::kdtree::CompoundSpace<
//     unc::robotics::kdtree::SO3Space<_Scalar>,
//     unc::robotics::kdtree::BoundedL2Space<_Scalar, 3>>
unc::robotics::kdtree::BoundedSE3Space<_Scalar, _qWeight, _tWeight>
makeBoundedSE3Space() {
    using namespace unc::robotics::kdtree;
    return BoundedSE3Space<_Scalar, _qWeight, _tWeight>(
        // return CompoundSpace<SO3Space<_Scalar>, BoundedL2Space<_Scalar, 3>>(
        SO3Space<_Scalar>(),
        makeBoundedL2Space<_Scalar, 3>());
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
        State q = StateSampler<Space>::randomState(rng, space);
        Distance zero = space.distance(q, q);
        tree.add(TestNode<State>(q, i));

        EXPECT(tree.size()) == i+1;
        EXPECT(tree.empty()) == false;
        int minDepth = std::floor(std::log(i+1) / std::log(2));
        // int maxDepth = std::ceil(minDepth*(1 + std::log(2)));
        EXPECT(tree.depth()) >= minDepth;
        EXPECT(tree.depth()) <= 3 * (minDepth+1);

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

TEST_CASE(KDTree_RatioWeightedRV3_add_double) {
    testAdd(makeRatioWeightedBoundedL2Space<double, 3, 11, 3>());
}

TEST_CASE(KDTree_SE3_add_double) {
    testAdd(makeBoundedSE3Space<double>());
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
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(k);
    for (std::size_t i=0 ; i<Q ; ++i) {
        auto q = StateSampler<Space>::randomState(rng, space);
        tree.nearest(nearest, q, k);

        EXPECT(nearest.size()) == k;
        
        std::partial_sort(nodes.begin(), nodes.begin() + k, nodes.end(), [&q, &space] (auto& a, auto& b) {
            return space.distance(q, a.state_) < space.distance(q, b.state_);
        });

        // for (std::size_t j=0 ; j<k ; ++j) {
        //     std::cout << i << ", " << j << ": got " << nearest[j].first << ", expected "
        //               << space.distance(q, nodes[j].state_) << std::endl;
        // }
        for (std::size_t j=0 ; j<k ; ++j) {
            if (nearest[j].second.name_ != nodes[j].name_) {
                std::cout << "key: " << test::PrintWrap<State>(q) << "\n"
                          << "got: " << test::PrintWrap<State>(nearest[j].second.state_) << "\n"
                          << "exp: " << test::PrintWrap<State>(nodes[j].state_) << std::endl;
            }
            EXPECT(nearest[j].second.name_) == nodes[j].name_;
            if (j) EXPECT(nearest[j-1].first) <= nearest[j].first;
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

TEST_CASE(KDTree_RatioWeightedRV3_nearestK_double) {
    testKNN(makeRatioWeightedBoundedL2Space<double, 3, 17, 5>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SE3_nearestK_double) {
    testKNN(makeBoundedSE3Space<double>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SE3_1_10_nearestK_double) {
    testKNN(makeBoundedSE3Space<double,1,10>(), 5000, 500, 20);
}

TEST_CASE(KDTree_SE3_10_1_nearestK_double) {
    testKNN(makeBoundedSE3Space<double,10,1>(), 5000, 500, 20);
}

template <typename Space, typename Refs>
static void testRemove(const Space& space, std::size_t N, const Refs&) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDTree<TestNode<State>*, Space, TestNodeKey, Refs> tree(TestNodeKey(), space);

    std::mt19937_64 rng;
    std::vector<TestNode<State>> nodes;
    std::vector<std::size_t> removeOrder;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i) {
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        removeOrder.push_back(i);
    }

    // std::vector<std::size_t> removeOrder;
    // std::generate_n(std::back_inserter(removeOrder), N, [&] { return removeOrder.size(); });

    for (std::size_t i=0 ; i<N ; ++i)
        tree.add(&nodes[i]);

    std::shuffle(removeOrder.begin(), removeOrder.end(), rng);

    EXPECT(nodes.size()) == N;
    
    std::vector<std::pair<Distance, TestNode<State>*>> nearest;
    nearest.reserve(N);
    for (std::size_t i=N ; i-->0 ; ) {
        // first attempt should remove it, and return true
        EXPECT(tree.remove(&nodes[removeOrder[i]])) == true;

        // if removal was successful, then the size should be decreased by 1
        EXPECT(tree.size()) == i;

        // second attempte should find that it was already removed, and return false
        EXPECT(tree.remove(&nodes[removeOrder[i]])) == false;

        // now check that a nearest query (for all elements) does not
        // return any of the removed elements.
        auto q = StateSampler<Space>::randomState(rng, space);
        tree.nearest(nearest, q, N);

        EXPECT(nearest.size()) == i;
        
        int n = nodes[removeOrder[i]].name_;
        bool found = std::any_of(nearest.begin(), nearest.end(), [n] (auto& p) { return p.second->name_ == n; });
        EXPECT(found) == false;
    }

    // TODO: need to check remove->add since we reuse nodes internally
    
    // TODO: test when the same node is added twice, and removed once, then again.
    // TODO: do should we detect and reject the same node being added twice?
    
    // TODO: remove should also blank out the value if possible for
    // when _T is shared_ptr or something in which its destruction
    // will have important side-effects.
}

TEST_CASE(KDTree_SE3_remove_double) {
    using namespace unc::robotics::kdtree;
    testRemove(makeBoundedSE3Space<double>(), 5000, PointerReferences{});
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
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::size_t total = 0;
    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(N);
    for (std::size_t i=0 ; i<Q ; ++i) {
        auto q = StateSampler<Space>::randomState(rng, space);
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

        for (std::size_t j=0 ; j<nearest.size() ; ++j) {
            EXPECT(nearest[j].second.name_) == nodes[j].name_;
            if (j) EXPECT(nearest[j-1].first) <= nearest[j].first;
        }

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
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);
        tree.add(nodes.back());
    }

    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(k);
    std::chrono::high_resolution_clock::duration maxElapsed = maxDuration;
    std::size_t count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (;;) {
        for (std::size_t i=0 ; i<100 ; ++i) {
            auto q = StateSampler<Space>::randomState(rng, space);
            tree.nearest(nearest, q, k);
        }
        count += 100;
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        if (elapsed > maxElapsed)
            return std::make_pair(count, std::chrono::duration<double>(elapsed).count());
    }
}

template <typename Space, typename _Duration>
static std::pair<std::size_t, double>
benchmarkKNNLinear(const Space& space, std::size_t N, std::size_t k, _Duration maxDuration) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);
    
    std::mt19937_64 rng;
    std::vector<TestNode<State>> nodes;
    nodes.reserve(N);
    for (std::size_t i=0 ; i<N ; ++i)
        nodes.emplace_back(StateSampler<Space>::randomState(rng, space), i);

    std::vector<std::pair<Distance, TestNode<State>>> nearest;
    nearest.reserve(k);
    std::chrono::high_resolution_clock::duration maxElapsed = maxDuration;
    std::size_t count = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (;;) {
        for (std::size_t i=0 ; i<100 ; ++i) {
            auto q = StateSampler<Space>::randomState(rng, space);
            std::partial_sort(nodes.begin(), nodes.begin() + k, nodes.end(), [&q, &space] (auto& a, auto& b) {
                return space.distance(q, a.state_) < space.distance(q, b.state_);
            });
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

template <typename _Scalar, int _dim, typename _Duration>
static std::pair<std::size_t, double>
benchmarkKNNL2Linear(std::size_t N, std::size_t k, _Duration maxDuration) {
    using namespace unc::robotics::kdtree;

    typedef BoundedL2Space<_Scalar, _dim> Space;

    Eigen::Array<_Scalar, _dim, 2> bounds;
    bounds.col(0) = -1;
    bounds.col(1) = 1;
    Space space((bounds));

    return benchmarkKNNLinear(space, N, k, maxDuration);
}

TEST_CASE(benchmark) {
    using namespace unc::robotics::kdtree;
    using namespace std::literals::chrono_literals;

    std::size_t N = 50000;
    std::size_t k = 1;
    
    auto result = benchmarkKNNL2<double, 3>(N, k, 1s);
    std::cout << "L2Space<double,3>()        "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNL2Linear<double, 3>(N, k, 1s);
    std::cout << "L2Space<double,3>() Linear "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNL2<double, 6>(N, k, 1s);
    std::cout << "L2Space<double,6>()        "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNN(SO3Space<double>(), N, k, 1s);
    std::cout << "SO3Space<double>()         "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNLinear(SO3Space<double>(), N, k, 1s);
    std::cout << "SO3Space<double>() Linear  "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNN(BoundedSE3Space<double, 100, 1>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,100,1>()   "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNLinear(BoundedSE3Space<double, 100, 1>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,100,1>() L "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNN(BoundedSE3Space<double>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,1,1>()     "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNLinear(BoundedSE3Space<double>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,1,1>() Lin "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNN(BoundedSE3Space<double, 1, 100>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,1,100>()   "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

    result = benchmarkKNNLinear(BoundedSE3Space<double, 1, 100>(SO3Space<double>(), makeBoundedL2Space<double,3>()), N, k, 1s);
    std::cout << "SE3Space<double,1,100>() L "
              << result.first << " queries in " << result.second << " s = "
              << result.second * 1e6 / result.first << " us/q" << std::endl;

}

