#include <iostream>
#include "../src/kdtree_multi_static.hpp"
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

    KDMultiStaticTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);

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


TEST_CASE(KDMultiStaticTree_RV3_add_double) {
    testAdd(makeBoundedL2Space<double, 3>());
}

template <typename Space>
static void testKNN(const Space& space, std::size_t N, std::size_t Q, std::size_t k) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    KDMultiStaticTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space);
    
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
