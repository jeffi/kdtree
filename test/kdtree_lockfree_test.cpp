#include "../src/kdtree.hpp"
#include "test.hpp"
#include "state_sampler.hpp"
#include <random>
#include <thread>
#include <list>

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

class CountDownLatch {
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned count_;

public:
    explicit CountDownLatch(unsigned count) : count_(count) {}
    void countDown() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--count_) {
            cv_.wait(lock, [this] { return !count_; });
        } else {
            cv_.notify_all();
        }
    }
};


template <typename Space>
static void testParallelUpdates(const Space& space, unsigned nThreads) {
    using namespace unc::robotics::kdtree;

    typedef typename Space::State State;
    // typedef typename Space::Distance Distance;

    KDTree<TestNode<State>, Space, TestNodeKey, LockFreePointerReferences> tree(TestNodeKey(), space);
    
    constexpr std::size_t N = 100000;
    
    std::vector<std::thread> threads;
    std::list<std::vector<TestNode<State>>> nodeLists;
    CountDownLatch startLatch(nThreads);
    
    for (unsigned tNo=1 ; tNo <= nThreads ; ++tNo) {
        nodeLists.emplace_back();
        std::vector<TestNode<State>>* nodes = &nodeLists.back();
        threads.emplace_back([&, tNo, nodes] {
            std::mt19937_64 rng(tNo * 104729);
            rng.discard(1000000);

            startLatch.countDown();
            
            for (std::size_t i=0 ; i<N ; ++i) {
                State q = StateSampler<Space>::randomState(rng, space);
                nodes->emplace_back(q, tNo * N + i);
                tree.add(nodes->back());

                const TestNode<State>* nearest = tree.nearest(q);
                EXPECT(nearest) != nullptr;
                EXPECT(nearest->name_) == tNo * N + i;
            }
        });
    }

    for (auto&& t : threads)
        t.join();

    // check that all nodes are there
    for (auto& nodes : nodeLists) {
        for (auto& node : nodes) {
            const TestNode<State>* nearest = tree.nearest(node.state_);
            EXPECT(nearest) != nullptr;
            EXPECT(nearest->name_) == node.name_;
        }
    }

    EXPECT(tree.size()) == N*nThreads;
}

TEST_CASE(KDTree_ParallelUpdates) {
    unsigned nThreads = std::thread::hardware_concurrency();
    std::cout << "using " << nThreads << " threads" << std::endl;

    for (int i=0 ; i<100 ; ++i) {
        std::cout << "iteration " << i << std::endl;
        testParallelUpdates(makeBoundedL2Space<double, 3>(), nThreads);
    }
}
