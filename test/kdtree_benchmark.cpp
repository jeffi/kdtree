#include <iostream>
#include "kdtree.hpp"
#include "se3kdtree.hpp"
#include "random_state.hpp"
#include <chrono>
#include <random>
#include <vector>


template <typename _State>
struct IndexKey {
    const std::vector<_State>& states_;
    IndexKey(const std::vector<_State>& states) : states_(states) {}
    inline const _State& operator() (int index) const { return states_[index]; }
};

template <typename Space>
void benchmark(const std::string& name, const Space& space, int N, int Q) {
    typedef typename Space::State State;

    std::vector<State> nodes;
    nodes.reserve(N);
    unc::robotics::kdtree::KDTree<int, Space, IndexKey<State>> tree(IndexKey<State>(nodes), space);
    std::mt19937_64 rng;
    for (int i=0 ; i<N ; ++i) {
        nodes.push_back(randomState(space, rng));
        tree.add(i);
    }

    std::vector<State> queries;
    queries.reserve(N);
    for (int i=0 ; i<Q ; ++i)
        queries.push_back(randomState(space, rng));


    typedef std::chrono::high_resolution_clock Clock;
    std::vector<int> results;
    results.reserve(Q);

    auto start = Clock::now();
    for (int i=0 ; i<Q ; ++i)
        results.push_back(*tree.nearest(queries[i]));
    auto end = Clock::now();

    double elapsed = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << name << " (depth=" << tree.depth() << "): "
              << elapsed/Q << " us/query (elapsed " << elapsed/1000 << " ms)" << std::endl;
}

template <typename _Scalar>
void benchmarkSE3(const std::string& name, const Eigen::Array<_Scalar, 3, 2>& bounds, int N, int Q) {
    using namespace unc::robotics::kdtree;
    
    typedef BoundedSE3Space<_Scalar> Space;
    typedef typename Space::State State;

    Space space(
        (SO3Space<_Scalar>()),
        (BoundedEuclideanSpace<_Scalar, 3>(bounds)));
    
    std::vector<State> nodes;
    std::vector<int> linear;
    nodes.reserve(N);
    SE3KDTree<int, _Scalar, IndexKey<State>> tree(IndexKey<State>(nodes), bounds);
    std::mt19937_64 rng;
    for (int i=0 ; i<N ; ++i) {
        nodes.push_back(randomState(space, rng));
        tree.add(i);
        linear.push_back(i);
    }

    std::vector<State> queries;
    queries.reserve(N);
    for (int i=0 ; i<Q ; ++i)
        queries.push_back(randomState(space, rng));


    typedef std::chrono::high_resolution_clock Clock;
    std::vector<int> results;
    results.reserve(Q);

    auto start = Clock::now();
    for (int i=0 ; i<Q ; ++i)
        results.push_back(*tree.nearest(queries[i]));
    auto end = Clock::now();

    for (int i=0 ; i<Q ; ++i) {
        const State& q = queries[i];
        int min = *std::min_element(linear.begin(), linear.end(), [&] (int a, int b) {
            return space.distance(nodes[a], q) < space.distance(nodes[b], q);
        });
        if (min != results[i])
            std::cout << "mismatch at " << i << ": "
                      << space.distance(nodes[min], q)
                      << " vs "
                      << space.distance(nodes[results[i]], q)
                      << "\n  q: " << q.template substate<0>().coeffs().transpose()
            
                      << "\n  e: " << nodes[min].template substate<0>().coeffs().transpose()
                      << "\n   "
                      << ": " << std::atan2(nodes[min].template substate<0>().coeffs()[1],
                                            nodes[min].template substate<0>().coeffs()[0])
                      << ", " << std::atan2(nodes[min].template substate<0>().coeffs()[2],
                                            nodes[min].template substate<0>().coeffs()[0])
                      << ", " << std::atan2(nodes[min].template substate<0>().coeffs()[3],
                                            nodes[min].template substate<0>().coeffs()[0])
            
                      << "\n  a: " << nodes[results[i]].template substate<0>().coeffs().transpose()
                      << "\n   "
                      << ": " << std::atan2(nodes[results[i]].template substate<0>().coeffs()[1],
                                            nodes[results[i]].template substate<0>().coeffs()[0])
                      << ", " << std::atan2(nodes[results[i]].template substate<0>().coeffs()[2],
                                            nodes[results[i]].template substate<0>().coeffs()[0])
                      << ", " << std::atan2(nodes[results[i]].template substate<0>().coeffs()[3],
                                            nodes[results[i]].template substate<0>().coeffs()[0])

                      << std::endl;
    }
    auto linEnd = Clock::now();
    
    double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
    double linElapsed = std::chrono::duration<double, std::micro>(linEnd - end).count();

    std::cout << name << " (depth=" << tree.depth() << "): "
              << elapsed/Q << " us/query (elapsed " << elapsed/1000 << " ms)" << std::endl;
    std::cout << "linear check: "
              << linElapsed/Q << " us/query (elapsed " << linElapsed/1000 << " ms)" << std::endl;


}


int main(int argc, char *argv[]) {

    using namespace unc::robotics::kdtree;
    constexpr int N = 100000;

    Eigen::Array<double, 3, 2> bounds;
    
    // benchmark("SO3Space<double>", SO3Space<double>(), N, Q);

    bounds.col(0) = -0.01;
    bounds.col(1) = 0.01;
    benchmark("SE3Space<double>(-0.01,0.01)", BoundedSE3Space<double>(
                  SO3Space<double>(),
                  BoundedEuclideanSpace<double, 3>(bounds)), N, 1000);
    benchmarkSE3<double>("SE3KDTree<double>(-0.01, 0.01)", bounds, N, 1000);

    bounds.col(0) = -1;
    bounds.col(1) = 1;
    benchmark("SE3Space<double>(-1,1)", BoundedSE3Space<double>(
                  SO3Space<double>(),
                  BoundedEuclideanSpace<double, 3>(bounds)), N, 100);
    benchmarkSE3<double>("SE3KDTree<double>(-1, 1)", bounds, N, 100);

    bounds.col(0) = -100;
    bounds.col(1) = 100;
    benchmark("SE3Space<double>(-100,100)", BoundedSE3Space<double>(
                  SO3Space<double>(),
                  BoundedEuclideanSpace<double, 3>(bounds)), N, 1000);
    benchmarkSE3<double>("SE3KDTree<double>(-100, 100)", bounds, N, 1000);


    bounds.col(0) = -1e6;
    bounds.col(1) = 1e6;
    benchmark("SE3Space<double>(-1e6,1e6)", BoundedSE3Space<double>(
                  SO3Space<double>(),
                  BoundedEuclideanSpace<double, 3>(bounds)), N, 1000);
    benchmarkSE3<double>("SE3KDTree<double>(-1e6, 1e6)", bounds, N, 1000);

    return 0;
}
