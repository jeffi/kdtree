#include "kdtree.hpp"
#include "random_state.hpp"
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

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

    std::cout << name << ": " << elapsed/Q << " us/query (elapsed " << elapsed/1000 << " ms)" << std::endl;
}

int main(int argc, char *argv[]) {

    using namespace unc::robotics::kdtree;
    constexpr int N = 100000;
    constexpr int Q = 10000;
    
    benchmark("SO3Space<double>", SO3Space<double>(), N, Q);
    
    return 0;
}
