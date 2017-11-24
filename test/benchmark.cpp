#include <iostream>
#include <iomanip>
#include <ctime>
#include "../src/kdtree.hpp"
#include "state_sampler.hpp"

template <typename _State>
struct Node {
    _State key_;

    Node(const _State& key)
        : key_(key)
    {
    }
};

struct GetKey {
    template <typename _State>
    constexpr const _State& operator() (const Node<_State>& state) const {
        return state.key_;
    }
};

template <typename Duration>
void printStats(std::size_t size, Duration elapsed, unsigned queryCount, double overhead) {
    auto elapsedNanos = std::chrono::duration<long long, std::nano>(elapsed).count()
        - static_cast<long long>(queryCount * overhead);
    
    std::cout << size << "\t"
              << elapsedNanos << "\t"
              << queryCount << "\t"
              << elapsedNanos / (queryCount * 1e3) << std::endl;
}           


template <typename _Space, typename _StepDuration, typename _SplitStrategy, typename _Locking>
void benchmark(
    const std::string& name,
    const _Space& space,
    std::size_t N, std::size_t Q, std::size_t k,
    _StepDuration stepDuration,
    double stepsPerExp,
    double overhead,
    const _SplitStrategy&,
    const _Locking&)
{
    using namespace unc::robotics::kdtree;

    typedef typename _Space::Distance Distance;
    typedef typename _Space::State Key;

    constexpr std::size_t nTrees = 16;
    
    std::vector<KDTree<Node<Key>, _Space, GetKey, _SplitStrategy, DynamicBuild, _Locking>> trees;
    trees.reserve(nTrees);
    for (std::size_t i=0 ; i<nTrees ; ++i)
        trees.emplace_back(space);

    std::mt19937_64 rng;
    std::vector<Node<Key>> nodes;
    nodes.reserve(N);

    typedef std::pair<Node<Key>, Distance> NodeDist;
    std::vector<NodeDist, Eigen::aligned_allocator<NodeDist>> nearest;
    nearest.reserve(k+1);

    std::vector<Key> queries;
    queries.reserve(Q);
    for (std::size_t j=0 ; j<Q ; ++j)
        queries.emplace_back(StateSampler<_Space>::randomState(rng, space));
    
    typedef std::chrono::high_resolution_clock Clock;

    Clock::duration elapsed{};
    unsigned queryCount = 0;
    unsigned treeNo = 0;

    std::size_t nextStat = 100;
    unsigned statCounter = static_cast<unsigned>(std::log(nextStat) / std::log(10) * stepsPerExp);
    Clock::duration timePerSize = Clock::duration(stepDuration) / (nextStat - trees[0].size());
        
    for (std::size_t i=1 ; i<=N ; ++i) {
        nodes.emplace_back(StateSampler<_Space>::randomState(rng, space));
        for (std::size_t j=0 ; j<nTrees ; ++j)
            trees[j].add(nodes.back());

        Clock::duration dt;
        auto start = Clock::now();
        do {
            trees[treeNo++ % nTrees].nearest(nearest, queries[queryCount++%Q], k);
        } while ((dt = Clock::now() - start) < timePerSize);
        
        elapsed += dt;
        
        if (i >= nextStat) {
            printStats(trees[0].size(), elapsed, queryCount, overhead);
            elapsed = Clock::duration::zero();
            queryCount = 0;
            do {
                nextStat = static_cast<std::size_t>(std::pow(10, ++statCounter/stepsPerExp) + 0.5);
            } while (nextStat <= i);
            timePerSize = Clock::duration(stepDuration) / (nextStat - trees[0].size());
        }
    }

    if (queryCount > 0)
        printStats(trees[0].size(), elapsed, queryCount, overhead);
}

template <typename _Duration>
double measureNow(_Duration checkTime) {
    typedef std::chrono::high_resolution_clock Clock;
    Clock::duration clockCheckTime(checkTime);
    Clock::duration elapsed;
    unsigned count = 0;
    Clock::time_point start = Clock::now();
    do {
        ++count;
    } while ((elapsed = Clock::now() - start) < clockCheckTime);

    double elapsedNanos = std::chrono::duration<double, std::nano>(elapsed).count();
    
    std::cout << "# Clock overhead: " << elapsedNanos / 1e6 << " ms over "
              << count << " calls = " << elapsedNanos / count << " ns/call"
              << std::endl;
                 
    return elapsedNanos / count;
}

int main(int argc, char *argv[]) {
    using namespace unc::robotics::kdtree;
    using namespace std::literals::chrono_literals;

    std::size_t N = 100000;
    std::size_t Q = 10000;
    std::size_t k = 20;

    auto stepTime = 25ms;
    double overhead = measureNow(1s);
    double steps = 250;

    std::time_t tm = std::time(nullptr);
    std::cout <<
        "set title '" << std::put_time(std::localtime(&tm), "%c") << "'\n"
        "set logscale x\n"
        "set key top left\n"
        // "plot '-' u 1:4 w lines title 'L2(3) Midpoint', "
        // "     '-' u 1:4 w lines title 'L2(3) Median'\n";
        "plot '-' u 1:4 w lines title 'SO(3) Midpoint', "
        "     '-' u 1:4 w lines title 'SO(3) Median'\n";

    Eigen::Array<double, 3, 2> bounds;
    bounds.col(0) = -1.0;
    bounds.col(1) = 1.0;
    
    // benchmark("SO(3)", BoundedL2Space<double,3>(bounds), N, Q, k, stepTime, steps, overhead, MidpointSplit{}, SingleThread{});
    // std::cout << "e" << std::endl;
    // benchmark("SO(3)", BoundedL2Space<double,3>(bounds), N, Q, k, stepTime, steps, overhead, MedianSplit{}, SingleThread{});
    // std::cout << "e" << std::endl;

    benchmark("SO(3)", SO3Space<double>(), N, Q, k, stepTime, steps, overhead, MidpointSplit{}, SingleThread{});
    std::cout << "e" << std::endl;
    benchmark("SO(3)", SO3Space<double>(), N, Q, k, stepTime, steps, overhead, MedianSplit{}, SingleThread{});
    std::cout << "e" << std::endl;

    return 0;
}




