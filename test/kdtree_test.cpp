#include <iostream>
#include "kdtree.hpp"
#include <vector>
#include <random>
#include "test.hpp"
#include <chrono>
#include <iomanip>
#include "random_state.hpp"

template <typename _T>
struct StatePrinter;

template <typename _State>
StatePrinter<_State> statePrinter(const _State& q) {
    return StatePrinter<_State>(q);
}

template <typename _Scalar, int _rows>
struct StatePrinter<Eigen::Matrix<_Scalar, _rows, 1>> {
    const Eigen::Matrix<_Scalar, _rows, 1>& value_;
    StatePrinter(const Eigen::Matrix<_Scalar, _rows, 1>& v) : value_(v) {}

    template <typename _Char, typename _Traits>
    friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const StatePrinter& q) {
        return os << q.value_.transpose();
    }
};

template <typename _Scalar>
struct StatePrinter<Eigen::Quaternion<_Scalar>> {
    const Eigen::Quaternion<_Scalar>& value_;
    StatePrinter(const Eigen::Quaternion<_Scalar>& v) : value_(v) {}

    template <typename _Char, typename _Traits>
    friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const StatePrinter& q) {
        return os << q.value_.coeffs().transpose();
    }
};

template <typename ... _States>
struct StatePrinter<unc::robotics::kdtree::CompoundState<_States...>> {
    typedef unc::robotics::kdtree::CompoundState<_States...> State;
    
    const State& value_;

    StatePrinter(const State& v) : value_(v) {}

    template <typename ... _T>
    static void ignore(const _T& ... arg) {}
    
    template <typename _Char, typename _Traits, std::size_t ... I>
    void printTo(std::basic_ostream<_Char, _Traits>& os, std::index_sequence<I...>) const {
        ignore(((I ? os << ", " : os) << statePrinter(value_.template substate<I>()))...);
    }

    template <typename _Char, typename _Traits>
    friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const StatePrinter& q) {
        q.printTo(os, std::make_index_sequence<sizeof...(_States)>{});
        return os;
    }
};


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

template <typename _State>
struct TestIndexKey {
    const std::vector<TestNode<_State>>& nodes_;
    TestIndexKey(const std::vector<TestNode<_State>>& nodes) : nodes_(nodes) {}
    inline const _State& operator() (int index) const {
        return nodes_[index].state_;
    }
};



template <typename _Space>
struct KDTreeTests {
    typedef _Space Space;
    typedef typename Space::State State;
    
    Space space_;

    KDTreeTests(const _Space& space)
        : space_(space)
    {
    }

    void testAdd(int N) const {
        unc::robotics::kdtree::KDTree<TestNode<State>, Space, TestNodeKey> tree(TestNodeKey(), space_);

        EXPECT(tree.size()) == 0;
        EXPECT(tree.empty()) == true;
        
        std::mt19937_64 rng;
        for (int i=0 ; i<N ; ++i) {
            TestNode<State> n(randomState(space_, rng), i);
            tree.add(n);
            EXPECT(tree.size()) == i+1;
            EXPECT(tree.empty()) == false;
            unsigned depth = std::floor(std::log(i+1.0) / std::log(2.0) + 1);
            EXPECT(tree.depth()) >= depth;
            EXPECT(tree.depth()) <= depth*(1 + std::log(2.0));
        }
    }

    void testNearest(int N, int Q) const {
        std::vector<TestNode<State>> nodes;
        std::vector<int> linear;
        nodes.reserve(N);
        linear.reserve(N);
        unc::robotics::kdtree::KDTree<int, Space, TestIndexKey<State>> tree(TestIndexKey<State>(nodes), space_);
        std::mt19937_64 rng;
        for (int i=0 ; i<N ; ++i) {
            nodes.emplace_back(randomState(space_, rng), i);
            tree.add(i);
            linear.push_back(i);
        }

        // int no = 0;
        // tree.visit([&no] (int index, unsigned depth) {
        //         std::cout << no++ << ": ";
        //         for (unsigned i=0 ; i<depth ; ++i)
        //             std::cout << "  ";
        //         std::cout << index << std::endl;
        //     });

        for (int i=0 ; i<Q ; ++i) {
            auto q = randomState(space_, rng);
            typename Space::Distance d;
            const int* index = tree.nearest(q, &d);
            EXPECT(index) != nullptr;
            std::partial_sort(linear.begin(), linear.begin()+1, linear.end(), [&] (int a, int b) {
                    return space_.distance(nodes[a].state_, q) <
                           space_.distance(nodes[b].state_, q);
                });
            // std::cout << "linear: " << space_.distance(nodes[linear[0]].state_, q) << std::endl;
            EXPECT(*index) == linear[0];
            EXPECT(d) == space_.distance(nodes[linear[0]].state_, q);
        }
    }

    void testBenchmark(int N, int Q) const {
        std::vector<TestNode<State>> nodes;
        std::vector<int> linear;
        nodes.reserve(N);
        linear.reserve(N);
        unc::robotics::kdtree::KDTree<int, Space, TestIndexKey<State>> tree(TestIndexKey<State>(nodes), space_);

        std::mt19937_64 rng;
        for (int i=0 ; i<N ; ++i) {
            nodes.emplace_back(randomState(space_, rng), i);
            tree.add(i);
            linear.push_back(i);
        }

        std::vector<State> queries;
        queries.reserve(N);
        for (int i=0 ; i<Q ; ++i)
            queries.push_back(randomState(space_, rng));

        std::vector<int> linearResults;
        std::vector<int> kdtreeResults;

        typedef std::chrono::high_resolution_clock Clock;

        Clock::time_point start = Clock::now();
        
        for (int i=0 ; i<Q ; ++i)
            kdtreeResults.push_back(*tree.nearest(queries[i]));

        Clock::time_point mid = Clock::now();

        for (int i=0 ; i<Q ; ++i) {
            auto& q = queries[i];
            std::partial_sort(linear.begin(), linear.begin()+1, linear.end(), [&] (int a, int b) {
                    return space_.distance(nodes[a].state_, q) <
                           space_.distance(nodes[b].state_, q);
                });
            linearResults.push_back(linear[0]);
        }

        Clock::time_point end = Clock::now();

        double kdtreeElapsed = std::chrono::duration<double, std::micro>(mid - start).count();
        double linearElapsed = std::chrono::duration<double, std::micro>(end - mid).count();

        std::cout << std::fixed << std::setprecision(2)
                  << "Linear: " << std::setw(6) << linearElapsed/Q << " us/query" << std::endl
                  << "KDTree: " << std::setw(6) << kdtreeElapsed/Q << " us/query ("
                  << kdtreeElapsed * 100 / linearElapsed << "%)" << std::endl;

        EXPECT(kdtreeElapsed) < linearElapsed;
    }
};

template <typename _Space>
bool run(const std::string& spaceName, const _Space& space) {
    KDTreeTests<_Space> tests(space);
    bool success = true;
    success &= runTest(spaceName + "::testAdd(1000)", [&] { tests.testAdd(1000); });
    success &= runTest(spaceName + "::testNearest(1000,100)", [&] { tests.testNearest(1000, 100); });
    success &= runTest(spaceName + "::testBenchmark(10000,100)", [&] { tests.testBenchmark(10000, 100); });
    return success;
}    

int main(int argc, char *argv[]) {
    using namespace unc::robotics::kdtree;
    
    bool success = true;

    success &= run("SO3Space<double>", SO3Space<double>());

    Eigen::Array<double, 4, 2> bounds4d;
    bounds4d <<
        -1.1, 1,
        -1.2, 2,
        -1.3, 3,
        -1.4, 4;
    success &= run("BoundedEuclideanSpace<double, 4>", BoundedEuclideanSpace<double, 4>(bounds4d));
    Eigen::Array<double, 6, 2> bounds6d;
    bounds6d.col(0) = -1;
    bounds6d.col(1) = 1;

    success &= run("BoundedEuclideanSpace<double, 6>", BoundedEuclideanSpace<double, 6>(bounds6d));

    Eigen::Array<double, 3, 2> bounds3d(bounds4d.block<3, 2>(0,0));
    success &= run("BoundedSE3Space<double>", BoundedSE3Space<double>(
                       SO3Space<double>(),
                       BoundedEuclideanSpace<double, 3>(bounds3d)));

    // KDTreeTests<unc::robotics::kdtree::SO3Space<double>> tests(space);
    // std::mt19937_64 rng;
    // auto q = randomState(space, rng);
    // std::cout << q.coeffs().transpose() << std::endl;

    return success ? 0 : 1;
}
