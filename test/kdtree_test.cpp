#include "kdtree.hpp"
#include <vector>
#include <random>
#include <iostream>
#include "test.hpp"

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

template <typename _Space, typename _RNG>
typename _Space::State randomState(const _Space& space, _RNG& rng);


template <typename _Scalar, typename _RNG>
typename unc::robotics::kdtree::SO3Space<_Scalar>::State
randomState(const unc::robotics::kdtree::SO3Space<_Scalar>&, _RNG& rng) {
    std::uniform_real_distribution<_Scalar> dist01(0, 1);
    std::uniform_real_distribution<_Scalar> dist2pi(0, 2*M_PI);
    _Scalar a = dist01(rng);
    _Scalar b = dist2pi(rng);
    _Scalar c = dist2pi(rng);

    return Eigen::Quaternion<_Scalar>(
        std::sqrt(1-a)*std::sin(b),
        std::sqrt(1-a)*std::cos(b),
        std::sqrt(a)*std::sin(c),
        std::sqrt(a)*std::cos(c));
}

template <typename _Scalar, int _dim, typename _RNG>
typename unc::robotics::kdtree::BoundedEuclideanSpace<_Scalar, _dim>::State
randomState(const unc::robotics::kdtree::BoundedEuclideanSpace<_Scalar, _dim>& space, _RNG& rng) {
    Eigen::Matrix<_Scalar, _dim, 1> q;
    
    for (int i=0 ; i<_dim ; ++i) {
        std::uniform_real_distribution<_Scalar> dist(space.bounds(i, 0), space.bounds(i, 1));
        q[i] = dist(rng);
    }

    return q;
}

template <typename _Space, std::intmax_t _num, std::intmax_t _den, typename _RNG>
typename _Space::State
randomState(const unc::robotics::kdtree::RatioWeightedSpace<_Space, _num, _den>& space, _RNG& rng) {
    return randomState(*static_cast<const _Space*>(&space), rng);
}

template <typename _RNG, typename _Space, std::size_t ... _I>
auto randomCompoundState(const _Space& space, _RNG& rng, std::index_sequence<_I...>) {
    return typename _Space::State(randomState(space.template subspace<_I>(), rng)...);
}


template <typename _RNG, typename ... _Spaces>
typename unc::robotics::kdtree::CompoundSpace<_Spaces...>::State
randomState(const unc::robotics::kdtree::CompoundSpace<_Spaces...>& space, _RNG& rng) {
    return randomCompoundState(space, rng, std::make_index_sequence<sizeof...(_Spaces)>{});
}

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

        for (int i=0 ; i<Q ; ++i) {
            auto q = randomState(space_, rng);
            typename Space::Distance d;
            const int* index = tree.nearest(q, &d);
            EXPECT(index) != nullptr;
            std::partial_sort(linear.begin(), linear.begin()+1, linear.end(), [&] (int a, int b) {
                    return space_.distance(nodes[a].state_, q) <
                        space_.distance(nodes[b].state_, q);
                });
            EXPECT(*index) == linear[0];
        }
    }
};

template <typename _Space>
bool run(const std::string& spaceName, const _Space& space) {
    KDTreeTests<_Space> tests(space);
    bool success = true;
    success &= runTest(spaceName + ".testAdd(1000)", [&] { tests.testAdd(1000); });
    success &= runTest(spaceName + ".testNearest(1000,100)", [&] { tests.testNearest(1000, 100); });
    return success;
}    

int main(int argc, char *argv[]) {
    bool success = true;
#if 0
    success &= run("SO3Space<double>", unc::robotics::kdtree::SO3Space<double>());
#endif
    Eigen::Array<double, 4, 2> bounds4d;
    bounds4d <<
        -1.1, 1,
        -1.2, 2,
        -1.3, 3,
        -1.4, 4;
    success &= run("BoundedEuclideanSpace<double, 4>", unc::robotics::kdtree::BoundedEuclideanSpace<double, 4>(bounds4d));

    #if 0
    Eigen::Array<double, 3, 2> bounds3d(bounds4d.block<3, 2>(0,0));
    success &= run("BoundedSE3Space<double>", unc::robotics::kdtree::BoundedSE3Space<double>(
            unc::robotics::kdtree::SO3Space<double>(),
            unc::robotics::kdtree::BoundedEuclideanSpace<double, 3>(bounds3d)));
    #endif
    // KDTreeTests<unc::robotics::kdtree::SO3Space<double>> tests(space);
    // std::mt19937_64 rng;
    // auto q = randomState(space, rng);
    // std::cout << q.coeffs().transpose() << std::endl;
    return success ? 0 : 1;
}
