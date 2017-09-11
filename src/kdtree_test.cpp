#include <iostream>
#include <iomanip>
#include "kdtree.hpp"
#include <vector>
#include <random>

template <typename _T>
class Expectation {
    const _T& value_;
    const char *file_;
    int line_;
    bool checked_;
    
public:
    Expectation(const _T& v, const char *file, int line)
        : value_(v), file_(file), line_(line), checked_(false)
    {
    }
    ~Expectation() {
        if (!checked_) {
            std::cout << "Assertion not checked"
                      << " at " << file_ << ":" << line_
                      << std::endl;
        }
    }

    template <typename _A>
    void operator == (const _A& actual) {
        if (!(value_ == actual)) {
            std::cout << "Assertion failed.  Expected: " << value_
                      << ", actual: " << actual
                      << " at " << file_ << ":" << line_
                      << std::endl;
        }
        checked_ = true;
    }
};

template <typename _T>
Expectation<_T> expect(const _T& x, const char *file, int line) {
    return Expectation<_T>(x, file, line);
}

#define EXPECT(x) expect(x, __FILE__, __LINE__)

template <typename _Scalar, typename _RNG>
Eigen::Quaternion<_Scalar> randomQuaternion(_RNG& rng) {
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

template <typename _Scalar, typename _RNG, int _rows>
Eigen::Matrix<_Scalar, _rows, 1> randomVector(_RNG& rng, const Eigen::Matrix<_Scalar, _rows, 2>& bounds) {
    Eigen::Matrix<_Scalar, _rows, 1> v;

    for (int i=0 ; i<_rows ; ++i) {
        std::uniform_real_distribution<_Scalar> dist(bounds(i,0), bounds(i,1));
        v[i] = dist(rng);
    }

    return v;
}

template <typename _Space>
struct Node {
    typename _Space::State q_;
    
    template <typename ... _Args>
    Node(_Args&& ... args) : q_(std::forward<_Args>(args)...) {}
};

struct NodeKey {
    template <typename _Space>
    const typename _Space::State& operator() (const Node<_Space>& n) const {
        return n.q_;
    }
};

void test_euclidean() {
    using namespace unc::robotics::kdtree;

    typedef BoundedEuclideanSpace<double, 4> Space;

    Eigen::Matrix<double, 4, 2> b;
    b << -1, 1,
        -2, 2,
        -3, 3,
        -4, 4;

    Space space(b);
    KDTree<Node<Space>, Space, NodeKey> tree(NodeKey(), space);

    std::mt19937_64 rng;
    constexpr int N = 1000;
    std::vector<Node<Space>> nodes;
    nodes.reserve(N);
    for (int i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomVector(rng, b));
        tree.add(nodes.back());
    }
    std::cout << "Euclidean depth = " << tree.depth() << std::endl;

}

void test_so3() {
    using namespace unc::robotics::kdtree;

    typedef SO3Space<long double> Space;

    KDTree<Node<Space>, Space, NodeKey> tree((NodeKey()));

    tree.add(Node<Space>(0, 0, 0, 1));
        
    std::mt19937_64 rng;
    constexpr int N = 1000;
    std::vector<Node<Space>> nodes;
    nodes.reserve(N);
    for (int i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomQuaternion<long double>(rng));
        tree.add(nodes.back());
    }
    std::cout << "SO3 depth = " << tree.depth() << std::endl;
    std::cout << "Expected depth: " << std::log(N) / std::log(2) << std::endl;
}

void test_se3() {
    using namespace unc::robotics::kdtree;

    typedef BoundedSE3Space<float> Space;

    Eigen::Matrix<float, 3, 2> b;
    b << -1.1f, 1.1f,
        -2.2f, 2.2f,
        -3.3f, 3.3f;


    // Space space((SO3Space<float>()), BoundedEuclideanSpace<float, 3>(b));
    Space space((SO3Space<float>()), b);
    KDTree<Node<Space>, Space, NodeKey> tree(NodeKey(), space);

    std::mt19937_64 rng;
    constexpr int N = 1000;
    std::vector<Node<Space>> nodes;
    nodes.reserve(N);
    for (int i=0 ; i<N ; ++i) {
        nodes.emplace_back(randomQuaternion<float>(rng),
                           randomVector(rng, b));
        tree.add(nodes.back());
    }

    std::cout << "SE3 depth = " << tree.depth() << std::endl;
}

void test_twose3() {
    using namespace unc::robotics::kdtree;

    typedef BoundedSE3Space<float> Space1;
    typedef BoundedSE3Space<double> Space2;

    typedef CompoundSpace<Space1, Space2> Space;

    Eigen::Matrix<float, 3, 2> b1;
    b1 << 0, 1, 2, 3, 4, 5;
    Eigen::Matrix<double, 3, 2> b2;
    b2 << 0, .1, .2, .3, .4, .5;

    Space space((Space1((SO3Space<float>()), b1)),
                (Space2((SO3Space<double>()), b2)));
    KDTree<Node<Space>, Space, NodeKey> tree(NodeKey(), space);

    std::mt19937_64 rng;
    constexpr int N = 1000;
    std::vector<Node<Space>> nodes;
    nodes.reserve(N);
    for (int i=0 ; i<N ; ++i) {
        nodes.emplace_back(Space1::State(randomQuaternion<float>(rng),
                                         randomVector(rng, b1)),
                           Space2::State(randomQuaternion<double>(rng),
                                         randomVector(rng, b2)));
        tree.add(nodes.back());
    }

    std::cout << "SE3x2 depth = " << tree.depth() << std::endl;
}

void test_twoso3() {
    using namespace unc::robotics::kdtree;

    typedef SO3Space<float> Space1;
    typedef SO3Space<double> Space2;

    typedef CompoundSpace<Space1, Space2> Space;

    struct Node {
        Space::State q_;
    };

    struct NodeKey {
        const Space::State& operator() (const Node& n) const {
            return n.q_;
        }
    };

    KDTree<Node, Space, NodeKey> tree((NodeKey()));
}



int main(int argc, char *argv[]) {
    test_euclidean();
    test_so3();
    test_se3();
    test_twose3();
    //test_twoso3();
    std::cout << "done" << std::endl;
    return 0;
}
