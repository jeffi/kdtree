#pragma once
#ifndef UNC_KDTREE_HPP
#define UNC_KDTREE_HPP

#include <ostream>
#include <array>
#include <atomic>
#include <Eigen/Dense>

namespace unc {
namespace robotics {
namespace kdtree {

template <typename _Scalar, int _rows>
class EuclideanSpace {
public:
    typedef _Scalar Distance;
    typedef Eigen::Matrix<_Scalar, _rows, 1> State;
    static constexpr int axes = _rows;
};

template <typename _Scalar, int _rows>
class BoundedEuclideanSpace : public EuclideanSpace<_Scalar, _rows> {
    Eigen::Array<_Scalar, _rows, 2> bounds_;

public:
    BoundedEuclideanSpace(const Eigen::Array<_Scalar, _rows, 2>& bounds)
        : bounds_(bounds)
    {
    }

    BoundedEuclideanSpace(
        const Eigen::Array<_Scalar, _rows, 1>& min,
        const Eigen::Array<_Scalar, _rows, 1>& max)
    {
        bounds_.col(0) = min;
        bounds_.col(1) = max;
    }

    inline const Eigen::Array<_Scalar, _rows, 2>& bounds() const {
        return bounds_;
    }
};

template <typename _Scalar>
class SO3Space {
public:
    typedef _Scalar Distance;
    typedef Eigen::Quaternion<_Scalar> State;
    static constexpr int axes = 3;
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
class RatioWeightedSpace : public _Space {
public:
    // inherit constructor
    using _Space::_Space;

    RatioWeightedSpace(const _Space& space)
        : _Space(space)
    {
    }
};

template <typename ... _States>
class CompoundState {
    std::tuple<_States...> states_;
    
public:
    template <typename ... _Args>
    CompoundState(_Args&& ... args)
        : states_(_States(std::forward<_Args>(args))...)
    {
    }

    template <std::size_t _index>
    const typename std::tuple_element<_index, std::tuple<_States...>>::type& substate() const {
        return std::get<_index>(states_);
    }
};

namespace detail {
template <typename ... _Scalars>
struct ScalarResult {};
template <typename _Scalar>
struct ScalarResult<_Scalar> { typedef _Scalar type; };
template <typename ... _Scalars>
struct ScalarResult<long double, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<long double, double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<long double, float, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, double, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<double, float, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, long double, _Scalars...> : ScalarResult<long double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, double, _Scalars...> : ScalarResult<double, _Scalars...> {};
template <typename ... _Scalars>
struct ScalarResult<float, float, _Scalars...> : ScalarResult<float, _Scalars...> {};

template <int ... values>
struct sum { static constexpr int value = 0; };
template <int first, int ... rest>
struct sum<first, rest...> { static constexpr int value = first + sum<rest...>::value; };

}


template <typename ... _Spaces>
class CompoundSpace {
    std::tuple<_Spaces...> spaces_;

public:
    typedef CompoundState<typename _Spaces::State...> State;

    static constexpr std::size_t size = sizeof...(_Spaces);
    static constexpr int axes = detail::sum<_Spaces::axes...>::value;

    typedef typename detail::ScalarResult<typename _Spaces::Distance...>::type Distance;

    template <typename ... _Args>
    CompoundSpace(_Args&& ... args)
        : spaces_(std::forward<_Args>(args)...)
    {
    }

    template <std::size_t _index>
    const typename std::tuple_element<_index, std::tuple<_Spaces...>>::type& subspace() const {
        return std::get<_index>(spaces_);
    }
};

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using SE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<EuclideanSpace<_Scalar, 3>, _tWeight, 1>>;

template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
using BoundedSE3Space = CompoundSpace<
    RatioWeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
    RatioWeightedSpace<BoundedEuclideanSpace<_Scalar, 3>, _tWeight, 1>>;


// template <typename _Space>
// struct CompoundSpace<_Space> {
// };

// template <typename _Space, typename ... _Rest>
// struct CompoundSpace<_Space, _Rest...> {
// };


// template <typename ... _Spaces>
// struct CompoundSpace {
//     typedef typename std::tuple<typename _Spaces::State...> State;
// };

// template <typename _Scalar, std::intmax_t _qWeight = 1, std::intmax_t _tWeight = 1>
// using SE3Space = CompoundSpace<
//     WeightedSpace<SO3Space<_Scalar>, _qWeight, 1>,
//     WeightedSpace<EuclideanSpace<_Scalar, 3>, _tWeight, 1>>;

namespace detail {

// // Bounding volume for EuclideanSpace with matching template arguments
// template <typename _Scalar, int _rows>
// struct EuclideanBounds {
//     Eigen::Array<_Scalar, _rows, 2> bounds_;
//     template <typename _Derived>
//     EuclideanBounds(const Eigen::DenseBase<_Derived>& bounds)
//         : bounds_(bounds)
//     {
//     }

//     _Scalar split(unsigned* axis) const {
//         (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
//         return bounds_.row(*axis).sum() * static_cast<_Scalar>(0.5);
//     }

//     inline _Scalar& operator() (int r, int c) {
//         return bounds_(r, c);
//     }

//     template <typename _Char, typename _Traits>
//     friend std::basic_ostream<_Char,_Traits>&
//     operator << (std::basic_ostream<_Char,_Traits>& os, const EuclideanBounds& b) {
//         return os << b.bounds_;
//     }
// };

// // Bounding volume for unbounded spaces (e.g. SO(3))
// struct UnBounds {
//     template <typename _Char, typename _Traits>
//     friend std::basic_ostream<_Char, _Traits>&
//     operator << (std::basic_ostream<_Char, _Traits>& os, const UnBounds&) {
//         return os << "UnBounds";
//     }
// };

// // Helper class for printing tuple values
// template <typename _Tuple, std::size_t _N>
// struct TuplePrinter {
//     template <typename _Char, typename _Traits>
//     static void print(std::basic_ostream<_Char,_Traits>& os, const _Tuple& t) {
//         TuplePrinter<_Tuple, _N-1>::print(os, t);
//         os << ", " << std::get<_N-1>(t);
//     }
// };

// template <typename _Tuple>
// struct TuplePrinter<_Tuple, 1> {
//     template <typename _Char, typename _Traits>
//     static void print(std::basic_ostream<_Char,_Traits>& os, const _Tuple& t) {
//         os << std::get<0>(t);
//     }
// };

// // Bounding volume for a compound space
// template <typename ... _Bounds>
// struct CompoundBounds {
//     std::tuple<_Bounds...> bounds_;
    
//     template <typename ... _Args>
//     CompoundBounds(_Args&& ... args)
//         : bounds_(std::forward<_Args>(args)...)
//     {
//     }    

//     template <typename _Char, typename _Traits>
//     friend std::basic_ostream<_Char,_Traits>&
//     operator << (std::basic_ostream<_Char,_Traits>& os, const CompoundBounds& b) {
//         TuplePrinter<decltype(bounds_), sizeof...(_Bounds)>::print(os, b.bounds_);
//         return os;
//     }
// };

// // Selects the bounding volume appropriate for a given space.
// // EuclideanSpace -> EuclideanBounds
// // SO3Space -> UnBounds
// // SE3Space -> EuclideanBounds
// // CompoundSpace<...> -> UnBounds, EuclideanBounds, or CompoundBounds<...>
// template <typename _Space>
// struct KDBoundsSelector;

// template <typename _Scalar, int _rows>
// struct KDBoundsSelector<EuclideanSpace<_Scalar, _rows>> {
//     typedef EuclideanBounds<_Scalar, _rows> type;
// };

// template <typename _Scalar>
// struct KDBoundsSelector<SO3Space<_Scalar>> {
//     typedef UnBounds type;
// };

// template <typename _Space, std::intmax_t _num, std::intmax_t _den>
// struct KDBoundsSelector<WeightedSpace<_Space, _num, _den>>
//     : KDBoundsSelector<_Space>
// {
// };

// // Final step in CompoundBounds selection, only use
// // CompoundBounds<...> for CompoundSpace's that need them.
// template <typename ... _Bounds>
// struct OptionalCompoundBounds { typedef CompoundBounds<_Bounds...> type; };
// template <typename _Bounds>
// struct OptionalCompoundBounds<_Bounds> { typedef _Bounds type; };
// template <>
// struct OptionalCompoundBounds<> { typedef UnBounds type; };

// // Helper class for building the bounding type of a CompoundSpace
// //
// // The _Bounds... are non-empty bounding types from spaces already processed,
// // The contained Build<...> template recurses arguments adding to
// // CompoundBoundsBuilder<...> type arguments as it goes.
// template <typename ... _Bounds>
// struct CompoundBoundsBuilder {
//     // Base case, no more spaces to peel off into _Bounds
//     //
//     // Now we return the appropriate bounding type for the list of
//     // bounds, depending on the number of bounds in the list.
//     // 0 bound types = use UnBounds,
//     // 1 bound type = use that type
//     // 2+ = wrap into CompoundBounds
//     template <typename ... _Spaces>
//     struct Build {
//         typedef typename OptionalCompoundBounds<_Bounds...>::type type;
//     };

//     template <typename _Space0, typename ... _Spaces>
//     struct Build<_Space0, _Spaces...> {
//         typedef typename KDBoundsSelector<_Space0>::type BoundsN;
//         typedef typename std::conditional<
//             std::is_empty<BoundsN>::value,
//             Build<_Spaces...>,
//             typename CompoundBoundsBuilder<_Bounds..., BoundsN>::template Build<_Spaces...>>::type::type type;
//     };
// };

// template <typename ... _Spaces>
// struct KDBoundsSelector<CompoundSpace<_Spaces...>> {
//     typedef typename CompoundBoundsBuilder<>::template Build<_Spaces...>::type type;
// };


// template <typename _Space>
// using KDBounds = typename KDBoundsSelector<_Space>::type;

// // TODO: use these in the traversing of the trees
// template <typename _Space>
// struct KDTraversalBounds;

// template <typename _Scalar, int _rows>
// struct KDTraversalBounds<EuclideanSpace<_Scalar, _rows>> {
    
// };

// template <typename _Scalar>
// struct KDTraversalBounds<SO3Space<_Scalar>> {
// };
    

template <typename _T>
struct KDNode {
    _T value_;
    std::array<KDNode*, 2> children_;
    KDNode(const _T& v) : value_(v), children_{{nullptr, nullptr}} {}
    ~KDNode() {
        for (int i=0 ; i<2 ; ++i)
            if (children_[i])
                delete children_[i];
    }
};

template <typename _Space>
struct KDAdder;

template <typename _Space>
struct KDAdderBase {
    typedef KDAdder<_Space> Derived;
    
    template <typename _T>
    unsigned operator() (KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        int axis;
        typename _Space::Distance d = static_cast<Derived*>(this)->maxAxis(&axis);
        return static_cast<Derived*>(this)->doAdd(*this, axis, d, p, n, depth);
    }
};

template <typename _Scalar, int _rows>
struct KDAdder<BoundedEuclideanSpace<_Scalar, _rows>> : KDAdderBase<BoundedEuclideanSpace<_Scalar, _rows>> {
    typedef BoundedEuclideanSpace<_Scalar, _rows> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    const State& key_;
    Eigen::Array<_Scalar, _rows, 2> bounds_;
    
    KDAdder(const State& key, const Space& space)
        : key_(key),
          bounds_(space.bounds())
    {
    }

    Distance maxAxis(int* axis) {
        return (bounds_.col(1) - bounds_.col(0)).maxCoeff(axis);
    }

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        _Scalar split = (bounds_(axis, 0) + bounds_(axis, 1)) * static_cast<_Scalar>(0.5);
        int childNo = (split - key_[axis]) < 0;
        if (KDNode<_T> *c = p->children_[childNo]) {
            bounds_(axis, 1-childNo) = split;
            return adder(c, n, depth+1);
        } else {
            p->children_[childNo] = n;
            return depth;
        }
    }
};

template <typename _Scalar>
int volumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    int index;
    q.coeffs().array().abs().maxCoeff(&index);
    return index;
}

// Compute a midpoint between the bounds along an SO(3) quaternion
// axis.
template <typename _DerivedMin, typename _DerivedMax>
Eigen::Array<typename _DerivedMin::Scalar, 2, 1> axisMidPoint(
    const Eigen::ArrayBase<_DerivedMin>& min,
    const Eigen::ArrayBase<_DerivedMax>& max,
    unsigned depth)
{
    // TODO: the computation of s0 can be replaced with a lookup table
    // based upon depth.
    typedef typename _DerivedMin::Scalar Scalar;
    Scalar dq = std::abs(min.matrix().dot(max.matrix()));
    // Scalar theta = acos(dq);
    // Scalar d = (1 / sin(theta));
    // Scalar s0 = sin(theta / 2) * d;
    Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));

    std::cout << "depth=" << depth << ", s0=" << std::setprecision(15) << s0 << std::endl;
    return (min + max) * s0;
}

template <typename _Scalar>
struct KDAdder<SO3Space<_Scalar>> : KDAdderBase<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    int vol_;
    int depth_;
    State key_;
    
    std::array<Eigen::Array<Scalar, 2, 3>, 2> bounds_;

    KDAdder(const State& key, const Space& space)
        : vol_(volumeIndex(key)),
          depth_(1),
          key_(key.coeffs()[vol_] < 0 ? -key.coeffs() : key.coeffs())
    {
        static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
        bounds_[0] = rt;
        bounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
    }
    KDAdder(const KDAdder&) = delete;
    KDAdder(KDAdder&& other) = default;
    //     : vol_(other.vol_),
    //       depth_(other.depth_),
    //       key_(other.key_),
    //       bounds_(other.bounds_)
    // {
    // }

    Distance maxAxis(int* axis) {
        *axis = depth_ % 3;
        // TODO: we only need the std::acos here when distances need
        // to be relative to other spaces (e.g. when this is part of
        // an SE(3) KDTree).  Otherwise, we can save a costly acos by
        // removing it, and the associated cos in doAdd.
        return std::acos(std::abs(bounds_[0].col(*axis).matrix().dot(
                                      bounds_[1].col(*axis).matrix())));
    }

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        int childNo;
        switch (++depth_) {
        case 2:
            if (KDNode<_T>* c = p->children_[childNo = vol_ < 2])
                return adder(c, n, depth+1); // tail recur
            break;
        case 3:
            if (KDNode<_T>* c = p->children_[childNo = vol_ & 1])
                return adder(c, n, depth+1); // tail recur
            break;
        default:
            // See TODO in maxAxis... here is the cos(d) that we need
            // to remove if we remove the acos in maxAxis.
            Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (std::cos(d) + 1));
            Eigen::Matrix<Scalar, 2, 1> mp =
                (bounds_[0].col(axis) + bounds_[1].col(axis)) * s0;
            Scalar dot = mp[0]*key_.coeffs()[vol_] + mp[1]*key_.coeffs()[(vol_ + axis + 1) % 4];

            // std::cout << "SO3::doAdd " << vol_ << ", " << depth << ", " << depth_
            //           << ", " << d << ", " << s0 << ", " << axis << ", " << dot << std::endl;
            
            if (KDNode<_T>* c = p->children_[childNo = (dot > 0)]) {
                bounds_[1-childNo].col(axis) = mp;
                return adder(c, n, depth+1); // tail recur
            }
            break;
        }

        p->children_[childNo] = n;
        return depth;        
    }
};

template <typename _Space, std::intmax_t _num, std::intmax_t _den>
struct KDAdder<RatioWeightedSpace<_Space, _num, _den>> : KDAdder<_Space> {
    typedef typename _Space::State State;

    // inherit constructor
    using KDAdder<_Space>::KDAdder;

    typename _Space::Distance maxAxis(int *axis) {
        return KDAdder<_Space>::maxAxis(axis) * _num / _den;
    }
};

// Helper template for computing the max axis in a compound space it
// tail recurses on incremented space indexes, tracking the best
// distance and axis as it goes.  Recursion ends once all subspaces
// are checked, and it returns the best distance found.  The axis is
// computed as the number of axes before the space + the axis within
// the subspace.
template <int _index, typename ... _Spaces>
struct CompoundMaxAxis {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;
    
    static Distance maxAxis(std::tuple<KDAdder<_Spaces>...>& adders, Distance bestDist, int *bestAxis, int axesBefore) {
        typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;
        int axis;
        Distance dist = std::get<_index>(adders).maxAxis(&axis);
        if (dist > bestDist) {
            *bestAxis = axesBefore + axis;
            bestDist = dist;
        }
        return CompoundMaxAxis<_index+1, _Spaces...>::maxAxis(
            adders, bestDist, bestAxis, axesBefore + Subspace::axes);
    }
};
// Base case
template <typename ... _Spaces>
struct CompoundMaxAxis<sizeof...(_Spaces), _Spaces...> {
    typedef typename CompoundSpace<_Spaces...>::Distance Distance;

    static Distance maxAxis(std::tuple<KDAdder<_Spaces>...>& adders, Distance bestDist, int *bestAxis, int axesBefore) {
        return bestDist;
    }
};

// Helper template for adding a node in a compound space.  It tail
// recurses until it finds the axis that is in the range of the
// subspace.  This essentially performs a linear search through the
// subspaces (again).
template <int _index, typename ... _Spaces>
struct CompoundDoAdd {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple_element<_index, std::tuple<_Spaces...>>::type Subspace;
    
    template <typename _RootAdder, typename _T>
    static unsigned doAdd(std::tuple<KDAdder<_Spaces>...>& adders, _RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        if (axis < Subspace::axes)
            return std::get<_index>(adders).doAdd(adder, axis, static_cast<typename Subspace::Distance>(d), p, n, depth);
        return CompoundDoAdd<_index+1, _Spaces...>::doAdd(adders, adder, axis - Subspace::axes, d, p, n, depth);
    }
};

// Base case.  The main difference is that it does not need to test if
// the axis is in range (though it does assert it).
template <typename ... _Spaces>
struct CompoundDoAdd<sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    static constexpr int N = sizeof...(_Spaces);
    typedef typename std::tuple_element<N-1, std::tuple<_Spaces...>>::type Subspace;
    
    template <typename _RootAdder, typename _T>
    static unsigned doAdd(std::tuple<KDAdder<_Spaces>...>& adders, _RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        assert(axis < Subspace::axes);
        return std::get<N-1>(adders).doAdd(adder, axis, static_cast<typename Subspace::Distance>(d), p, n, depth);
    }
};


template <typename ... _Spaces>
struct KDAdder<CompoundSpace<_Spaces...>> : KDAdderBase<CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    std::tuple<KDAdder<_Spaces>...> adders_;
    
    typedef std::make_index_sequence<Space::size> _Indexes;
    
    template <std::size_t ... I>
    static auto make(const State& key, const Space& space, std::index_sequence<I...>) {
        return std::make_tuple(
            KDAdder<typename std::tuple_element<I, std::tuple<_Spaces...>>::type>(
                key.template substate<I>(),
                space.template subspace<I>())...);
    }
    
    inline KDAdder(const State& key, const CompoundSpace<_Spaces...>& space)
        : adders_(make(key, space, _Indexes{}))
    {
    }

    Distance maxAxis(int *axis) {
        typedef typename std::tuple_element<0, std::tuple<_Spaces...>>::type Subspace0;
        Distance dist = std::get<0>(adders_).maxAxis(axis);
        return CompoundMaxAxis<1, _Spaces...>::maxAxis(adders_, dist, axis, Subspace0::axes);
    }    

    template <typename _RootAdder, typename _T>
    unsigned doAdd(_RootAdder& adder, int axis, Distance d, KDNode<_T>* p, KDNode<_T>* n, unsigned depth) {
        return CompoundDoAdd<0, _Spaces...>::doAdd(adders_, adder, axis, d, p, n, depth);
    }
};

} // namespace detail



template <typename _T, typename _Space, typename _TtoKey>
class KDTree {
    typedef detail::KDNode<_T> Node;

    _Space space_;
    Node *root_;
    std::size_t size_;
    unsigned depth_;

    _TtoKey tToKey_;

public:
    KDTree(_TtoKey tToKey, const _Space& space = _Space())
        : space_(space),
          root_(nullptr),
          size_(0),
          depth_(0),
          tToKey_(tToKey)
    {
    }

    ~KDTree() {
        if (root_)
            delete root_;
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0;
    }

    unsigned depth() const {
        return depth_;
    }

    void add(const _T& value) {
        Node* n = new Node(value);
        Node* p = root_;
        unsigned depth = 1;
        
        if (p == nullptr) {
            root_ = n;
        } else {
            const typename _Space::State& key = tToKey_(value);
            detail::KDAdder<_Space> adder(key, space_);
            depth = adder(p, n, 2);
        }
        
        ++size_;
        if (depth_ < depth)
            depth_ = depth;
    }
};

} // namespace kdtree
} // namespace robotics
} // namespace unc

#endif // UNC_KDTREE_HPP
