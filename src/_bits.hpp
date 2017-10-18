#pragma once
#ifndef UNC_ROBOTICS_KDTREE_BITS_HPP
#define UNC_ROBOTICS_KDTREE_BITS_HPP

#include <Eigen/Dense>

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

// helper to enable builtin wrapper for long types.  The problem this
// resolves is that `int` and `long` may be the same type on some
// systems and not on others.
template <typename T>
struct enable_builtin_long {
    static constexpr bool value = std::is_integral<T>::value
        && sizeof(T)==sizeof(long) && sizeof(T) != sizeof(int);
};

// helper to enable builtin wrapper for long long types.  The problem
// this resolves is that `long` and `long long` may be the same type
// on some systems and not on others.
template <typename T>
struct enable_builtin_long_long {
    static constexpr bool value = std::is_integral<T>::value
        && sizeof(T)==sizeof(long long) && sizeof(T) != sizeof(long);
};

// clz returns the number of leading 0-bits in argument, starting with
// the most significant bit.  If x is 0, the result is undefined.  The
// builtins make use of processor instructions, and are defined for
// unsigned, unsigned long, and unsigned long long types.
constexpr int clz(unsigned x) { return __builtin_clz(x); }

template <typename T>
constexpr typename std::enable_if<enable_builtin_long<T>::value, int>::type
clz(T x) { return __builtin_clzl(x); }

template <typename T>
constexpr typename std::enable_if<enable_builtin_long_long<T>::value, int>::type
clz(T x) { return __builtin_clzll(x); }

template <typename UInt>
constexpr int log2(UInt x) { return sizeof(x)*8 - 1 - clz(x); }


struct DistValuePairCompare {
    template <typename _Dist, typename _Value>
    inline bool operator() (const std::pair<_Dist, _Value>& a, const std::pair<_Dist, _Value>& b) const {
        return a.first < b.first;
    }
};

template <typename _Derived>
unsigned so3VolumeIndex(const Eigen::MatrixBase<_Derived>& q) {
    unsigned index;
    q.array().abs().maxCoeff(&index);
    return index;
}

template <typename _Scalar>
unsigned so3VolumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    return so3VolumeIndex(q.coeffs());
}

// TODO: this can be shared with KDTree
template <typename _T, typename _Space, typename _TtoKey>
struct KDStaticTreeBase {
    _Space space_;
    _TtoKey tToKey_;

    KDStaticTreeBase(const _TtoKey& tToKey, const _Space& space)
        : space_(space),
          tToKey_(tToKey)
    {
    }
};

template <typename _T, typename _Distance, typename _Offset>
struct KDStaticNode {
    _T value_;

    unsigned axis_;
    
    union {
        _Distance split_;
        _Offset offset_;
    };
    
    inline KDStaticNode(const _T& v)
        : value_(v)
#ifndef NDEBUG
        , axis_(~0)
        , split_(std::numeric_limits<_Distance>::quiet_NaN())
#endif
    {
    }
};

template <typename _Space>
struct KDStaticTraversal;

template <typename _Space>
struct KDStaticAccum;


} // namespace unc::robotics::kdtree::detail
} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_BITS_HPP
