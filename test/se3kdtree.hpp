#pragma once

#include "kdtree.hpp"
#include <Eigen/Dense>
#include <array>

namespace unc {
namespace robotics {
namespace kdtree {

template <typename _T, typename _Scalar, typename _TtoKey,
          std::intmax_t _qWeight = 1,
          std::intmax_t _tWeight = 1>
class SE3KDTree {

public:
    typedef _Scalar Scalar;
    typedef BoundedSE3Space<_Scalar> Space;
    typedef typename Space::State State;

private:
    struct Node {
        _T value_;
#ifdef KD_DEBUG
        int space_;
        int axis_;
        Scalar dist_;
#endif
        std::array<Node*, 2> children_;
        Node(const _T& value) : value_(value), children_{{nullptr, nullptr}} {
#ifdef KD_DEBUG
            space_ = -1;
#endif
        }
        ~Node() {
            for (int i=0 ; i<2 ; ++i)
                if (children_[i])
                    delete children_[i];
        }
    };

    _TtoKey tToKey_;
    Eigen::Array<_Scalar, 3, 2> bounds_;

    std::size_t size_;
    unsigned depth_;
    
    std::array<Node*, 4> roots_;

    static Scalar rvDist(const Eigen::Matrix<Scalar, 3, 1>& a,
                         const Eigen::Matrix<Scalar, 3, 1>& b)
    {
        return (a - b).norm();
    }

    static Scalar so3Dist(const Eigen::Quaternion<Scalar>& a,
                          const Eigen::Quaternion<Scalar>& b)
    {
        return std::acos(std::abs(a.coeffs().matrix().dot(b.coeffs().matrix())));
    }
    
    static Scalar distance(const State& a, const State& b) {
        return so3Dist(a.template substate<0>(),
                       b.template substate<0>()) * _qWeight
            + rvDist(a.template substate<1>(),
                     b.template substate<1>()) * _tWeight;
    }

    struct Nearest {
        const SE3KDTree& tree_;
        State key_;
        Eigen::Array<Scalar, 3, 2> rvBounds_;
        Eigen::Array<Scalar, 3, 1> rvDeltas_;
        std::array<Eigen::Array<Scalar, 2, 3>, 2> soBounds_;
        int soDepth_;
        int vol_;

        Scalar dist_;
        const Node* nearest_;

        Nearest(const SE3KDTree& tree, const State& key)
            : tree_(tree),
              key_(key),
              rvBounds_(tree.bounds_),
              soDepth_(0),
              dist_(std::numeric_limits<Scalar>::infinity()),
              nearest_(nullptr)
        {
            rvDeltas_.setZero();
            
            static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
            soBounds_[0] = rt;
            soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
        }

        void update(const Node* n) {
            Scalar d = distance(tree_.tToKey_(n->value_), key_);
            if (d < dist_) {
                // std::cout << "d = " << d << std::endl;
                dist_ = d;
                nearest_ = n;
            }
        }

        void traverse(const Node* n) {
            int rvAxis;
            Scalar rvDist = (rvBounds_.col(1) - rvBounds_.col(0)).maxCoeff(&rvAxis);
            int soAxis = soDepth_ % 3;
            Scalar soDist = M_PI/(2 << (soDepth_ / 3));

            if (rvDist > soDist) {
                // rv split
#ifdef KD_DEBUG
                if (n->space_ == -1) {
                    assert(n->children_[0] == nullptr);
                    assert(n->children_[1] == nullptr);
                } else {
                    assert(n->space_ == 0);
                    assert(n->axis_ == rvAxis);
                    assert(n->dist_ == rvDist);
                }
#endif
                Scalar split = (rvBounds_(rvAxis, 0) + rvBounds_(rvAxis, 1)) * static_cast<Scalar>(0.5);
                Scalar delta = (split - key_.template substate<1>()[rvAxis]); 
                int childNo = delta < 0;
                if (const Node* c = n->children_[childNo]) {
                    Scalar tmp = rvBounds_(rvAxis, 1-childNo);
                    rvBounds_(rvAxis, 1-childNo) = split;
                    traverse(c);
                    rvBounds_(rvAxis, 1-childNo) = tmp;
                }
                update(n);
                if (const Node* c = n->children_[1-childNo]) {
                    delta *= delta;
                    Scalar oldDelta = rvDeltas_[rvAxis];
                    rvDeltas_[rvAxis] = delta;
                    if (rvDeltas_.sum() <= dist_*dist_) {
                        Scalar tmp = rvBounds_(rvAxis, childNo);
                        rvBounds_(rvAxis, childNo) = split;
                        traverse(c);
                        rvBounds_(rvAxis, childNo) = tmp;                        
                    }
                    rvDeltas_[rvAxis] = oldDelta;
                }
            } else {
                // so split
#ifdef KD_DEBUG
                if (n->space_ == -1) {
                    assert(n->children_[0] == nullptr);
                    assert(n->children_[1] == nullptr);
                } else {
                    assert(n->space_ == 1);
                    assert(n->axis_ == soAxis);
                    assert(n->dist_ == soDist);
                }
#endif
                Scalar dq = std::abs(soBounds_[0].col(soAxis).matrix().dot(
                                         soBounds_[1].col(soAxis).matrix()));
                Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));
                Eigen::Matrix<Scalar, 2, 1> mp =
                    (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis)) * s0;
                Scalar dot = mp[0]*key_.template substate<0>().coeffs()[vol_]
                    + mp[1]*key_.template substate<0>().coeffs()[(vol_ + soAxis + 1)%4];
                ++soDepth_;
                int childNo = (dot > 0);
                if (const Node* c = n->children_[childNo]) {
                    Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(soAxis);
                    soBounds_[1-childNo].col(soAxis) = mp;
                    traverse(c);
                    soBounds_[1-childNo].col(soAxis) = tmp;
                }
                update(n);
                if (const Node* c = n->children_[1-childNo]) {
                    Scalar df =
                        soBounds_[childNo](0, soAxis) * key_.template substate<0>().coeffs()[vol_] +
                        soBounds_[childNo](1, soAxis) * key_.template substate<0>().coeffs()[(vol_ + soAxis + 1) % 4];
                    df = std::min(std::abs(dot), std::abs(df));
                    //if (asinXlessThanY(df, t.minDist())) {
                    if (std::asin(df) <= dist_) {
                        Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(soAxis);
                        soBounds_[childNo].col(soAxis) = mp;
                        traverse(c);
                        soBounds_[childNo].col(soAxis) = tmp;
                    }
                }
                --soDepth_;
            }
        }

        void traverseRoots(const std::array<Node*, 4>& roots) {
            int mainVol = detail::volumeIndex(key_.template substate<0>());
            for (int i=0 ; i<4 ; ++i) {
                if (const Node* root = roots[vol_ = (mainVol + i)%4]) {
                    if (key_.template substate<0>().coeffs()[vol_] < 0)
                        key_.template substate<0>().coeffs() = -key_.template substate<0>().coeffs();
                    traverse(root);
                }
            }
        }
    };

public:
    SE3KDTree(_TtoKey tToKey, const Eigen::Array<_Scalar, 3, 2>& bounds)
        : tToKey_(tToKey),
          bounds_(bounds),
          size_(0),
          depth_(0),
          roots_{{nullptr, nullptr, nullptr, nullptr}}
    {
    }

    ~SE3KDTree() {
        for (int i=0 ; i<4 ; ++i)
            if (roots_[i])
                delete roots_[i];
    }

    std::size_t size() const {
        return size_;
    }

    unsigned depth() const {
        return depth_;
    }

    void add(const _T& t) {
        const State& key = tToKey_(t);

        Eigen::Quaternion<Scalar> soKey = key.template substate<0>();
        int vol = detail::volumeIndex(soKey);
        Node *n = new Node(t);
        Node *p, *c;
        
        if ((p = roots_[vol]) == nullptr) {
            roots_[vol] = n;
            size_ = 1;
            depth_ = 1;
            return;
        }

        if (soKey.coeffs()[vol] < 0)
            soKey.coeffs() = -soKey.coeffs();

        Eigen::Array<_Scalar, 3, 2> rvBounds(bounds_);
        std::array<Eigen::Array<_Scalar, 2, 3>, 2> soBounds;
        static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
        soBounds[0] = rt;
        soBounds[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
        int soDepth = 0;
        int childNo;
        unsigned depth = 2;

        // std::cout << M_PI/(2 << (soDepth / 3)) << std::endl;
        
        for ( ; ; p = c, ++depth) {
            int rvAxis;
            Scalar rvDist = (rvBounds.col(1) - rvBounds.col(0)).maxCoeff(&rvAxis);
            int soAxis = soDepth % 3;
            Scalar soDist = M_PI/(2 << (soDepth / 3));

// #ifdef KD_DEBUG
//             if (size_ == 98422) std::cout << depth << ": " << rvDist << ", " << soDist << std::endl;
// #endif
            if (rvDist > soDist) {
                // rv split
                Scalar split = (rvBounds(rvAxis, 0) + rvBounds(rvAxis, 1)) * static_cast<Scalar>(0.5);
                childNo = (split - key.template substate<1>()[rvAxis]) < 0;
                if ((c = p->children_[childNo]) == nullptr) {
#ifdef KD_DEBUG
                    p->space_ = 0;
                    p->axis_ = rvAxis;
                    p->dist_ = rvDist;
#endif
                    break;
                }

                rvBounds(rvAxis, 1-childNo) = split;
            } else {
                // so split
                Scalar dq = std::abs(soBounds[0].col(soAxis).matrix().dot(
                                         soBounds[1].col(soAxis).matrix()));
                Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));

                Eigen::Matrix<Scalar, 2, 1> mp =
                    (soBounds[0].col(soAxis) + soBounds[1].col(soAxis)) * s0;
                Scalar dot = mp[0]*soKey.coeffs()[vol] + mp[1]*soKey.coeffs()[(vol + soAxis + 1)%4];
                if ((c = p->children_[childNo = (dot > 0)]) == nullptr) {
#ifdef KD_DEBUG
                    p->space_ = 1;
                    p->axis_ = soAxis;
                    p->dist_ = soDist;
#endif
                    break;
                }
                
                soBounds[1-childNo].col(soAxis) = mp;
                ++soDepth;
            }
        }

        p->children_[childNo] = n;
        depth_ = std::max(depth_, depth);
        ++size_;
    }

    const _T* nearest(const State& state, Scalar* dist = 0) const {
        Nearest nearest(*this, state);
        nearest.traverseRoots(roots_);
        if (dist) *dist = nearest.dist_;
        return &nearest.nearest_->value_;
    }
    
}; // class unc::robotics::kdtree::SE3KDTree

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

