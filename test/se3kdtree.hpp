#pragma once

#include "kdtree.hpp"
#include <Eigen/Dense>
#include <array>

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

template <typename _Scalar>
Eigen::Array<_Scalar, 3, 1> quaternionToVolAngles(int vol, const Eigen::Quaternion<_Scalar>& q) {
    // TODO: both std::atan2(y,x) and std::atan(y/x) work just as well
    // given he domain restrictions on the function.  The choice
    // between the two can thus be performance based.  Though there is
    // a possibility that x will be 0 or close to it when not in the
    // rotation's main volume, thus atan2 will likely be better
    // behaved.
    
    Eigen::Array<_Scalar, 3, 1> r;
    for (int i=0 ; i<3 ; ++i)
        r[i] = std::atan2(q.coeffs()[(vol + i + 1)%4],
                          q.coeffs()[vol]);
    return r;
}

template <typename _Derived>
Eigen::Quaternion<typename _Derived::Scalar> volAnglesToQuaternion(
    int vol, const Eigen::ArrayBase<_Derived>& r)
{
    // TODO: this implementation is problematic when vol is no the
    // main vol.  we may end up with angles close to a = +/- pi/2
    // which will result in tan(a) -> inf.  This will happen when
    // c[vol] == 0 as the equations try to put the 1 component in the
    // correct place.
    typedef typename _Derived::Scalar Scalar;
    
    Eigen::Matrix<Scalar, 4, 1> c;
    for (int i=0 ; i<3 ; ++i)
        c[(vol + i + 1)%4] = std::tan(r[i]);
    c[vol] = 1;
    
    return Eigen::Quaternion<Scalar>(c.normalized());
}

template <typename _Scalar>
_Scalar dist(const Eigen::Quaternion<_Scalar>& a, const Eigen::Quaternion<_Scalar>& b) {
    return std::acos(std::abs(a.coeffs().matrix().dot(b.coeffs().matrix())));
}

template <typename _Scalar>
_Scalar dist(int vol, const Eigen::Array<_Scalar, 3, 1>& a, const Eigen::Quaternion<_Scalar>& b) {
    // _Scalar expected = dist(volAnglesToQuaternion(vol, a), b);

    // s(0+1) = s0c1 + c0s1
    
    _Scalar c0 = std::cos(a[0]);
    _Scalar s0 = std::sin(a[0]);
    _Scalar c1 = std::cos(a[1]);
    _Scalar s1 = std::sin(a[1]);
    _Scalar c2 = std::cos(a[2]);
    _Scalar s2 = std::sin(a[2]);

    _Scalar a0 = s0*c1*c2;
    _Scalar a1 = s1*c2*c0;
    _Scalar a2 = s2*c0*c1;
    _Scalar a3 = c0*c1*c2;

    _Scalar den = std::sqrt(a0*a0 + a1*a1 + a2*a2 + a3*a3);

    _Scalar d = std::acos(std::abs((a0*b.coeffs()[(vol+1)%4] +
                                    a1*b.coeffs()[(vol+2)%4] +
                                    a2*b.coeffs()[(vol+3)%4] +
                                    a3*b.coeffs()[vol])/den));
    
    // assert(std::abs(expected - d) < 1e-9);
    return d;
}


}

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
        Eigen::Array<Scalar, 2, 3> soAngles_;
        Eigen::Array<Scalar, 3, 1> soKeyVolAngles_;
        Eigen::Array<Scalar, 3, 1> soKeyVolSplits_;
        Scalar rvMinDist_;
        Scalar soMinDist_;
        int soDepth_;
        int vol_;

        Scalar dist_;
        const Node* nearest_;

        Nearest(const SE3KDTree& tree, const State& key)
            : tree_(tree),
              key_(key),
              rvBounds_(tree.bounds_),
              rvMinDist_(0),
              soMinDist_(0),
              soDepth_(0),
              dist_(std::numeric_limits<Scalar>::infinity()),
              nearest_(nullptr)
        {
            rvDeltas_.setZero();
            
            static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
            soBounds_[0] = rt;
            soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
            soAngles_.colwise() = Eigen::Array<Scalar, 2, 1>(-M_PI_4, M_PI_4);
        }

        void update(const Node* n) {
            Scalar d = distance(tree_.tToKey_(n->value_), key_);
            if (d < dist_) {
                // std::cout << "d = " << d << std::endl;
                dist_ = d;
                nearest_ = n;
            }
        }

        bool inRange(int axis) const {
            assert(soAngles_(0, axis) < soAngles_(1, axis));
            return soAngles_(0, axis) <= soKeyVolAngles_[axis]
                && soAngles_(1, axis) >= soKeyVolAngles_[axis];
        }

        Scalar faceDist(int axis, int bound) const {
            Eigen::Array<Scalar, 3, 1> face = soKeyVolAngles_;
            face[axis] = soAngles_(bound, axis);
            // return detail::dist(detail::volAnglesToQuaternion(vol_, face),
            //                     key_.template substate<0>());
            return detail::dist(vol_, face, key_.template substate<0>());
        }

        Scalar faceDists(int axis) const {
            // Scalar minDist = std::numeric_limits<Scalar>::infinity();
            // Eigen::Array<Scalar, 3, 1> face = soKeyVolAngles_;
            // for (int i=0 ; i<2 ; ++i) {
            //     face[i] = soAngles_(i, axis);
            //     minDist = std::min(minDist, detail::dist(vol_, face, key_.template substate<0>()));
            // }
            return std::min(faceDist(axis, 0), faceDist(axis, 1));
            // return minDist;
        }

        Scalar edgeDists(int a0, int a1, int a2) const {
            Scalar minDist = std::numeric_limits<Scalar>::infinity();
            Eigen::Array<Scalar, 3, 1> edge;
            edge[a0] = soKeyVolAngles_[a0];
            
            for (int i=0 ; i<2 ; ++i) {
                edge[a1] = soAngles_(i, a1);
                for (int j=0 ; j<2 ; ++j) {
                    edge[a2] = soAngles_(j, a2);
                    minDist = std::min(
                        minDist,
                        detail::dist(vol_, edge, key_.template substate<0>()));
                        // detail::dist(detail::volAnglesToQuaternion(vol_, edge),
                        //              key_.template substate<0>()));     
                }
            }
            return minDist;       
        }

        Scalar cornerDists() const {
            Scalar minDist = std::numeric_limits<Scalar>::infinity();
            Eigen::Array<Scalar, 3, 1> corner;
            for (int i=0 ; i<2 ; ++i) {
                corner[0] = soAngles_(i, 0);
                for (int j=0 ; j<2 ; ++j) {
                    corner[1] = soAngles_(j, 1);
                    for (int k=0 ; k<2 ; ++k) {
                        corner[2] = soAngles_(k, 2);
                        // std::cout << "  " << i << ", " << j << ", " << k
                        //           << " | "
                        //           << corner.transpose()
                        //           << ": "
                        //           << detail::dist(vol_, corner, key_.template substate<0>()) << std::endl;
                        minDist = std::min(
                            minDist,
                            detail::dist(vol_, corner, key_.template substate<0>()));
                            // detail::dist(detail::volAnglesToQuaternion(vol_, corner),
                            //              key_.template substate<0>()));
                    }
                }
            }
            return minDist;
        }

        Scalar soMinSplitDist() const {
            int inRangeBits = 0;
            for (int i=0 ; i<3 ; ++i)
                if (inRange(i))
                    inRangeBits |= (1 << i);

            Eigen::Array<Scalar, 2, 3> boxDots;
            int inRangeBits2 = 0;
            
            for (int axis=0 ; axis<3 ; ++axis) {
                Eigen::Matrix<Scalar, 4, 1> n;
                n.setZero();
                for (int bound=0 ; bound<2 ; ++bound) {
                    Scalar a = soAngles_(bound, axis);
                    n[vol_] = (bound*2-1)*std::sin(a);
                    n[(vol_ + axis + 1) % 4] = (1-bound*2) * std::cos(a);
                    boxDots(bound, axis) = n.dot(key_.template substate<0>().coeffs().matrix());
                }

                if ((boxDots.col(axis) >= 0).all())
                    inRangeBits2 |= (1 << axis);
            }

            // std::cout << boxDots << std::endl;
            if (inRangeBits2 != inRangeBits) {
            std::cout << "Ranges: " << inRangeBits << ", " << inRangeBits2 << " | " << vol_ << " | "
                      << soKeyVolAngles_.transpose() << " | "
                      << key_.template substate<0>().coeffs().transpose()
                      << std::endl;
            }

            // assert(inRangeBits2 == inRangeBits);

            switch (inRangeBits2) {
            case 0b111: // all in range
                return 0;
            case 0b011: // x & y in range: check 2 z faces
                return faceDists(2);
            case 0b101: // x & z in range: check 2 y faces
                return faceDists(1);
            case 0b110: // y & z in range: check 2 x faces
                return faceDists(0);
            case 0b001: // x in range: check 4 yz edges
                return edgeDists(0, 1, 2);
            case 0b010: // y in range: check 4 xz edges
                return edgeDists(1, 2, 0);
            case 0b100: // z in range: check 4 xy edges
                return edgeDists(2, 0, 1);
            case 0b000: // non in range: check 8 corners
                return cornerDists();
            default:
                abort();
            }
            // detail::dist(detail::volAnglesToQuaternion(vol_, soKeyVolSplits_),
            //              key_.template substate<0>());
        }

        Scalar soFaceDist(int i, int j, int k) {
            if (!inRange(i))
                return 0;
            
            const auto &q = key_.template substate<0>();
            
            for (int bi=0 ; bi<2; ++bi) {
                Eigen::Quaternion<Scalar> n = volAnglesToQuaternion(vol_, soAngles_(bi, i))
                Scalar cosTheta = n.coeffs().matrix().dot(q.coeffs().matrix());
                Eigen::Matrix<Scalar, 4, 1> p = q.coeffs().matrix() - n.coeffs().matrix() * cosTheta;

                bool inBounds = true;
                if (p.dot(volAnglesToQuaternion(vol_, soAngles_(0, j))) < 0 ||
                    p.dot(volAnglesToQuaternion(vol_, soAngles_(1, j))) > 0)
                {
                    // use edge i, j distance
                    inBounds = false;
                }
                if (p.dot(volAnglesToQuaternion(vol_, soAngles_(0, k))) < 0 ||
                    p.dot(volAnglesToQuaternion(vol_, soAngles_(1, k))) > 0)
                {
                    // use edge i, k distance
                    inBounds = false;
                }
                if (inBounds) {
                    // use face distance
                }
            }
        }

        Scalar soMinSplitDist2() {
            soFaceDist(0, 1, 2);
            soFaceDist(1, 2, 0);
            soFaceDist(2, 0, 1);
        }

        // Scalar soMinSplitDist2() const {
        //     Eigen::Array<Scalar, 2, 3> boxDots;
        //     int inRangeBits = 0;
            
        //     for (int axis=0 ; axis<3 ; ++axis) {
        //         Eigen::Matrix<Scalar, 4, 1> n;
        //         n.setZero();
        //         for (int bound=0 ; bound<2 ; ++bound) {
        //             Scalar a = soAngles_(bounds, axis);
        //             n[vol_] = std::sin(a);
        //             n[(vol_ + axis + 1) % 4] = std::cos(a);
        //             boxDots(bounds, axis) = n.dot(key_.template substate<0>().coeffs().matrix());
        //         }

        //         if ((boxDists.col(axis) >= 0).all())
        //             inRange |= (1 << axis);
        //     }

        //     Scalar dist = std::numeric_limits<Scalar>::infinity();
            
        //     switch (inRangeBits) {
        //     case 0b111: // all in range
        //         return 0;
        //     case 0b011: // x&y in range: check against both z bounds
        //         for (int i=0 ; i<2 ; ++i)
        //             if (boxDots(i, 2) < 0)
        //                 dist = std::min(dist, -std::asin(boxDots(i, 2)));
        //         break;
        //     case 0b101: // x&z in range: check against both y bounds
        //         for (int i=0 ; i<2 ; ++i)
        //             if (boxDots(i, 1) < 0)
        //                 dist = std::min(dist, -std::asin(boxDots(i, 1)));
        //         break;
        //     case 0b110: // y&z in range: check against both z bounds
        //         for (int i=0 ; i<2 ; ++i)
        //             if (boxDots(i, 0) < 0)
        //                 dist = std::min(dist, -std::asin(boxDots(i, 0)));
        //         break;
        //     case 0b001: // x in range, check against y&z bounds

        //     case 0b010: // y in range, check against x&z bounds

        //     case 0b100: // z in range, check against x&y bounds

        //     case 0b000: // none in range
        //     }

        //     return dist;
        // }

        void traverse(const Node* n) {
            int rvAxis;
            Scalar rvDist = (rvBounds_.col(1) - rvBounds_.col(0)).maxCoeff(&rvAxis);
            int soAxis = soDepth_ % 3;
            Scalar soDist = M_PI/(2 << (soDepth_ / 3));
            Scalar soDist2 = soAngles_(1, soAxis) - soAngles_(0, soAxis);

            // std::cout << soDist << ", " << soDist2 << std::endl;
            assert(std::abs(soDist2 - soDist) < 1e-11);

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
                    Scalar oldMinDist = rvMinDist_;
                    rvDeltas_[rvAxis] = delta;
                    rvMinDist_ = std::sqrt(rvDeltas_.sum());
                    // if (std::sqrt(rvDeltas_.sum()) <= dist_) {
                    assert(std::sqrt(rvDeltas_.sum()) == rvMinDist_);
                    std::cout << soMinSplitDist() << " ? " << soMinDist_ << std::endl;
                    assert(soMinSplitDist() == soMinDist_);
                    if (rvMinDist_ + soMinDist_ <= dist_) {
                        Scalar tmp = rvBounds_(rvAxis, childNo);
                        rvBounds_(rvAxis, childNo) = split;
                        traverse(c);
                        rvBounds_(rvAxis, childNo) = tmp;                        
                    }
                    rvDeltas_[rvAxis] = oldDelta;
                    rvMinDist_ = oldMinDist;
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
                Scalar splitAngle = (soAngles_(0, soAxis) + soAngles_(1, soAxis))/2;
                Scalar dq = std::abs(soBounds_[0].col(soAxis).matrix().dot(
                                         soBounds_[1].col(soAxis).matrix()));
                Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));
                Eigen::Matrix<Scalar, 2, 1> mp =
                    (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis)) * s0;
                Scalar dot = mp[0]*key_.template substate<0>().coeffs()[vol_]
                           + mp[1]*key_.template substate<0>().coeffs()[(vol_ + soAxis + 1)%4];
                assert((soKeyVolAngles_[soAxis] > splitAngle) == (dot > 0));
                ++soDepth_;
                int childNo = (dot > 0);
                if (const Node* c = n->children_[childNo]) {
                    Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(soAxis);
                    Scalar tmpAngle = soAngles_(1-childNo, soAxis);
                    soBounds_[1-childNo].col(soAxis) = mp;
                    soAngles_(1-childNo, soAxis) = splitAngle;
                    traverse(c);
                    soAngles_(1-childNo, soAxis) = tmpAngle;
                    soBounds_[1-childNo].col(soAxis) = tmp;
                }
                update(n);
                // if (detail::volumeIndex(key_.template substate<0>()) == vol_) {
                //     std::cout << splitAngle << " - "
                //               << soKeyVolAngles_[soAxis] << ": "
                //               << mp.transpose() << " : "
                //               << std::asin(std::abs(dot)) << ", "
                //               << std::abs(splitAngle - soKeyVolAngles_[soAxis])
                //               << std::endl;
                //     assert(std::abs(splitAngle - soKeyVolAngles_[soAxis])
                //            >= std::asin(std::abs(dot)));
                // }

                if (const Node* c = n->children_[1-childNo]) {
                    Scalar tmpSo = soKeyVolSplits_[soAxis];
                    soKeyVolSplits_[soAxis] = splitAngle;
                    
                    // Scalar df =
                    //     soBounds_[childNo](0, soAxis) * key_.template substate<0>().coeffs()[vol_] +
                    //     soBounds_[childNo](1, soAxis) * key_.template substate<0>().coeffs()[(vol_ + soAxis + 1) % 4];
                    // df = std::min(std::abs(dot), std::abs(df));
                    // //if (asinXlessThanY(df, t.minDist())) {

                    // if (std::abs(dist_ - 0.0420631) <= 1e-7 &&
                    //     dist_ <= std::abs(soKeyVolAngles_[soAxis] - splitAngle) &&
                    //     std::abs(soKeyVolAngles_[soAxis] - splitAngle) <= dist_ + 0.004)
                    // {
                    //     std::cout << "so vol " << vol_ << " axis " << soAxis
                    //               << " " << soKeyVolAngles_[soAxis]
                    //               << " in " << soAngles_(0,soAxis)
                    //               << " .. " << splitAngle
                    //               << " .. " << soAngles_(1,soAxis)
                    //               << " dist " << dist_ << " vs " << std::abs(soKeyVolAngles_[soAxis] - splitAngle)
                    //               << " " << std::abs(soKeyVolAngles_[soAxis] - soAngles_(1-childNo, soAxis))
                    //               << std::endl;
                    // }
                    // if (//std::min(
                    //         std::abs(soKeyVolAngles_[soAxis] - splitAngle)
                    //         //  M_PI_2 - std::abs(soKeyVolAngles_[soAxis] - splitAngle)
                    //         // std::abs(soKeyVolAngles_[soAxis] - soAngles_(1-childNo, soAxis))
                    //          <= dist_) {
                    Scalar tmpAngle = soAngles_(childNo, soAxis);
                    soAngles_(childNo, soAxis) = splitAngle;
                    Scalar oldMinDist = soMinDist_;
                    soMinDist_ = soMinSplitDist();
                    assert(std::sqrt(rvDeltas_.sum()) == rvMinDist_);
                    assert(soMinSplitDist() == soMinDist_);
                    if (rvMinDist_ + soMinDist_ <= dist_) {
                        // if (std::asin(df) <= dist_) {
                        Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(soAxis);
                        soBounds_[childNo].col(soAxis) = mp;
                        traverse(c);
                        soBounds_[childNo].col(soAxis) = tmp;
                    }
                    soAngles_(childNo, soAxis) = tmpAngle;
                    soKeyVolSplits_[soAxis] = tmpSo;
                    soMinDist_ = oldMinDist;
                }
                --soDepth_;
            }
        }

        void traverseRoots(const std::array<Node*, 4>& roots) {
            // std::cout << key_.template substate<0>().coeffs().transpose() << std::endl;
            int mainVol = detail::volumeIndex(key_.template substate<0>());
            for (int i=0 ; i<4 ; ++i) {
                if (const Node* root = roots[vol_ = (mainVol + i)%4]) {
                    if (key_.template substate<0>().coeffs()[vol_] < 0)
                        key_.template substate<0>().coeffs() = -key_.template substate<0>().coeffs();
                    
                    soKeyVolAngles_ = detail::quaternionToVolAngles(vol_, key_.template substate<0>());
                    soKeyVolSplits_ = soKeyVolAngles_;
                    soMinDist_ = soMinSplitDist(); // TODO: this is always 0 when i == 0

                    // std::cout << "== " << i << ": " << soMinDist_ << " ==" << std::endl;
                    
                    // std::cout << soKeyVolAngles_.transpose() << std::endl;

                    // std::cout << mainVol << ": " << i << std::endl;
                    // std::cout << soAngles_ << std::endl;
                    // std::cout << soKeyVolAngles_.transpose() << std::endl;
                        
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

        Eigen::Array<_Scalar, 3, 1> soKeyVolAngles = detail::quaternionToVolAngles(vol, soKey);
        
        Eigen::Array<_Scalar, 3, 2> rvBounds(bounds_);
        std::array<Eigen::Array<_Scalar, 2, 3>, 2> soBounds;
        Eigen::Array<_Scalar, 2, 3> soAngles;
        soAngles.colwise() = Eigen::Array<_Scalar, 2, 1>(-M_PI_4, M_PI_4);
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
            Scalar soDist2 = soAngles(1, soAxis) - soAngles(0, soAxis);

            // std::cout << soDist << ", " << soDist << std::endl;
            assert((soDist - soDist2) < 1e-11);

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
                Scalar splitAngle = (soAngles(0, soAxis) + soAngles(1, soAxis))/2;
                Scalar dq = std::abs(soBounds[0].col(soAxis).matrix().dot(
                                         soBounds[1].col(soAxis).matrix()));
                Scalar s0 = std::sqrt(static_cast<Scalar>(0.5) / (dq + 1));

                Eigen::Matrix<Scalar, 2, 1> mp =
                    (soBounds[0].col(soAxis) + soBounds[1].col(soAxis)) * s0;
                Scalar dot = mp[0]*soKey.coeffs()[vol] + mp[1]*soKey.coeffs()[(vol + soAxis + 1)%4];
                assert((soKeyVolAngles[soAxis] > splitAngle) == (dot > 0));
                if ((c = p->children_[childNo = (dot > 0)]) == nullptr) {
#ifdef KD_DEBUG
                    p->space_ = 1;
                    p->axis_ = soAxis;
                    p->dist_ = soDist;
#endif
                    break;
                }

                soAngles(1-childNo, soAxis) = splitAngle;
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

