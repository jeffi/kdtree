#pragma once

#include <Eigen/Dense>
#include <array>

namespace unc {
namespace robotics {
namespace kdtree {

template <typename _Scalar>
int volumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    int m;
    q.coeffs().array().abs().maxCoeff(&m);
    return m;
}

template <typename _T, typename _Scalar, typename _TtoKey,
          std::intmax_t _qWeight = 1,
          std::intmax_t _tWeight = 1>
class SE3KDTree {
public:
    typedef _Scalar Scalar;
    typedef Eigen::Quaternion<Scalar> SOState;
    typedef Eigen::Matrix<Scalar, 3, 1> RVState;
    typedef std::tuple<SOState, RVState> State;

private:
    static Scalar dot(const SOState& a, const SOState& b) {
        return a.coeffs().matrix().dot(b.coeffs().matrix());
    }
    
    static Scalar distance(const SOState& a, const SOState& b) {
        return std::acos(std::abs(dot(a, b)));
    }

    static Scalar distance(const RVState& a, const RVState& b) {
        return (a - b).norm();
    }
    
    static Scalar distance(const State& a, const State& b) {
        return distance(std::get<0>(a), std::get<0>(b)) * _qWeight
            + distance(std::get<1>(a), std::get<1>(b)) * _tWeight;
    }

    static bool inSoBounds(
        int vol, int axis,
        const std::array<Eigen::Array<Scalar, 2, 3>, 2>& soBounds,
        const Eigen::Quaternion<Scalar>& q)
    {
        const auto& c = q.coeffs();
        
        Scalar d0 = soBounds[0](0, axis)*c[vol] + soBounds[0](1, axis)*c[(vol + axis + 1)%4];
        Scalar d1 = soBounds[1](0, axis)*c[vol] + soBounds[1](1, axis)*c[(vol + axis + 1)%4];

        return d0 > 0 && d1 < 0;
    }

    
    struct Node {
        _T value_;
        std::array<Node*, 2> children_;
        Node(const _T& value) : value_(value), children_{{nullptr, nullptr}} {}
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
                dist_ = d;
                nearest_ = n;
            }
        }

        Scalar rvBoundsDist() {
            return std::sqrt(rvDeltas_.sum());
        }

        template <typename _Derived>
        Scalar dotBounds(int b, int axis, const Eigen::DenseBase<_Derived>& q) {
            assert(b == 0 || b == 1);
            assert(0 <= axis && axis < 3);
            
            return soBounds_[b](0, axis)*q[vol_]
                 + soBounds_[b](1, axis)*q[(vol_ + axis + 1)%4];
        }
        
        Scalar soBoundsDist() {
            const SOState& q = std::get<0>(key_);
            Scalar qv = q.coeffs()[vol_];
            Scalar minDist = std::numeric_limits<Scalar>::infinity();
            
            for (int axis1 = 0 ; axis1 < 3 ; ++axis1) {
                Scalar qa = q.coeffs()[(vol_ + axis1 + 1)%4];
                Scalar d0 = soBounds_[0](0, axis1)*qv + soBounds_[0](1, axis1)*qa;
                Scalar d1 = soBounds_[1](0, axis1)*qv + soBounds_[1](1, axis1)*qa;

                if (d0 >= 0 && d1 <= 0) // in bounds
                    continue;

                // if not in bounds, then only one side will be out of
                // bounds.  Were this not to be the case, then we
                // would have a quaternion whose negation would be in
                // bounds, which, by construction is not possible.
                assert((d0 < 0) ^ (d1 > 0));

                Scalar dot;
                int b1;
                if (d0 < 0) {
                    // left of min
                    dot = d0;
                    b1 = 0;
                } else {
                    // right of max
                    dot = d1;
                    b1 = 1;
                }

                // compute `pf` which is the point on the face that is
                // closest to the query point.  This point may not be
                // within the bounds of the other axes, even if q is.
                // 
                // pf = (q - n d0) / sqrt(1 - d0*d0)
                
                Eigen::Matrix<Scalar, 4, 1> pf = q.coeffs();
                pf[vol_]                -= soBounds_[b1](0, axis1)*dot;
                pf[(vol_ + axis1 + 1)%4] -= soBounds_[b1](1, axis1)*dot;
                assert(std::abs(pf.norm() - std::sqrt(1 - dot*dot)) < 1e-9);
                pf.normalize();

                if (pf[vol_] < 0)
                    pf = -pf;

                // std::cout << "vol = " << vol_ << std::endl;
                // std::cout << "q = " << q.coeffs().transpose() << std::endl;
                // std::cout << "p = " << p.transpose() << std::endl;
                // std::cout << "b = " << soBounds_[b1].col(axis1).transpose()
                //           << ", norm=" << soBounds_[b1].col(axis1).matrix().norm()
                //           << std::endl;

                // std::cout << "b" <<  b << ", " << dot << ": " << std::abs(dotBounds(b, axis1, p)) << std::endl;
                // std::cout << "dp: " << std::acos(std::abs(p.dot(q.coeffs().matrix()))) << std::endl
                //           << "db: " << std::asin(std::abs(dot)) << std::endl;

                // p should be on the boundary b (i.e. dot = 0)
                assert(std::abs(dotBounds(b1, axis1, pf)) < 1e-9);
                // distance from p to q should be distance from q to boundary
                assert(std::abs(std::acos(std::abs(pf.dot(q.coeffs().matrix()))) -
                                std::asin(std::abs(dot))) < 1e-9);
                
                Scalar edgeDist = std::numeric_limits<Scalar>::infinity();
                for (int a2 = 1 ; a2 <= 2 ; ++a2) {
                    int axis2 = (axis1 + a2) % 3;
                    int axis3 = (axis1 + 3 - a2) % 3;
                    assert(axis1 != axis2);
                    assert(axis1 != axis3);
                    assert(axis2 != axis3);
                    

                    // compute the angle of pf from the bounds of axis2
                    Scalar de0 = soBounds_[0](0, axis2)*pf[vol_]
                        +        soBounds_[0](1, axis2)*pf[(vol_ + axis2 + 1)%4];
                    Scalar de1 = soBounds_[1](0, axis2)*pf[vol_]
                        +        soBounds_[1](1, axis2)*pf[(vol_ + axis2 + 1)%4];
                    
                    // std::cout << "  a: " << de0 << ", " << de1 << std::endl;
                    
                    if (de0 >= 0 && de1 <= 0) {
                        // in bounds
                        continue;
                    }
                    assert((de0 < 0) ^ (de1 > 0));

                    Scalar dote;
                    int b2;
                    if (de0 < 0) {
                        // left of min
                        dote = de0;
                        b2 = 0;
                    } else {
                        // right of max
                        dote = de1;
                        b2 = 1;
                    }

                    // compute `pe`, the point on the edge between the
                    // two normals n0 and n1 that is closest to q.

                    Eigen::Matrix<Scalar, 4, 1> pe;
                    Scalar t1 = soBounds_[b1](0, axis1)/soBounds_[b1](1, axis1);
                    Scalar t2 = soBounds_[b2](0, axis2)/soBounds_[b2](1, axis2);
                    Scalar r = q.coeffs()[vol_]
                        - t1*q.coeffs()[(vol_ + axis1 + 1)%4]
                        - t2*q.coeffs()[(vol_ + axis2 + 1)%4];
                    Scalar s = t1*t1 + t2*t2 + 1;
                    pe[vol_] = r;
                    pe[(vol_ + axis1 + 1)%4] = -t1*r;
                    pe[(vol_ + axis2 + 1)%4] = -t2*r;
                    pe[(vol_ + axis3 + 1)%4] = q.coeffs()[(vol_ + axis3 + 1)%4] * s;
                    pe.normalize();

                    if (pe[vol_] < 0)
                        pe = pe;
                    
                    // assert(pe[vol_] >= 0);

                    // pe should be on the boundaries
                    assert(std::abs(dotBounds(b1, axis1, pe)) < 1e-9);
                    assert(std::abs(dotBounds(b2, axis2, pe)) < 1e-9);
                    // the distance from q to pe should be greater than q to pf
                    assert(std::abs(q.coeffs().matrix().dot(pe)) <
                           std::abs(q.coeffs().matrix().dot(pf)));

                    // check that pe is in bounds
                    Scalar dc0 = soBounds_[0](0, axis3) * pe[vol_]
                        +        soBounds_[0](1, axis3) * pe[(vol_ + axis3 + 1)%4];
                    Scalar dc1 = soBounds_[1](0, axis3) * pe[vol_]
                        +        soBounds_[1](1, axis3) * pe[(vol_ + axis3 + 1)%4];
                    
                    if (dc0 >= 0 && dc1 <= 0) {
                        // in bounds
                        edgeDist = std::min(edgeDist, std::acos(std::abs(q.coeffs().matrix().dot(pe))));
                        continue;
                    }

                    assert((dc0 < 0) ^ (dc1 > 0));

                    int b3;
                    if (dc0 < 0) {
                        // left of min
                        b3 = 0;
                    } else {
                        // right of max
                        b3 = 1;
                    }

                    // use the distance to the corner
                    Eigen::Matrix<Scalar, 4, 1> pc;
                    Scalar ax = soBounds_[b1](1, axis1);
                    Scalar aw = soBounds_[b1](0, axis1);
                    Scalar by = soBounds_[b2](1, axis2);
                    Scalar bw = soBounds_[b2](0, axis2);
                    Scalar cz = soBounds_[b3](1, axis3);
                    Scalar cw = soBounds_[b3](0, axis3);

                    pc[(vol_ + axis1 + 1)%4] = aw*by*cz;
                    pc[(vol_ + axis2 + 1)%4] = ax*bw*cz;
                    pc[(vol_ + axis3 + 1)%4] = ax*by*cw;
                    pc[vol_] = -ax*by*cz;
                    pc.normalize();

                    // the corner should be on all 3 bounding faces
                    assert(std::abs(dotBounds(b1, axis1, pc)) < 1e-0);
                    assert(std::abs(dotBounds(b2, axis2, pc)) < 1e-0);
                    assert(std::abs(dotBounds(b3, axis3, pc)) < 1e-0);
                    
                    Scalar cornerDist = std::acos(std::abs(q.coeffs().matrix().dot(pc)));
                    edgeDist = std::min(edgeDist, cornerDist);
                }

                Scalar faceDist;
                if (std::numeric_limits<Scalar>::infinity() == edgeDist) {
                    faceDist = std::asin(std::abs(dot));
                } else {
                    faceDist = edgeDist;
                    // std::cout << "face = " << std::asin(std::abs(dot)) << std::endl
                    //           << "edge = " << edgeDist << std::endl;
                    assert(std::asin(std::abs(dot)) <= edgeDist);
                }
                minDist = std::min(minDist, faceDist);
            }
            return minDist == std::numeric_limits<Scalar>::infinity() ? 0 : minDist;
        }

        void traverse(const Node* n, int depth) {
            int rvAxis;
            Scalar rvDist = (rvBounds_.col(1) - rvBounds_.col(0)).maxCoeff(&rvAxis);
            int soAxis = soDepth_ % 3;
            Scalar soDist = M_PI/(2 << (soDepth_ / 3));
                
            if (rvDist > soDist) {
                Scalar split = (rvBounds_(rvAxis, 0) + rvBounds_(rvAxis, 1)) * static_cast<Scalar>(0.5);
                Scalar delta = (split - std::get<1>(key_)[rvAxis]);
                int childNo = delta < 0;
                if (const Node* c = n->children_[childNo]) {
                    Scalar tmp = rvBounds_(rvAxis, 1-childNo);
                    rvBounds_(rvAxis, 1-childNo) = split;
                    traverse(c, depth+1);
                    rvBounds_(rvAxis, 1-childNo) = tmp;
                }
                update(n);
                if (const Node* c = n->children_[1-childNo]) {
                    delta *= delta;
                    Scalar oldDelta = rvDeltas_[rvAxis];
                    rvDeltas_[rvAxis] = delta;
                    if (soBoundsDist() + rvBoundsDist() <= dist_) {
                        Scalar tmp = rvBounds_(rvAxis, childNo);
                        rvBounds_(rvAxis, childNo) = split;
                        traverse(c, depth+1);
                        rvBounds_(rvAxis, childNo) = tmp;
                    }
                    rvDeltas_[rvAxis] = oldDelta;
                }
            } else {
                Eigen::Matrix<Scalar, 2, 1> mp = (soBounds_[0].col(soAxis) + soBounds_[1].col(soAxis))
                    .matrix().normalized();
                Scalar dot = mp[0]*std::get<0>(key_).coeffs()[vol_]
                           + mp[1]*std::get<0>(key_).coeffs()[(vol_ + soAxis + 1)%4];
                ++soDepth_;
                int childNo = (dot > 0);
                if (const Node *c = n->children_[childNo]) {
                    Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[1-childNo].col(soAxis);
                    soBounds_[1-childNo].col(soAxis) = mp;
                    traverse(c, depth+1);
                    soBounds_[1-childNo].col(soAxis) = tmp;
                }
                update(n);
                if (const Node *c = n->children_[1-childNo]) {
                    Eigen::Matrix<Scalar, 2, 1> tmp = soBounds_[childNo].col(soAxis);
                    soBounds_[childNo].col(soAxis) = mp;
                    if (soBoundsDist() + rvBoundsDist() <= dist_)
                        traverse(c, depth+1);
                    soBounds_[childNo].col(soAxis) = tmp;
                }
                --soDepth_;
            }            
        }

        void traverseRoots(const std::array<Node*, 4>& roots) {
            int mainVol = detail::volumeIndex(std::get<0>(key_));
            for (int i=0 ; i<4 ; ++i) {
                if (const Node* root = roots[vol_ = (mainVol + i)%4]) {
                    if (std::get<0>(key_).coeffs()[vol_] < 0)
                        std::get<0>(key_).coeffs() = -std::get<0>(key_).coeffs();

                    traverse(root, 1);
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

    std::size_t size() const { return size_; }
    unsigned depth() const { return depth_; }

    void add(const _T& t) {
        const State& key = tToKey_(t);
        SOState soKey = std::get<0>(key);
        int vol = volumeIndex(soKey);
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

        std::array<Eigen::Array<Scalar, 2, 3>, 2> soBounds;
        Eigen::Array<Scalar, 3, 2> rvBounds = bounds_;
        static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));
        soBounds[0] = rt;
        soBounds[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
        int soDepth = 0;
        int childNo;
        unsigned depth = 2;

        for ( ;; p = c, ++depth) {
            int rvAxis;
            Scalar rvDist = (rvBounds.col(1) - rvBounds.col(0)).maxCoeff(&rvAxis);
            int soAxis = soDepth % 3;
            Scalar soDist = M_PI/(2 << (soDepth / 3));

            if (rvDist > soDist) {
                Scalar split = (rvBounds(rvAxis, 0) + rvBounds(rvAxis, 1)) * static_cast<Scalar>(0.5);
                childNo = (split - std::get<1>(key)[rvAxis]) < 0;
                if ((c = p->children_[childNo]) == nullptr)
                    break;
                rvBounds(rvAxis, 1-childNo) = split;
            } else {
                Eigen::Matrix<Scalar, 2, 1> mp = (soBounds[0].col(soAxis) + soBounds[1].col(soAxis))
                    .matrix().normalized();
                
                assert(inSoBounds(vol, 0, soBounds, soKey));
                assert(inSoBounds(vol, 1, soBounds, soKey));
                assert(inSoBounds(vol, 2, soBounds, soKey));
                
                Scalar dot = mp[0]*soKey.coeffs()[vol] + mp[1]*soKey.coeffs()[(vol + soAxis + 1)%4];
                if ((c = p->children_[childNo = (dot > 0)]) == nullptr)
                    break;
                soBounds[1-childNo].col(soAxis) = mp;
                ++soDepth;
            }
        }

        p->children_[childNo] = n;
        depth_ = std::max(depth_, depth);
        ++size_;
    }

    const _T* nearest(const State& state, Scalar* dist = 0) const {
        if (size_ == 0) return nullptr;
        Nearest nearest(*this, state);
        nearest.traverseRoots(roots_);
        if (dist) *dist = nearest.dist_;
        return &nearest.nearest_->value_;
    }
    
}; // class SE3KDTree

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc
