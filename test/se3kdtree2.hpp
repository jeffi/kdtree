#pragma once

#include <Eigen/Dense>
#include <array>

namespace unc {
namespace robotics {
namespace kdtree {

namespace detail {
template <typename _Scalar>
int volumeIndex(const Eigen::Quaternion<_Scalar>& q) {
    int m;
    q.coeffs().array().abs().maxCoeff(&m);
    return m;
}
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
        Scalar rvBoundsDistCache_;
        Scalar soBoundsDistCache_;
        int soDepth_;
        int vol_;
        unsigned explored_;
        
        Scalar dist_;
        const Node* nearest_;
        
        Nearest(const SE3KDTree& tree, const State& key)
            : tree_(tree),
              key_(key),
              rvBounds_(tree.bounds_),
              rvBoundsDistCache_(0),
              soBoundsDistCache_(0),
              soDepth_(0),
              explored_(0),
              dist_(std::numeric_limits<Scalar>::infinity()),
              nearest_(nullptr)
        {
            rvDeltas_.setZero();

            static const Scalar rt = 1 / std::sqrt(static_cast<Scalar>(2));

            soBounds_[0] = rt;
            soBounds_[1].colwise() = Eigen::Array<Scalar, 2, 1>(-rt, rt);
        }

        void update(const Node* n) {
            ++explored_;
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

        Scalar soBoundsDistAll() {
            const auto& q = std::get<0>(key_).coeffs().matrix();
            Scalar dotMax = 0; // dist = std::numeric_limits<Scalar>::infinity();
            Scalar qv = q[vol_];
            int edgeChecked = 0;
            int cornerChecked = 0;
            for (int a0 = 0 ; a0 < 3 ; ++a0) {
                int i0 = (vol_ + a0 + 1) % 4;
                Scalar qa0 = q[i0];
                Eigen::Matrix<Scalar, 2, 1> dot0(
                    soBounds_[0](0, a0)*qv + soBounds_[0](1, a0)*qa0,
                    soBounds_[1](0, a0)*qv + soBounds_[1](1, a0)*qa0);
                int b0 = dot0[0] >= 0;
                if (b0 && dot0[1] <= 0) {
                    // face @ a0 is in bounds
                    continue;
                }
                assert(b0 ^ (dot0[1] <= 0)); // only outside of one bound

                Eigen::Matrix<Scalar, 4, 1> p0 = q;
                p0[vol_] -= soBounds_[b0](0, a0) * dot0[b0];
                p0[i0  ] -= soBounds_[b0](1, a0) * dot0[b0];
                p0.normalize();
                // if (p0[vol_] < 0) // FLIP?
                //     p0 = -p0;

                // check that the projected point is on the bound
                assert(std::abs(dotBounds(b0, a0, p0)) < 1e-9);
                // check that the distance to the projected point is
                // the same as the distance to the bound.
                assert(std::abs(std::acos(std::abs(p0.normalized().dot(q))) -
                                std::asin(std::abs(dot0[b0]))) < 1e-9);
                
                bool faceInBounds = true;
                for (int a1 = a0+1 ; (a1 = a1%3) != a0 ; ++a1) {
                    int a2 = 3 - (a0 + a1);
                    assert(a1 != a0 && a2 != a0 && a1 != a2 && a2 < 3);
                    int i1 = (vol_ + a1 + 1) % 4;
                    int i2 = (vol_ + a2 + 1) % 4;
                    Eigen::Matrix<Scalar, 2, 1> dot1(
                        soBounds_[0](0, a1)*p0[vol_] + soBounds_[0](1, a1)*p0[i1],
                        soBounds_[1](0, a1)*p0[vol_] + soBounds_[1](1, a1)*p0[i1]);
                    int b1 = dot1[0] >= 0;
                    if (b1 && dot1[1] <= 0) {
                        // p0 @ a1 is in bounds
                        continue;
                    }
                    assert(b1 ^ (dot[1] <= 0)); // only outside of one bound
                    // std::cout << "face " << a0 << " out of bounds at " << a1 << "," << b1 << std::endl;
                    faceInBounds = false;

                    int edgeCode = 1 << ((a2 << 2) | (b1 << 1) | b0);
                    if (edgeChecked & edgeCode)
                        continue;
                    edgeChecked |= edgeCode;

                    // p1 = project q onto the edge
                    Eigen::Matrix<Scalar, 4, 1> p1;
                    Scalar t0 = soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
                    Scalar t1 = soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
                    Scalar r = q[vol_] - t0*q[i0] - t1*q[i1];
                    Scalar s = t0*t0 + t1*t1 + 1;
                    p1[vol_] = r;
                    p1[i0] = -t0*r;
                    p1[i1] = -t1*r;
                    p1[i2] = q[i2] * s;
                    p1.normalize();
                    // if (p1[vol_] < 0) // FLIP?
                    //     p1 = -p1;

                    // distance to p1 should be the distance to the plane
                    assert(std::abs(
                               std::acos(std::abs(p1.dot(q))) -
                               std::asin(std::abs(dotBounds(b0, a0, q)))) < 1e-9);

                    // p1 should be on both bounds
                    assert(std::abs(dotBounds(b0, a0, p1)) < 1e-9);
                    assert(std::abs(dotBounds(b1, a1, p1)) < 1e-9);

                    // check that p1 is in bounds of remaining axis
                    Eigen::Matrix<Scalar, 2, 1> dot2(
                        soBounds_[0](0, a2)*p1[vol_] + soBounds_[0](1, a2)*p1[i2],
                        soBounds_[1](0, a2)*p1[vol_] + soBounds_[1](1, a2)*p1[i2]);
                    int b2 = dot2[0] >= 0;
                    if (b2 && dot2[1] <= 0) {
                        // edge is in bounds, use the distance to this edge
                        // Scalar edgeDist = std::acos(std::abs(p1.normalized().dot(q)));
                        // std::cout << "edge "
                        //           << a0 << "," << b0 << " - "
                        //           << a1 << "," << b1 << " = " << edgeDist << std::endl;
                        // dist = std::min(dist, edgeDist);
                        dotMax = std::max(dotMax, std::abs(p1.normalized().dot(q)));
                    } else {
                        assert(b2 ^ (dot2[1] <= 0));

                        b2 = 1-b2;
                        
                        int cornerCode = 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
                        if (cornerChecked & cornerCode)
                            continue;
                        cornerChecked |= cornerCode;
                        
                        // edge is not in bounds, use the distance to the corner
                        Eigen::Matrix<Scalar, 4, 1> cp;
                        Scalar aw = soBounds_[b0](0, a0);
                        Scalar ax = soBounds_[b0](1, a0);
                        Scalar bw = soBounds_[b1](0, a1);
                        Scalar by = soBounds_[b1](1, a1);
                        Scalar cw = soBounds_[b2](0, a2);
                        Scalar cz = soBounds_[b2](1, a2);

                        cp[i0]   = aw*by*cz;
                        cp[i1]   = ax*bw*cz;
                        cp[i2]   = ax*by*cw;
                        cp[vol_] = -ax*by*cz;
                        cp.normalize();

                        assert(std::abs(dotBounds(b0, a0, cp)) < 1e-9);
                        assert(std::abs(dotBounds(b1, a1, cp)) < 1e-9);
                        assert(std::abs(dotBounds(b2, a2, cp)) < 1e-9);
                        
                        // Scalar cornerDist = std::acos(std::abs(q.dot(cp)));
                        // int corner[3];
                        // corner[a0] = b0;
                        // corner[a1] = b1;
                        // corner[a2] = b2;
                        // std::cout << "corner "
                        //           << corner[0]
                        //           << corner[1]
                        //           << corner[2]
                        //           << " = " << cornerDist << std::endl;
                        // dist = std::min(dist, cornerDist);
                        dotMax = std::max(dotMax, std::abs(q.dot(cp)));
                    }
                }

                if (faceInBounds) {
                    Scalar faceDist = std::asin(std::abs(dot0[b0]));
                    // std::cout << "face " << a0 << " = " << faceDist << std::endl;
                    // dist = std::min(dist, faceDist);
                    return faceDist;
                }
            }

            return dotMax == 0 ? 0 : std::acos(dotMax);
        }

//         Scalar soBoundsDist2() {
//             const auto& q = std::get<0>(key_).coeffs().matrix();

//             // Edge indices:
//             // edge (0,1) 
//             // edge (0,2)  x 4 = 12 possible edges
//             // edge (1,2)
//             // 00 00
//             // 00 01
//             // 00 10
//             // 00 11
//             int edges = 0;
            
//             Scalar qv = q[vol_];
//             int oobFaces = 0;
//             for (int a0 = 0 ; a0<3 ; ++a0) {
//                 int i0 = (vol_ + a0 + 1)%4;
//                 Scalar qa0 = q[i0];
//                 Eigen::Matrix<Scalar, 2, 1> dot;
//                 for (int b=0 ; b<2 ; ++b)
//                     dot[b] = soBounds_[b](0, a0)*qv + soBounds_[b](1, a0)*qa0;

//                 int b0 = dot[0] >= 0;
                
//                 // q is within the bounds
//                 if (b0 && dot[1] <= 0)
//                     continue;

//                 assert(b0 ^ (dot[1] <= 0));

//                 ++oobFaces;
                    
//                 // orthonally project q onto the to face
//                 // p = (q - n dot) / sqrt(1 - dot^2)
//                 Eigen::Matrix<Scalar, 4, 1> p = q;
//                 p[vol_] -= soBounds_[b0](0, a0)*dot[b0];
//                 p[i0  ] -= soBounds_[b0](1, a0)*dot[b0];
//                 if (p[vol_] < 0)
//                     p = -p;

//                 // fp should be on boundary b0
//                 assert(std::abs(dotBounds(b0, a0, p.normalized())) < 1e-9);
//                 // distance from p to q should be distance from q to boundary
//                 assert(std::abs(std::acos(std::abs(p.normalized().dot(q))) -
//                                 std::asin(std::abs(dot[b0]))) < 1e-9);
                
//                 // check if fp is in bounds, if so the face
//                 // contains (and the projected point is) the
//                 // closest point to q in the volume.
//                 // otherwise it is not this face.
//                 bool oob = false;
//                 for (int a1 = a0+1 ; (a1 = a1%3) != a0 ; ++a1) {
//                     Scalar qa1 = q[(vol_ + a1 + 1)%4];
//                     Eigen::Matrix<Scalar, 2, 1> edot;
//                     for (int b=0 ; b<2 ; ++b)
//                         edot[b] = soBounds_[b](0, a1)*qv + soBounds_[b](1, a1)*qa1;
//                     int b1 = edot[0] >= 0;
//                     if (b1 && edot[1] <= 0)
//                         continue;
//                     oob = true;
//                     // record that (a0,b0) and (a1,b1) are two faces
//                     // that define an edge that needs to be checked

//                     // TODO: if the face pair was already checked, then its bit should already be set.
//                     int edgeCode = (a0 + a1 - 1) << 2;
//                     if (a0 < a1) {
//                         edgeCode |= (b1 << 1) | b0;
//                     } else {
//                         edgeCode |= (b0 << 1) | b1;
//                     }
//                     std::cout << "Edge: " << a0 << "," << b0 << " - " << a1 << "," << b1 << std::endl;
//                     edges |= (1 << edgeCode);
//                 }

//                 // face is in bounds, the closest point must be on the face.
//                 if (!oob)
//                     return std::asin(std::abs(dot[b0]));
//             }

//             if (edges == 0) {
//                 std::cout << "in bounds" << std::endl;
//                 return 0; // all in bounds
//             }
            
//             Scalar dist = std::numeric_limits<Scalar>::infinity();
            
//             // corners: a0,a1,a2: 000 ... 111
//             // 8 possible
//             int corners = 0;

//             std::cout << "checking" << std::endl;

//             // after checking faces, we have to check the edges.
//             for (int edgeCode = 0 ; edgeCode < 12 ; ++edgeCode) {
//                 if (((edges >> edgeCode) & 1) == 0)
//                     continue;

//                 //     a0 a1 a2
//                 // 0 -> 0, 1  2
//                 // 1 -> 0, 2  1
//                 // 2 -> 1, 2  0
//                 int faces = edgeCode >> 2;
//                 int a0 = faces >> 1;
//                 int a1 = (faces+3) >> 1;
//                 int a2 = 2 - faces;
                
//                 int i0 = (vol_ + a0 + 1)%4;
//                 int i1 = (vol_ + a1 + 1)%4;
//                 int i2 = (vol_ + a2 + 1)%4;

//                 int b0 = edgeCode & 1;
//                 int b1 = (edgeCode >> 1) & 1;

//                 std::cout << "Edge: " << a0 << "," << b0 << " - " << a1 << "," << b1 << std::endl;
                
//                 // check edge a0,b0, a1,b1
                
//                 // project q onto the edge
//                 Scalar t0 = soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
//                 Scalar t1 = soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
//                 Scalar r = q[vol_] - t0*q[i0] - t1*q[i1];
//                 Scalar s = t0*t0 + t1*t1 + 1;
                
//                 Eigen::Matrix<Scalar, 4, 1> p;
//                 p[vol_] = r;
//                 p[i0] = -t0*r;
//                 p[i1] = -t1*r;
//                 p[i2] = q[i2] * s;
//                 p.normalize();
                    
//                 // if (p[vol_] < 0)
//                 //     p = -p;

//                 // ep should be on both boundaries
//                 assert(std::abs(dotBounds(b0, a0, p)) < 1e-9);
//                 assert(std::abs(dotBounds(b1, a1, p)) < 1e-9);
                    
// #ifndef NDEBUG
//                 // make sure that ep minimizes the distance.  A small
//                 // change in ep[i2] will stand on the edge, but should
//                 // increase the distance to q.  (Since distance is
//                 // based upon the acos, this means the dot product
//                 // must decrease for a distance increase)
//                 for (int delta = -1 ; delta <= 1 ; delta += 2) {
//                     Eigen::Matrix<Scalar, 4, 1> pc = p;
//                     pc[i2] += delta * 1e-3;
//                     pc.normalize();
//                     assert(std::abs(dotBounds(b0, a0, pc)) < 1e-9);
//                     assert(std::abs(dotBounds(b1, a1, pc)) < 1e-9);
//                     // std::cout << q.dot(p) << "\t" << q.dot(pc) << std::endl;
//                     assert(std::abs(q.dot(p)) > std::abs(q.dot(pc)));
//                 }
// #endif
                    
//                 // check that the prjected edge point is in bounds on
//                 // the remaining axis
//                 Eigen::Matrix<Scalar, 2, 1> cdot(dotBounds(0, a2, p), dotBounds(1, a2, p));

//                 int b2 = cdot[0] >= 0;
//                 if (b2 && cdot[1] <= 0) {
//                     dist = std::min(dist, std::acos(std::abs(p.dot(q))));
//                 } else {
//                     corners |= 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
//                 }
//             }

//             for (int cornerCode = 0 ; cornerCode < 8 ; ++cornerCode) {
//                 if (((corners >> cornerCode) & 1) == 0)
//                     continue;

//                 int b0 = cornerCode & 1;
//                 int b1 = (cornerCode & 2) != 0;
//                 int b2 = (cornerCode & 4) != 0;
                
//                 Eigen::Matrix<Scalar, 4, 1> cp;
//                 Scalar aw = soBounds_[b0](0, 0);
//                 Scalar ax = soBounds_[b0](1, 0);
//                 Scalar bw = soBounds_[b1](0, 1);
//                 Scalar by = soBounds_[b1](1, 1);
//                 Scalar cw = soBounds_[b2](0, 2);
//                 Scalar cz = soBounds_[b2](1, 2);

//                 cp[(vol_ + 1) % 4] = aw*by*cz;
//                 cp[(vol_ + 2) % 4] = ax*bw*cz;
//                 cp[(vol_ + 3) % 4] = ax*by*cw;
//                 cp[vol_]          = -ax*by*cz;
//                 cp.normalize();

//                 // the corner should be on all 3 bounding faces
//                 assert(std::abs(dotBounds(b0, 0, cp)) < 1e-9);
//                 assert(std::abs(dotBounds(b1, 1, cp)) < 1e-9);
//                 assert(std::abs(dotBounds(b2, 2, cp)) < 1e-9);

//                 dist = std::min(dist, std::acos(std::abs(q.dot(cp))));
//             }

//             return dist;
//         }
        
        // Scalar soBoundsDist2() {
        //     // cases:
        //     // 0: within 3 bounds => 0
        //     // 1: within 2 of 3 bounds (out of 1 bound)
        //     //    project q to face => p
        //     //    check p is within other 2 bounds => asin(dot)
        //     //    else project q to edge(s) (up to two)
        //     // 2: within 1 of 3 bounds (out of 2 bounds)
        //     //    project q to 2 faces
        //     // 3: within 0 of 3 bounds
        //     //    project q to 3 faces
        //     //
        //     // _|______/___
        //     // \|     /
        //     //  |    /
        //     //  |\  /
        //     //  | \/
        //     //  | /\
        //     //
        //     // if a q projected onto a face is in bounds, is that sufficient?
            
        //     const auto& q = std::get<0>(key_).coeffs().matrix();
        //     Scalar dist = 0;
            
        //     for (int a0=0 ; a0<3 ; ++a0) {
        //         int i0 = (vol_ + a0 + 1)%4;                
        //         Eigen::Matrix<Scalar, 2, 1> fdot(dotBounds(0, a0, q), dotBounds(1, a0, q));

        //         // check if in bounds
        //         if (fdot[0] >= 0 && fdot[1] <=0)
        //             continue;

        //         int b0 = (fdot[1] > 0);

        //         Eigen::Matrix<Scalar, 4, 1> pf = q.coeffs();
        //         pf[vol_] -= soBounds_[b0](0, a0)*fdot[b0];
        //         pf[i0  ] -= soBounds_[b0](1, a0)*fdot[b0];
        //         pf.normalize();

        //         if (pf[vol_] < 0)
        //             pf = -pf;

        //         // pf should be on the boundary b0
        //         assert(std::abs(dotBounds(b0, a0, pf)) < 1e-9);
        //         // distance from p to q should be distance from q to boundary
        //         assert(std::abs(std::acos(std::abs(pf.dot(q))) -
        //                         std::asin(std::abs(fdot[b0]))) < 1e-9);

        //         // check if pf is in bounds on the other faces
        //         for (int a1=a0+1 ; a1<3 ; ++ai) {
        //             Eigen::Matrix<Scalar, 2, 1> edot(dotBounds(0, a1, pf), dotBounds(1, a1, pf));

        //             // check if in bounds
        //             if (edot[0] >= 0 && edot[1] <= 0)
        //                 continue;

        //             int b1 = (edot[1] > 0);

        //             int a2 = (a0 + 3 - a1)%3;
        //             int i0 = (vol_ + a1 + 1)%4;
        //             int i2 = (vol_ + a2 + 1)%4;
                    
        //             Scalar t0 = soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
        //             Scalar t1 = soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
        //             Scalar r = q[vol_] - t0*q[i0] - t1*q[i1];
        //             Scalar s = t1*t1 + t2*t2 + 1;
                    
        //             Eigen::Matrix<Scalar, 4, 1> pe;
        //             pe[vol_] = r;
        //             pe[i0] = -t0*r;
        //             pe[i1] = -t1*r;
        //             pe[i2] = q[i2] * s;
        //             pe.normalize();

        //             if (pe[vol_] < 0)
        //                 pe = -pe;

        //             Eigen::Matrix<Scalar, 2, 1> cdot(dotBounds(0, a2, pe), dotBounds(1, a2, pe));

        //             if (cdot[0] >= 0 && cdot[1] <= 0) {
        //                 // projected point on edge is in bound of
        //                 // third axis, so we can return distance to
        //                 // edge.
        //                 return std::acos(std::abs(q.dot(pe)));
        //             }

        //             int b2 = (cdot[1] > 0);
        //             int i2 = (vol_ + a3 + 1)%4;
                    
        //             Eigen::Matrix<Scalar, 4, 1> pc;
        //             Scalar ax = soBounds_[b0](1, a0);
        //             Scalar aw = soBounds_[b0](0, a0);
        //             Scalar by = soBounds_[b1](1, a1);
        //             Scalar bw = soBounds_[b1](0, a1);
        //             Scalar cz = soBounds_[b2](1, a2);
        //             Scalar cw = soBounds_[b2](0, a2);

        //             pc[i0] = aw*by*cz;
        //             pc[i1] = ax*bw*cz;
        //             pc[i2] = ax*by*cw;
        //             pc[vol_] = -ax*by*cz;
        //             pc.normalize();

        //             // return distance to corner.
        //             return std::acos(std::abs(q.dot(pc)));
        //         }

        //         return std::asin(std::abs(fdot[b0]));
        //     }
        //     return dist;
        // }
        
        // Scalar soBoundsDist2() {
        //     const auto& q = std::get<0>(key_).coeffs().matrix();

        //     //   e   c
        //     // ____|__
        //     //     |
        //     //  in | e

        //     int i = (vol_ + 1)%4;
        //     int j = (vol_ + 2)%4;
        //     int k = (vol_ + 3)%4;

        //     Eigen::Matrix<Scalar, 4, 1> p;

        //     Scalar maxDot = 0;
        //     for (int b0 = 0 ; b0<2 ; ++b0) {
        //         Scalar ax = soBounds_[b0](1, 0);
        //         Scalar aw = soBounds_[b0](0, 0);
        //         for (int b1 = 0 ; b1<2 ; ++b1) {
        //             Scalar by = soBounds_[b1](1, 1);
        //             Scalar bw = soBounds_[b1](0, 1);
        //             for (int b2 = 0 ; b2<2 ; ++b2) {
        //                 Scalar cz = soBounds_[b2](1, 2);
        //                 Scalar cw = soBounds_[b2](0, 2);
        //                 p[i] = aw*by*cz;
        //                 p[j] = ax*bw*cz;
        //                 p[k] = ax*by*cw;
        //                 p[vol_] = -ax*by*cz;
        //                 maxDot = std::max(maxDot, std::abs(q.dot(p)));
        //             }
        //         }
        //     }

        //     // edges (0 1), (0 2), (1 2)
        //     //    x  (+ +), (+ -), (- +), (- -) = 12
        //     for (int a0 = 0 ; a0<2 ; ++a0) {
        //         for (int a1 = a0+1 ; a1<3 ; ++a1) {
        //             int a2 = 3 - a0 - a1; // 3 -0-1 = 2, 3 -0-2 = 1, 3 -1-2 = 0
        //             for (int b0 = 0 ; b0<2 ; ++b0) {
        //                 Scalar ax = soBounds_[b0](1, a0);
        //                 Scalar aw = soBounds_[b0](0, a0);
        //                 for (int b1 = 0 ; b1<2 ; ++b1) {
        //                     Scalar by = soBounds_[b1](1, a1);
        //                     Scalar bw = soBounds_[b1](0, a1);
        //                     Scalar t1 = aw/ax;
        //                     Scalar t2 = bw/by;
        //                     Scalar r = q[vol_] - t1*q[(vol_ + a0 + 1)%4] - t2*q[(vol_ + a1 + 1)%4];
        //                     Scalar s = t1*t1 + t2*t2 + 1;
        //                     p[vol_] = r;
        //                     p[(vol_ + a0 + 1)%4] = -t1*r;
        //                     p[(vol_ + a1 + 1)%4] = -t2*r;
        //                     p[(vol_ + a2 + 1)%4] = q[(vol_ + a2 + 1)%4] * s;
        //                     p.normalize(); // TODO: only do when needed...
        //                     if (p[vol_] < 0)
        //                         p = -p;

        //                     Scalar d0 = soBounds_[0](0, a2) * p[vol_]
        //                         +       soBounds_[0](1, a2) * p[(vol_ + a2 + 1)%4];
        //                     Scalar d1 = soBounds_[1](0, a2) * p[vol_]
        //                         +       soBounds_[1](1, a2) * p[(vol_ + a2 + 1)%4];

        //                     if (d0 >= 0 && d1 <= 0) {
        //                         // edge is in bounds
        //                         maxDot = std::max(maxDot, std::abs(q.dot(p)));
        //                     }
        //                 }
        //             }
        //         }
        //     }
            
        //     Scalar dist = std::acos(maxDot);

        //     // faces 0, 1, 2
        //     for (int a0 = 0 ; a0<3 ; ++a0) {
        //         int a1 = (a0+1)%3;
        //         int a2 = (a0+2)%3;

        //         Scalar dot0 = soBounds_[0](0, a0)*q[vol_] + soBounds_[0](1, a0)*q[(vol_ + a0 + 1)%4];
        //         Scalar dot1 = soBounds_[1](0, a0)*q[vol_] + soBounds_[1](1, a0)*q[(vol_ + a0 + 1)%4];

        //         if (dot0 >= 0 && dot1 <= 0) // in bounds
        //             continue;

        //         // project q onto the face of the bounding volume that
        //         // it is outside of.  This is computed using an
        //         // orthonormal basis: p = (q - n dot) / sqrt(1 - dot^2)
        //         p = q;
        //         if (dot0 < 0) {
        //             p[vol_]              -= soBounds_[0](0, a0)*dot0;
        //             p[(vol_ + a0 + 1)%4] -= soBounds_[0](1, a0)*dot0;
        //         } else {
        //             assert(dot1 > 0);
        //             p[vol_]              -= soBounds_[1](0, a0)*dot1;
        //             p[(vol_ + a0 + 1)%4] -= soBounds_[1](1, a0)*dot1;
        //         }

        //         if (p[vol_] < 0)
        //             p = -p;

        //         // check if p is in bounds.
        //     }

        // }
        
        Scalar soBoundsDist1() {
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
                pf[vol_]                 -= soBounds_[b1](0, axis1)*dot;
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


                    for (int delta = -1 ; delta <= 1 ; delta += 2) {
                        Eigen::Matrix<Scalar, 4, 1> pe2 = pe;
                        pe2[(vol_ + axis3 + 1)%4] += delta * 1e-3;
                        pe2.normalize();
                        assert(std::abs(dotBounds(b1, axis1, pe2)) < 1e-9);
                        assert(std::abs(dotBounds(b2, axis2, pe2)) < 1e-9);
                        assert(std::abs(q.coeffs().matrix().dot(pe)) >
                               std::abs(q.coeffs().matrix().dot(pe2)));
                    }
                                        
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
                        // std::cout << "edge " << axis1 << "," << b1 << " - " << axis2 << "," << b2 << ": " << edgeDist << std::endl;
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
                    assert(std::abs(dotBounds(b1, axis1, pc)) < 1e-9);
                    assert(std::abs(dotBounds(b2, axis2, pc)) < 1e-9);
                    assert(std::abs(dotBounds(b3, axis3, pc)) < 1e-9);

                    // distance to the corner should be greater than the distance to the edge
                    assert(std::abs(pc.dot(q.coeffs().matrix())) <
                           std::abs(pe.dot(q.coeffs().matrix())));

                    Scalar cornerDist = std::acos(std::abs(q.coeffs().matrix().dot(pc)));
                    int corner[3];
                    corner[axis1] = b1;
                    corner[axis2] = b2;
                    corner[axis3] = b3;
                    // std::cout << "corner " << corner[0] << corner[1] << corner[2] << " = "
                    //           << cornerDist << std::endl;
                    edgeDist = std::min(edgeDist, cornerDist);
                }

                Scalar faceDist;
                if (std::numeric_limits<Scalar>::infinity() == edgeDist) {
                    faceDist = std::asin(std::abs(dot));
                    // std::cout << "face " << axis1 << ": " << faceDist << std::endl;
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

        Scalar soBoundsDist() {
            // Scalar d1 = soBoundsDist1();
            // Scalar d2 = soBoundsDistAll();

            // // std::cout << "d1 = " << d1 << ", d2 = " << d2 << std::endl;
            // assert(std::abs(d1 - d2) < 1e-9);

            // return d1;

            return soBoundsDistAll();
        }

        Scalar computeSOFaceDist(int bf, int axis1) {
            const SOState& q = std::get<0>(key_);
            Scalar qv = q.coeffs()[vol_];            
            Scalar qa = q.coeffs()[(vol_ + axis1 + 1)%4];
            Scalar d0 = soBounds_[0](0, axis1)*qv + soBounds_[0](1, axis1)*qa;
            Scalar d1 = soBounds_[1](0, axis1)*qv + soBounds_[1](1, axis1)*qa;
            
            if (d0 >= 0 && d1 <= 0) // in bounds
                return 0;

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

                // distance to the corner should be greater than the distance to the edge
                assert(std::abs(pc.dot(q.coeffs().matrix())) <
                       std::abs(pe.dot(q.coeffs().matrix())));
                           
                Scalar cornerDist = std::acos(std::abs(q.coeffs().matrix().dot(pc)));
                edgeDist = std::min(edgeDist, cornerDist);
            }

            if (std::numeric_limits<Scalar>::infinity() == edgeDist) {
                return std::asin(std::abs(dot));
            } else {
                // std::cout << "face = " << std::asin(std::abs(dot)) << std::endl
                //           << "edge = " << edgeDist << std::endl;
                assert(std::asin(std::abs(dot)) <= edgeDist);
                return edgeDist;
            }
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
                    Scalar rvBoundsDistOld = rvBoundsDistCache_;
                    rvBoundsDistCache_ = rvBoundsDist();
                    if (soBoundsDistCache_ + rvBoundsDistCache_ <= dist_) {
                        Scalar tmp = rvBounds_(rvAxis, childNo);
                        rvBounds_(rvAxis, childNo) = split;
                        traverse(c, depth+1);
                        rvBounds_(rvAxis, childNo) = tmp;
                    }
                    rvBoundsDistCache_ = rvBoundsDistOld;
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
                    //Scalar faceDist = computeSOFaceDist(childNo, soAxis);
                    Scalar soBoundsDistCacheOld = soBoundsDistCache_;
                    soBoundsDistCache_ = soBoundsDist();
                    //std::cout << faceDist << " , " << bounDist << std::endl;
                    //assert(std::abs(faceDist - bounDist) < 1e-9);
                    if (soBoundsDistCache_ + rvBoundsDistCache_ <= dist_)
                        traverse(c, depth+1);
                    soBoundsDistCache_ = soBoundsDistCacheOld;
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

                    soBoundsDistCache_ = soBoundsDist();
                    if (soBoundsDistCache_ <= dist_)
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

    const _T* nearest(const State& state, Scalar* dist = 0, unsigned *explored = 0) const {
        if (size_ == 0) return nullptr;
        Nearest nearest(*this, state);
        nearest.traverseRoots(roots_);
        if (dist) *dist = nearest.dist_;
        if (explored) *explored = nearest.explored_;
        return &nearest.nearest_->value_;
    }
    
}; // class SE3KDTree

} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc
