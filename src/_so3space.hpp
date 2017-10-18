#pragma once
#ifndef UNC_ROBOTICS_KDTREE_SO3SPACE_HPP
#define UNC_ROBOTICS_KDTREE_SO3SPACE_HPP

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 2, 1> projectToAxis(
    const Eigen::QuaternionBase<Derived>& q, int vol, int axis)
{
    typedef typename Derived::Scalar Scalar;
    
    Eigen::Matrix<Scalar, 2, 1> vec(-q.coeffs()[(vol + 1 + axis)%4], q.coeffs()[vol]);
    Scalar norm = 1 / vec.norm();
    if (vec[1] < 0) norm = -norm;
    return vec*norm;
}

template <typename _Scalar>
struct KDStaticAccum<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<_Scalar> Space;

    Eigen::Array<_Scalar, 2, 3> min_;
    Eigen::Array<_Scalar, 2, 3> max_;
    
    int vol_ = -1;
    
    inline KDStaticAccum(const Space& space) {
    }

    inline unsigned dimensions() {
        return 3;
    }

    template <typename _Derived>
    void init(const Eigen::QuaternionBase<_Derived>& q) {
        if (vol_ < 0) return;
        for (unsigned axis = 0 ; axis<3 ; ++axis)
            min_.col(axis) = max_.col(axis) = projectToAxis(q, vol_, axis);
    }

    template <typename _Derived>
    void accum(const Eigen::QuaternionBase<_Derived>& q) {
        if (vol_ < 0) return;
        for (unsigned axis = 0 ; axis<3 ; ++axis) {
            Eigen::Matrix<_Scalar, 2, 1> split = projectToAxis(q, vol_, axis);
            if (split[0] < min_(0, axis))
                min_.col(axis) = split;
            if (split[0] > max_(0, axis))
                max_.col(axis) = split;
        }
    }

    _Scalar maxAxis(unsigned *axis) {
        if (vol_ < 0) {
            *axis = 0;
            return M_PI;
        } else {
            // Compute:
            //   (x_min * x_max) + (w_min * w_max) for wach axis
            //
            // This is the dot product between the min and max
            // boundaries.  By finding the minimum we find the maximum
            // acos distance.

            return (min_ * max_).colwise().sum().minCoeff(axis);
        }
    }
    
    template <typename _Builder, typename _Iter, typename _ToKey>
    void partition(_Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        if (vol_ < 0) {
            if (std::distance(begin, end) < 4) {
                for (_Iter it = begin ; it != end ; ++it)
                    it->offset_ = 0;
                return;
            }
            
            // TODO: sort 4 partitions
            Eigen::Array<std::size_t, 4, 1> counts;
            counts.setZero();

            for (_Iter it = begin ; it != end ; ++it)
                counts[so3VolumeIndex(toKey(*it))]++;
            
            std::array<_Iter, 4> its;
            std::array<_Iter, 3> stops;
            its[0] = begin;
            for (int i=0 ; i<3 ; ++i)
                its[i+1] = stops[i] = its[i] + counts[i];
            assert(its[3]+counts[3] == end);
            for (int i=0 ; i<3 ; ++i)
                for (int v ; its[i] != stops[i] ; ++(its[v]))
                    if ((v = so3VolumeIndex(toKey(*its[i]))) != i)
                        std::iter_swap(its[i], its[v]);

            // [begin q0                                             end)
            // begin [q0 ..                q2) [q2 ..                end)
            // begin  q0 (q0 .. q1) [q1 .. q2)  q2 (q2 .. q3) [q3 .. end)
            
            // select the volume with the most elements to be the root
            // this will help balance the subtrees out.
            
            for (int i = 0, v ; i<3 ; ++i) {
                counts.maxCoeff(&v);
                // swap s.t. partitioning is maintained, but
                // element from largest partition is moved to
                // (begin+i)
                //          i 0 1 2
                // before: [0011223333]
                //             0 1 2
                // after:  [3001122333]
                for (int j=0 ; j<v ; ++j)
                    std::iter_swap(begin+i, stops[j]++);
                
                counts[v]--;
            }


            begin[0].offset_ = stops[0] - begin;
            begin[1].offset_ = stops[1] - begin;
            begin[2].offset_ = stops[2] - begin;

            // for (_Iter it = begin+3 ; it != stops[0] ; ++it)
            //     assert(so3VolumeIndex(toKey(*it)) == 0);
            // for (_Iter it = stops[0] ; it != stops[1] ; ++it)
            //     assert(so3VolumeIndex(toKey(*it)) == 1);
            // for (_Iter it = stops[1] ; it != stops[2] ; ++it)
            //     assert(so3VolumeIndex(toKey(*it)) == 2);
            // for (_Iter it = stops[2] ; it != end ; ++it)
            //     assert(so3VolumeIndex(toKey(*it)) == 3);
            
            vol_ = 0;
            builder(begin+3, stops[0]);
            vol_ = 1;
            builder(stops[0], stops[1]);
            vol_ = 2;
            builder(stops[1], stops[2]);
            vol_ = 3;
            builder(stops[2], end);
            vol_ = -1;
        } else {
            // std::cout << std::distance(begin, end) << std::endl;
            // for (_Iter it = begin ; it != end ; ++it) {
            //     auto& q = toKey(*it);
            //     assert(so3VolumeIndex(q) == vol_);
            //     for (int axis = 0 ; axis < 3 ; ++axis) {
            //         assert(dotBounds(0, axis, q.coeffs()) > 0);
            //         // std::cout << dotBounds(1, axis, q.coeffs()) << std::endl;
            //         assert(dotBounds(1, axis, q.coeffs()) < 0);
            //     }
            // }
            
            _Iter mid = begin + (std::distance(begin, end)-1)/2;
            std::nth_element(begin, mid, end, [&] (auto& a, auto& b) {
                Eigen::Matrix<Scalar, 2, 1> aProj = projectToAxis(toKey(a), vol_, axis);
                Eigen::Matrix<Scalar, 2, 1> bProj = projectToAxis(toKey(b), vol_, axis);
                return aProj[0] < bProj[0];
            });
            std::iter_swap(begin, mid);
            Eigen::Matrix<Scalar, 2, 1> split = projectToAxis(toKey(*begin), vol_, axis);
            begin->split_ = split[0];
            // assert(std::abs(std::sqrt(1 - begin->split_ * begin->split_) - split[1]) < 1e-6);

            ++mid;
            
            // for (_Iter it = begin+1 ; it != mid ; ++it) {
            //     auto& q = toKey(*it);
            //     _Scalar qv = q.coeffs()[vol_];
            //     _Scalar qa = q.coeffs()[(vol_ + axis + 1)%4];
            //     _Scalar dot = split[0] * qv + split[1] * qa;
            //     if (qv < 0) dot = -dot;
                
            //     assert(dot >= 0);
            // }

            // for (_Iter it = mid ; it != end ; ++it) {
            //     auto& q = toKey(*it);
            //     _Scalar qv = q.coeffs()[vol_];
            //     _Scalar qa = q.coeffs()[(vol_ + axis + 1)%4];
            //     _Scalar dot = split[0] * qv + split[1] * qa;
            //     if (qv < 0) dot = -dot;
                
            //     assert(dot < 0);
            // }


            // assert(split[1] >= M_SQRT1_2);
            // assert(-M_SQRT1_2 < split[0] && split[0] < M_SQRT1_2);

            //Eigen::Array<Scalar, 2, 1> tmp = soBounds_[0].col(axis);
            // soBounds_[0].col(axis) = split;
            // static int indent_;
            // ++indent_;            
            builder(begin+1, mid);
            // soBounds_[0].col(axis) = tmp;

            // if (vol_ == 0) {
            //     std::cout << axis << ": " << (begin-1)->split_ << "\t";
            //     for (int i=1 ; i<indent_ ; ++i)
            //         std::cout << "  ";                
            //     std::cout << begin->value_.name_ << std::endl;
            // }

            // tmp = soBounds_[1].col(axis);
            // soBounds_[1].col(axis) = split;
            builder(mid, end);
            // soBounds_[1].col(axis) = tmp;
            // --indent_;
        }
    }
};

template <typename _Scalar>
struct KDStaticTraversal<SO3Space<_Scalar>> {
    typedef _Scalar Scalar;
    typedef SO3Space<Scalar> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    State key_;
    int keyVol_;
    int vol_ = -1;
    Scalar distToRegion_ = 0;

    std::array<Eigen::Array<Scalar, 2, 3, Eigen::RowMajor>, 2> soBounds_;

    KDStaticTraversal(const SO3Space<_Scalar>& space, const State& key)
        : key_(key),
          keyVol_(so3VolumeIndex(key))
    {
        soBounds_[0] = M_SQRT1_2;
        soBounds_[1].colwise() = Eigen::Array<_Scalar, 2, 1>(-M_SQRT1_2, M_SQRT1_2);
    }

    unsigned dimensions() {
        return 3;
    }

    template <typename _Derived>
    Distance keyDistance(const Eigen::QuaternionBase<_Derived>& q) {
        Distance dot = std::abs(key_.coeffs().matrix().dot(q.coeffs().matrix()));
        return dot < 0 ? M_PI_2 : dot > 1 ? 0 : std::acos(dot);
    }

    template <typename _Derived>
    inline Scalar dotBounds(int b, unsigned axis, const Eigen::DenseBase<_Derived>& q) {
        // assert(b == 0 || b == 1);
        // assert(0 <= axis && axis < 3);
        assert(q[vol_] >= 0);
        return soBounds_[b](0, axis)*q[vol_]
            +  soBounds_[b](1, axis)*q[(vol_ + axis + 1)%4];
    }

    inline Scalar distToRegion() {
        return distToRegion_;
    }
    
    inline Scalar computeDistToRegion() {
        const auto& q = key_.coeffs();
        int edgesToCheck = 0;
        
        // check faces
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            Eigen::Matrix<Scalar, 2, 1> dot(dotBounds(0, a0, q), dotBounds(1, a0, q));
            int b0 = dot[0] >= 0;
            if (b0 && dot[1] <= 0)
                continue; // in bounds

            Eigen::Matrix<Scalar, 4, 1> p0 = q;
            p0[vol_]              -= soBounds_[b0](0, a0) * dot[b0];
            p0[(vol_ + a0 + 1)%4] -= soBounds_[b0](1, a0) * dot[b0];
            if (p0[vol_] < 0) p0 = -p0;
            
            int a1 = (a0+1)%3;
            if (dotBounds(1, a1, p0) > 0 || dotBounds(0, a1, p0) < 0) {
                edgesToCheck |= 1 << (a0+a1);
                continue; // not on face with this axis
            }
            int a2 = (a0+2)%3;
            if (dotBounds(1, a2, p0) > 0 || dotBounds(0, a2, p0) < 0) {
                edgesToCheck |= 1 << (a0+a2);
                continue; // not on face with this axis
            }
            // the projected point is on this face, the distance to
            // the projected point is the closest point in the bounded
            // region to the query key.  Use asin of the dot product
            // to the bounding face for the distance, instead of the
            // acos of the dot product to p, since p0 is not
            // normalized for efficiency.
            return std::asin(dot[b0]);
        }

        // if the query point is within all bounds of all 3 axes, then it is within the region.
        if (edgesToCheck == 0)
            return 0;

        // int cornerChecked = 0;
        int cornersToCheck = 0;
        Eigen::Matrix<Scalar, 2, 3> T;
        T.row(0) = soBounds_[0].row(0) / soBounds_[0].row(1);
        T.row(1) = soBounds_[1].row(0) / soBounds_[1].row(1);
        
        // check edges
        // ++, +-, --, -+ for 01, 12, 20
        Scalar dotMax = 0;
        for (int a0 = 0 ; a0 < 3 ; ++a0) {
            int a1 = (a0 + 1)%3;
            int a2 = (a0 + 2)%3;
            
            if ((edgesToCheck & (1 << (a0+a1))) == 0)
                continue;

            for (int edge = 0 ; edge < 4 ; ++edge) {
                int b0 = edge & 1;
                int b1 = edge >> 1;

                Eigen::Matrix<Scalar, 4, 1> p1;
                Scalar t0 = T(b0, a0); // soBounds_[b0](0, a0) / soBounds_[b0](1, a0);
                Scalar t1 = T(b1, a1); // soBounds_[b1](0, a1) / soBounds_[b1](1, a1);
                Scalar r = q[vol_] - t0*q[(vol_ + a0 + 1)%4] - t1*q[(vol_ + a1 + 1)%4];
                Scalar s = t0*t0 + t1*t1 + 1;

                // bounds check only requires p1[3] and p1[a2], and
                // p1[3] must be non-negative.  If in bounds, then
                // [a0] and [a1] are required to compute the distance
                // to the edge.
                p1[vol_] = r;
                // p1[a0] = -t0*r;
                // p1[a1] = -t1*r;
                p1[(vol_ + a2 + 1)%4] = q[(vol_ + a2 + 1)%4] * s;
                if (p1[vol_] < 0) p1 = -p1;
                
                int b2;
                if ((b2 = dotBounds(0, a2, p1) >= 0) && dotBounds(1, a2, p1) <= 0) {
                    // projection onto edge is in bounds of a2, this
                    // point will be closer than the corners.
                    p1[(vol_ + a0 + 1)%4] = -t0*r;
                    p1[(vol_ + a1 + 1)%4] = -t1*r;
                    dotMax = std::max(dotMax, std::abs(p1.dot(q)) / p1.norm());
                    continue;
                }
                if (r < 0) b2 = 1-b2;

                int cornerCode = 1 << ((b0 << a0) | (b1 << a1) | (b2 << a2));
                cornersToCheck |= cornerCode;
                
                // if (cornerChecked & cornerCode)
                //     continue;
                // cornerChecked |= cornerCode;
                // // edge is not in bounds, use the distance to the corner
                // Eigen::Matrix<Scalar, 4, 1> p2;
                // Scalar aw = soBounds_[b0](0, a0);
                // Scalar ax = soBounds_[b0](1, a0);
                // Scalar bw = soBounds_[b1](0, a1);
                // Scalar by = soBounds_[b1](1, a1);
                // Scalar cw = soBounds_[b2](0, a2);
                // Scalar cz = soBounds_[b2](1, a2);

                // p2[a0] =  aw*by*cz;
                // p2[a1] =  ax*bw*cz;
                // p2[a2] =  ax*by*cw;
                // p2[ 3] = -ax*by*cz;

                // // // p2 should be on both bounds
                // // assert(std::abs(dotBounds(b0, a0, p2)) < 1e-7);
                // // assert(std::abs(dotBounds(b1, a1, p2)) < 1e-7);
                // // assert(std::abs(dotBounds(b2, a2, p2)) < 1e-7);
            
                // dotMax = std::max(dotMax, std::abs(q.dot(p2)) / p2.norm());
            }
        }

        for (int i=0 ; i<8 ; ++i) {
            if ((cornersToCheck & (1 << i)) == 0)
                continue;

            int b0 = i&1;
            int b1 = (i>>1)&1;
            int b2 = i>>2;
            
            Eigen::Matrix<Scalar, 4, 1> p2;
            Scalar aw = soBounds_[b0](0, 0);
            Scalar ax = soBounds_[b0](1, 0);
            Scalar bw = soBounds_[b1](0, 1);
            Scalar by = soBounds_[b1](1, 1);
            Scalar cw = soBounds_[b2](0, 2);
            Scalar cz = soBounds_[b2](1, 2);

            p2[(vol_ + 1)%4] =  aw*by*cz;
            p2[(vol_ + 2)%4] =  ax*bw*cz;
            p2[(vol_ + 3)%4] =  ax*by*cw;
            p2[vol_] = -ax*by*cz;

            // // p2 should be on both bounds
            // assert(std::abs(dotBounds(b0, a0, p2)) < 1e-7);
            // assert(std::abs(dotBounds(b1, a1, p2)) < 1e-7);
            // assert(std::abs(dotBounds(b2, a2, p2)) < 1e-7);
            
            dotMax = std::max(dotMax, std::abs(q.dot(p2)) / p2.norm());
        }
        
        return std::acos(dotMax);
    }


    // _Iter = std::vector<KDStaticNode<_T, _D, _O>>::iterator    
    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter begin, _Iter end) {
        if (vol_ < 0) {
            // at an SO(3) quadrant root
            if (std::distance(begin, end) < 4) {
                for (_Iter it = begin ; it != end ; ++it)
                    nearest.update(*it);
                return;
            }

            std::array<_Iter, 5> iters{{
                begin + 3,
                begin + begin[0].offset_,
                begin + begin[1].offset_,
                begin + begin[2].offset_,
                end
            }};
            
            for (int v = 0 ; v < 4 ; ++v) {
                if (key_.coeffs()[vol_ = (keyVol_ + v) % 4] < 0)
                    key_.coeffs() = -key_.coeffs();
                if (v != 0)
                    distToRegion_ = computeDistToRegion();
                if (nearest.distToRegion() <= nearest.dist())
                    nearest(iters[vol_], iters[vol_+1]);
            }
            vol_ = -1;
            distToRegion_ = 0;
            
            for (int i=0 ; i<3 ; ++i)
                nearest.update(begin[i]);

        } else {
            // const KDStaticNode<...>& n = *min++;
            const auto& n = *begin++;
            _Iter mid = begin + std::distance(begin, end)/2;
            // std::array<_Iter, 3> iters{{begin, begin + std::distance(begin, end)/2, end}};
            Distance q0 = key_.coeffs()[vol_];
            Distance qa = key_.coeffs()[(vol_ + axis + 1)%4];

            Eigen::Matrix<_Scalar, 2, 1> split;
            split[0] = n.split_;
            split[1] = std::sqrt(1 - n.split_*n.split_);

            Distance dot = split[0] * q0 + split[1] * qa;
            // assert(q0 > 0);
            int childNo = (dot > 0);

            Eigen::Matrix<_Scalar, 2, 1> tmp = soBounds_[1-childNo].col(axis);
            soBounds_[1-childNo].col(axis) = split;
            // dRangeCheck = soBounds_[0].col(axis).matrix().dot(soBounds_[1].col(axis).matrix());
            // std::cout << "  " << dRangeCheck << "\t" << std::acos(dRangeCheck) * 180 / M_PI << std::endl;
            // assert(dRange <= dRangeCheck);
            Scalar prevDistToRegion = distToRegion_;
            distToRegion_ = computeDistToRegion();
            if (nearest.distToRegion() <= nearest.dist()) {
                if (childNo) nearest(begin, mid); else nearest(mid, end);
                // nearest(iters[1-childNo], iters[2-childNo]);
            }
            soBounds_[1-childNo].col(axis) = tmp;

            // Distance dRange = soBounds_[0].col(axis).matrix().dot(soBounds_[1].col(axis).matrix());
            // std::cout << dRange << "\t" << std::acos(dRange) * 180 / M_PI << std::endl;
            tmp = soBounds_[childNo].col(axis);
            soBounds_[childNo].col(axis) = split;
            // Distance dRangeCheck = soBounds_[0].col(axis).matrix().dot(soBounds_[1].col(axis).matrix());
            // std::cout << "  " << dRangeCheck << "\t" << std::acos(dRangeCheck) * 180 / M_PI << std::endl;
            // assert(dRange <= dRangeCheck);
            distToRegion_ = computeDistToRegion();
            if (nearest.distToRegion() <= nearest.dist()) {
                if (childNo) nearest(mid, end); else nearest(begin, mid);
            }
            // nearest(iters[childNo], iters[childNo+1]);
            soBounds_[childNo].col(axis) = tmp;

            distToRegion_ = prevDistToRegion;
            nearest.update(n);

        }
    }
};



} // namespace unc::robotics::kdtree::detail
} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_SO3SPACE_HPP
