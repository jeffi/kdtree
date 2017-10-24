#pragma once
#ifndef UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP
#define UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP

namespace unc { namespace robotics { namespace kdtree { namespace detail {

template <typename _Node, int _index, typename ... _Spaces>
struct MidpointCompoundHelper {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;

    typedef MidpointCompoundHelper<_Node, _index+1, _Spaces...> Next;

    template <typename _Traversals>
    static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
        unsigned axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        if (d > bestDist) {
            *bestAxis = dimBefore + axis;
            bestDist = d;
        }
        return Next::maxAxis(traversals, dimBefore + std::get<_index>(traversals).dimensions(), bestDist, bestAxis);
    }

    template <typename _Traversals, typename _Adder>
    static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            std::get<_index>(traversals).addImpl(adder, axis - dimBefore, p, n);
        } else {
            Next::addImpl(traversals, adder, dimAfter, axis, p, n);
        }
    }

    template <typename _Traversals, typename _State>
    static inline Distance keyDistance(_Traversals& traversals, const _State& q, Distance sum) {
        return Next::keyDistance(traversals, q, sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q)));
    }

    template <typename _Traversals>
    static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
        return Next::distToRegion(traversals, sum + std::get<_index>(traversals).distToRegion());
    }

    template <typename _Traversals, typename _Nearest>
    static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        if (axis < dimAfter) {
            std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore);
        } else {
            Next::traverse(traversals, nearest, n, dimAfter, axis);
        }
    }
};


template <typename _Node, typename ...  _Spaces>
struct MidpointCompoundHelper<_Node, sizeof...(_Spaces)-1, _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    static constexpr int _index = sizeof...(_Spaces)-1;
    
    template <typename _Traversals>
    static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
        unsigned axis;
        typename Space::Distance d = std::get<_index>(traversals).maxAxis(&axis);
        if (d > bestDist) {
            *bestAxis = dimBefore + axis;
            bestDist = d;
        }
        return bestDist;
    }

    template <typename _Traversals, typename _Adder>
    static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
        std::get<_index>(traversals).addImpl(adder, axis - dimBefore, p, n);
    }

    template <typename _Traversals, typename _State>
    static inline Distance keyDistance(_Traversals& traversals, const _State& q, Distance sum) {
        return sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q));
    }

    template <typename _Traversals>
    static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
        return sum + std::get<_index>(traversals).distToRegion();
    }

    template <typename _Traversals, typename _Nearest>
    static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
        unsigned dimAfter = dimBefore + std::get<_index>(traversals).dimensions();
        assert(axis < dimAfter);
        std::get<_index>(traversals).traverse(nearest, n, axis - dimBefore);
    }
};


// Alternate base case for MidpointCompoundHelper, 
//
// template <typename _Node, typename ...  _Spaces>
// struct MidpointCompoundHelper<_Node, sizeof...(_Spaces), _Spaces...> {
//     typedef CompoundSpace<_Spaces...> Space;
//     typedef typename Space::State Key;
//     typedef typename Space::Distance Distance;

//     template <typename _Traversals>
//     static inline Distance maxAxis(_Traversals& traversals, unsigned dimBefore, Distance bestDist, unsigned *bestAxis) {
//         return bestDist;
//     }
    
//     template <typename _Traversals, typename _Adder>
//     static inline void addImpl(_Traversals& traversals, _Adder& adder, unsigned dimBefore, unsigned axis, _Node* p, _Node* n) {
//         assert(false); // should not happen
//     }

//     template <typename _Traversals, typename _State>
//     static inline Distance keyDistance(_Traversals& traversals, const _State& q, Distance sum) {
//         return sum;
//     }

//     template <typename _Traversals>
//     static inline Distance distToRegion(_Traversals& traversals, Distance sum) {
//         return sum;
//     }

//     template <typename _Traversals, typename _Nearest>
//     static inline void traverse(_Traversals& traversals, _Nearest& nearest, const _Node* n, unsigned dimBefore, unsigned axis) {
//         assert(false); // should not happen
//     }
// };

template <typename _Node, typename ... _Spaces>
struct MidpointAddTraversal<_Node, CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple<_Spaces...> Tuple;

    const Space& space_;
    std::tuple<MidpointAddTraversal<_Node, _Spaces>...> traversals_;

    template <std::size_t ... I>
    MidpointAddTraversal(const Space& space, const Key& key, std::index_sequence<I...>)
        : space_(space),
          traversals_(MidpointAddTraversal<_Node, typename std::tuple_element<I, Tuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }

    MidpointAddTraversal(const Space& space, const Key& key)
        : MidpointAddTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    inline Distance maxAxis(unsigned* axis) {
        Distance d = std::get<0>(traversals_).maxAxis(axis);
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::maxAxis(
            traversals_, std::get<0>(space_).dimensions(), d, axis);
    }

    template <typename _Adder>
    void addImpl(_Adder& adder, unsigned axis, _Node* p, _Node* n) {
        MidpointCompoundHelper<_Node, 0, _Spaces...>::addImpl(
            traversals_, adder, 0, axis, p, n);
    }
};

template <typename _Node, typename ... _Spaces>
struct MidpointNearestTraversal<_Node, CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State Key;
    typedef typename Space::Distance Distance;
    typedef typename std::tuple<_Spaces...> Tuple;

    const Space& space_;
    std::tuple<MidpointNearestTraversal<_Node, _Spaces>...> traversals_;

    template <std::size_t ... I>
    MidpointNearestTraversal(
        const Space& space, const Key& key, std::index_sequence<I...>)
        : space_(space),
          traversals_(MidpointNearestTraversal<_Node, typename std::tuple_element<I, Tuple>::type>(
                          std::get<I>(space), std::get<I>(key))...)
    {
    }

    MidpointNearestTraversal(const Space& space, const Key& key)
        : MidpointNearestTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    template <typename _State>
    Distance keyDistance(const _State& q) {
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::keyDistance(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    inline Distance distToRegion() {
        return MidpointCompoundHelper<_Node, 1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _Nearest>
    void traverse(_Nearest& nearest, const _Node* n, unsigned axis) {
        MidpointCompoundHelper<_Node, 0, _Spaces...>::traverse(
            traversals_, nearest, n, 0, axis);
    }
};


}}}}

#endif // UNC_ROBOTICS_KDTREE_COMPOUNDSPACE_HPP
