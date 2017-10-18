#pragma once
#ifndef UNC_ROBOTICS_KDTREE_COMPOUND_SPACE_HPP
#define UNC_ROBOTICS_KDTREE_COMPOUND_SPACE_HPP

namespace unc {
namespace robotics {
namespace kdtree {
namespace detail {


template <int _index, typename ... _Spaces>
struct KDCompoundAccum {
    
    template <typename _Accums>
    inline static unsigned dimensions(_Accums& accums, unsigned sum) {
        return KDCompoundAccum<_index+1, _Spaces...>::dimensions(accums, sum + std::get<_index>(accums).dimensions());
    }
    
    template <typename _Accums, typename _State>
    inline static void init(_Accums& accums, const _State& q) {
        std::get<_index>(accums).init(std::get<_index>(q));
        KDCompoundAccum<_index+1, _Spaces...>::init(accums, q);
    }

    template <typename _Accums, typename _State>
    inline static void accum(_Accums& accums, const _State& q) {
        std::get<_index>(accums).accum(std::get<_index>(q));
        KDCompoundAccum<_index+1, _Spaces...>::accum(accums, q);
    }

    template <typename _Accums, typename _Scalar>
    inline static _Scalar maxAxis(_Accums& accums, unsigned dimBefore, _Scalar dist, unsigned *axis) {
        unsigned a;
        _Scalar d = std::get<_index>(accums).maxAxis(&a);
        if (d > dist) {
            dist = d;
            *axis = a + dimBefore;
        }
        return KDCompoundAccum<_index+1, _Spaces...>::maxAxis(
            accums, dimBefore + std::get<_index>(accums).dimensions(), dist, axis);
    }

    template <typename _Accums, typename _Builder, typename _Iter, typename _ToKey>
    static void partition(_Accums& accums, _Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        int dim = std::get<_index>(accums).dimensions();
        if (axis < dim) {
            std::get<_index>(accums).partition(builder, axis, begin, end, [&] (auto& t) { return std::get<_index>(toKey(t)); });
        } else {
            KDCompoundAccum<_index+1, _Spaces...>::partition(accums, builder, axis - dim, begin, end, toKey);
        }
    }
};

template <typename ... _Spaces>
struct KDCompoundAccum<sizeof...(_Spaces), _Spaces...> {
    template <typename _Accums>
    inline static unsigned dimensions(_Accums& accums, unsigned sum) { return sum; }

    template <typename _Accums, typename _State>
    inline static void init(_Accums& accums, const _State& q) {}

    template <typename _Accums, typename _State>
    inline static void accum(_Accums& accums, const _State& q) {}

    template <typename _Accums, typename _Scalar>
    inline static _Scalar maxAxis(_Accums& accums, unsigned dimBefore, _Scalar dist, unsigned *axis) {
        return dist;
    }

    template <typename _Accums, typename _Builder, typename _Iter, typename _ToKey>
    inline static void partition(_Accums& accums, _Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        assert(false); // should not be called.
    }
};


template <typename ... _Spaces>
struct KDStaticAccum<CompoundSpace<_Spaces...>> {
    typedef KDStaticAccum NestedAccum;
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    
    std::tuple<KDStaticAccum<_Spaces>...> accums_;
    
    template <std::size_t ...I>
    KDStaticAccum(const Space& space, std::index_sequence<I...>)
        : accums_(std::get<I>(space)...)
    {
    }
    
    KDStaticAccum(const Space& space)
        : KDStaticAccum(space, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }
    
    unsigned dimensions() {
        return KDCompoundAccum<0, _Spaces...>::dimensions(accums_, 0);
    }
    
    template <typename _State>
    void init(const _State& q) {
        KDCompoundAccum<0, _Spaces...>::init(accums_, q);
    }

    template <typename _State>
    void accum(const _State& q) {
        KDCompoundAccum<0, _Spaces...>::accum(accums_, q);
    }

    Distance maxAxis(unsigned *axis) {
        Distance d = std::get<0>(accums_).maxAxis(axis);
        return KDCompoundAccum<1, _Spaces...>::maxAxis(accums_, std::get<0>(accums_).dimensions(), d, axis);
    }

    template <typename _Builder, typename _Iter, typename _ToKey>
    void partition(_Builder& builder, int axis, _Iter begin, _Iter end, const _ToKey& toKey) {
        KDCompoundAccum<0, _Spaces...>::partition(accums_, builder, axis, begin, end, toKey);
    }
    
};


template <int _index, typename ... _Spaces>
struct CompoundStaticTraversal {
    template <typename _Traversals, typename _Nearest, typename _Iter>
    static void traverse(_Traversals& traversals, _Nearest& nearest, unsigned axis, _Iter min, _Iter max) {
        unsigned dim = std::get<_index>(traversals).dimensions();
        if (axis < dim) {
            std::get<_index>(traversals).traverse(nearest, axis, min, max);
        } else {
            CompoundStaticTraversal<_index+1, _Spaces...>::traverse(
                traversals, nearest, axis - dim, min, max);
        }
    }
};

template <typename ... _Spaces>
struct CompoundStaticTraversal<sizeof...(_Spaces)-1, _Spaces...> {
    static constexpr int _index = sizeof...(_Spaces)-1;
    
    template <typename _Traversals, typename _Nearest, typename _Iter>
    static void traverse(_Traversals& traversals, _Nearest& nearest, unsigned axis, _Iter min, _Iter max) {
        std::get<_index>(traversals).traverse(nearest, axis, min, max);
    }    
};

template <int _index, typename ... _Spaces>
struct CompoundStaticKeyDistance {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    inline static Distance accum(
        std::tuple<KDStaticTraversal<_Spaces>...>& traversals,
        const State& q,
        Distance sum)
    {
        return CompoundStaticKeyDistance<_index+1, _Spaces...>::accum(
            traversals, q, sum + std::get<_index>(traversals).keyDistance(std::get<_index>(q)));
    }
};

template <typename ... _Spaces>
struct CompoundStaticKeyDistance<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;

    inline static Distance accum(
        std::tuple<KDStaticTraversal<_Spaces>...>& traversals,
        const State& q,
        Distance sum)
    {
        return sum;
    }
};

template <int _index, typename ... _Spaces>
struct CompoundStaticDistToRegion {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    
    inline static Distance distToRegion(
        std::tuple<KDStaticTraversal<_Spaces>...>& traversals,
        Distance sum)
    {
        return CompoundStaticDistToRegion<_index+1, _Spaces...>::distToRegion(
            traversals, sum + std::get<_index>(traversals).distToRegion());
    }
};

template <typename ... _Spaces>
struct CompoundStaticDistToRegion<sizeof...(_Spaces), _Spaces...> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::Distance Distance;
    
    inline static Distance distToRegion(
        std::tuple<KDStaticTraversal<_Spaces>...>& traversals,
        Distance sum)
    {
        return sum;
    }
};


template <typename ... _Spaces>
struct KDStaticTraversal<CompoundSpace<_Spaces...>> {
    typedef CompoundSpace<_Spaces...> Space;
    typedef typename Space::State State;
    typedef typename Space::Distance Distance;
    
    typedef std::tuple<_Spaces...> SpaceTuple;
    
    std::tuple<KDStaticTraversal<_Spaces>...> traversals_;

    template <std::size_t ... I>
    KDStaticTraversal(const Space& space, const State& key, std::index_sequence<I...>)
        : traversals_(
            KDStaticTraversal<typename std::tuple_element<I, SpaceTuple>::type>(
                std::get<I>(space),
                std::get<I>(key))...)
    {
    }
    
    KDStaticTraversal(const Space& space, const State& key)
        : KDStaticTraversal(space, key, std::make_index_sequence<sizeof...(_Spaces)>{})
    {
    }

    Distance distToRegion() {
        return CompoundStaticDistToRegion<1, _Spaces...>::distToRegion(
            traversals_, std::get<0>(traversals_).distToRegion());
    }

    template <typename _State>
    inline Distance keyDistance(const _State& q) {
        return CompoundStaticKeyDistance<1, _Spaces...>::accum(
            traversals_, q, std::get<0>(traversals_).keyDistance(std::get<0>(q)));
    }

    template <typename _Nearest, typename _Iter>
    void traverse(_Nearest& nearest, unsigned axis, _Iter min, _Iter max) {
        CompoundStaticTraversal<0, _Spaces...>::traverse(
            traversals_, nearest, axis, min, max);
    }
};

} // namespace unc::robotics::kdtree::detail
} // namespace unc::robotics::kdtree
} // namespace unc::robotics
} // namespace unc

#endif // UNC_ROBOTICS_KDTREE_COMPOUND_SPACE_HPP

