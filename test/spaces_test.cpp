#include <iostream>
#include "../src/_spaces.hpp"
#include "test.hpp"

TEST_CASE(L2Distance) {
    using namespace unc::robotics::kdtree;
    
    typedef L2Space<double, 2> Space;
    typedef Space::State State;

    Space space;
    State a(1.2, -3.1);
    State b(5.1, 6.7);

    EXPECT(space.distance(a, b)) == std::sqrt(
        std::pow(5.1 - 1.2, 2) +
        std::pow(6.7 + 3.1, 2));
}

TEST_CASE(SO3Distance) {
    using namespace unc::robotics::kdtree;

    typedef SO3Space<double> Space;
    typedef Space::State State;

    Space space;

    State a(1, 0, 0, 0);
    State b(0, 1, 0, 0);

    EXPECT(space.distance(a, b)) == M_PI_2;

    State c(std::sin(M_PI/6), std::cos(M_PI/6), 0, 0);

    EXPECT(std::abs(space.distance(a, c) - M_PI/3)) < 1e-13;
    EXPECT(std::abs(space.distance(b, c) - M_PI/6)) < 1e-13;
}

TEST_CASE(RatioWeightedDistance) {
    using namespace unc::robotics::kdtree;
    
    typedef RatioWeightedSpace<L2Space<double, 2>, std::ratio<17, 3>> Space;
    typedef Space::State State;

    Space space;
    State a(1.2, -3.1);
    State b(5.1, 6.7);

    EXPECT(space.distance(a, b)) == std::sqrt(
        std::pow(5.1 - 1.2, 2) +
        std::pow(6.7 + 3.1, 2)) * 17 / 3;
}

TEST_CASE(SE3Distance) {
    using namespace unc::robotics::kdtree;
    
    typedef SE3Space<double, 5, 3> Space;
    typedef Space::State State;

    Space space;
    // (
    //     (RatioWeightedSpace<SO3Space<double>>(SO3Space<double>())),
    //     (RatioWeightedSpace<L2Space<double, 3>>(L2Space<double,3>())));

    // State a(SO3Space<double>::State(1, 0, 0, 0),
    //         L2Space<double, 3>::State(-1.2, 3.4, 5.6));
    // State b(SO3Space<double>::State(0, 1, 0, 0),
    //         L2Space<double, 3>::State(9.8, -7.6, 5.4));

    State a({1, 0, 0, 0}, {-1.2, 3.4, 5.6});
    State b({0, 1, 0, 0}, {9.8, -7.6, 5.4});

    EXPECT(std::abs(space.distance(a, b) - (std::sqrt(
        std::pow(9.8 + 1.2, 2) +
        std::pow(3.4 + 7.6, 2) +
        std::pow(5.6 - 5.4, 2))*3 + M_PI_2*5))) < 1e-9 ;
}

TEST_CASE(MixedScalarCompound) {
    using namespace unc::robotics::kdtree;
    
    typedef L2Space<float, 2> FloatSpace;
    typedef L2Space<double, 2> DoubleSpace;

    EXPECT((std::is_same<float, CompoundSpace<FloatSpace, FloatSpace>::Distance>::value)) == true;
    EXPECT((std::is_same<double, CompoundSpace<FloatSpace, DoubleSpace>::Distance>::value)) == true;
    EXPECT((std::is_same<double, CompoundSpace<DoubleSpace, FloatSpace>::Distance>::value)) == true;
}
