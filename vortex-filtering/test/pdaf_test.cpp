#include <gtest/gtest.h>
#include <vortex_filtering/filters/pdaf.hpp>
#include <iostream>

using SimplePDAF = PDAF<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;

TEST(PDAF, init)
{
    SimplePDAF pdaf;
    EXPECT_EQ(pdaf.gate_threshold_, 0.0);
    EXPECT_EQ(pdaf.prob_of_detection_, 0.0);
    EXPECT_EQ(pdaf.clutter_intensity_, 0.0);
}

TEST(PDAF, init_with_params)
{
    SimplePDAF pdaf(1.0, 0.9, 0.1);
    EXPECT_EQ(pdaf.gate_threshold_, 1.0);
    EXPECT_EQ(pdaf.prob_of_detection_, 0.9);
    EXPECT_EQ(pdaf.clutter_intensity_, 0.1);
}