#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/filters/ipda.hpp>
#include <vortex_filtering/plotting/utils.hpp>

using IPDA = vortex::filter::IPDA<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;

TEST(IPDA, ipda_runs)
{
}

TEST(IPDA, get_existence_probability_is_calculating)
{
}