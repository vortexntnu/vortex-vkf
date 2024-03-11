#include <gtest/gtest.h>
#include <vortex_filtering/types/type_aliases.hpp>
#include <iostream>
TEST(Types, x_2_z_1)
{
  using T = vortex::Types_xz<2, 1>;
  T::Vec_x x;
  T::Vec_z z;

  static_assert(std::is_same<decltype(x), Eigen::Vector2d>::value, "x is not a Vector2d");

  ASSERT_TRUE(true);
  ASSERT_TRUE(true);

  std::cout << "x: " << x << std::endl;
}