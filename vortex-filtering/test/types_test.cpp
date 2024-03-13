#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/types/model_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>
TEST(Types, x_2_z_1)
{
  using T = vortex::Types_xz<2, 1>;
  T::Vec_x x;
  T::Vec_z z;

  static_assert(std::is_same<decltype(x), Eigen::Vector2d>::value, "x is not a Vector2d");
  static_assert(std::is_same<decltype(z), Eigen::Vector<double, 1>>::value, "z is not a Vector1d");

  ASSERT_TRUE(true);
  ASSERT_TRUE(true);

  std::cout << "x: " << x << std::endl;
}

TEST(Concepts, Model)
{
  constexpr size_t N_DIM_x = 5;
  constexpr size_t N_DIM_z = 4;
  constexpr size_t N_DIM_u = 3;
  constexpr size_t N_DIM_v = 2;
  constexpr size_t N_DIM_w = 1;

  using DynMod   = vortex::models::interface::DynamicModel<N_DIM_x, N_DIM_v, N_DIM_u>;
  using DynModLTV  = vortex::models::interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>;
  using SensMod = vortex::models::interface::SensorModel<N_DIM_x, N_DIM_z, N_DIM_w>;


  static_assert(vortex::concepts::model::DynamicModelLTV<DynModLTV, 5, 0, N_DIM_x>);

  ASSERT_TRUE(true);
}