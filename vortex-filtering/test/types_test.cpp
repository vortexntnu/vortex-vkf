#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/models/state.hpp>
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
}

#include <Eigen/Dense>
#include <concepts>
#include <type_traits>

TEST(Concepts, MatConvertibleTo)
{
  static_assert(std::convertible_to<Eigen::Matrix3d, Eigen::Matrix2d>);
  static_assert(vortex::concepts::mat_convertible_to<Eigen::Matrix3d, Eigen::Matrix3d>);
  static_assert(!vortex::concepts::mat_convertible_to<Eigen::Matrix3d, Eigen::Matrix2d>);

  ASSERT_TRUE(true);
}

TEST(Concepts, MultiVarGaussLike)
{
  using vortex::prob::Gauss2d, vortex::prob::Gauss3d;

  static_assert(vortex::concepts::MultiVarGaussLike<Gauss2d, 2>);
  static_assert(vortex::concepts::MultiVarGaussLike<Gauss3d, 3>);

  static_assert(!vortex::concepts::MultiVarGaussLike<Gauss2d, 3>);

  using S      = StateName;
  using StateT = State<S::position, S::position, S::velocity, S::velocity>;
  static_assert(vortex::concepts::MultiVarGaussLike<StateT, StateT::N_STATES>);

  ASSERT_TRUE(true);
}

TEST(Concepts, Models)
{
  constexpr size_t N_DIM_x = 5;
  constexpr size_t N_DIM_z = 4;
  constexpr size_t N_DIM_u = 3;
  constexpr size_t N_DIM_v = 2;
  constexpr size_t N_DIM_w = 1;

  using DynMod     = vortex::models::interface::DynamicModel<N_DIM_x, N_DIM_u, N_DIM_v>;
  using DynModLTV  = vortex::models::interface::DynamicModelLTV<N_DIM_x, N_DIM_u, N_DIM_v>;
  using SensMod    = vortex::models::interface::SensorModel<N_DIM_x, N_DIM_z, N_DIM_w>;
  using SensModLTV = vortex::models::interface::SensorModelLTV<N_DIM_x, N_DIM_z, N_DIM_w>;

  static_assert(vortex::concepts::DynamicModel<DynMod, N_DIM_x, N_DIM_u, N_DIM_v>);
  static_assert(vortex::concepts::DynamicModelLTV<DynModLTV, N_DIM_x, N_DIM_u, N_DIM_v>);
  static_assert(vortex::concepts::SensorModel<SensMod, N_DIM_x, N_DIM_z, N_DIM_w>);
  static_assert(vortex::concepts::SensorModelLTV<SensModLTV, N_DIM_x, N_DIM_z, N_DIM_w>);

  static_assert(vortex::concepts::DynamicModelWithDefinedSizes<DynMod>);
  static_assert(vortex::concepts::DynamicModelLTVWithDefinedSizes<DynModLTV>);
  static_assert(vortex::concepts::SensorModelWithDefinedSizes<SensMod>);
  static_assert(vortex::concepts::SensorModelLTVWithDefinedSizes<SensModLTV>);

  static_assert(vortex::concepts::DynamicModelWithDefinedSizes<DynModLTV>);
  static_assert(!vortex::concepts::DynamicModelLTVWithDefinedSizes<DynMod>);
  static_assert(vortex::concepts::SensorModelWithDefinedSizes<SensModLTV>);
  static_assert(!vortex::concepts::SensorModelLTVWithDefinedSizes<SensMod>);

  static_assert(!vortex::concepts::DynamicModel<DynMod, 1, 1, 1>);
  static_assert(!vortex::concepts::DynamicModelLTV<DynModLTV, 1, 1, 1>);
  static_assert(!vortex::concepts::SensorModel<SensMod, 1, 1, 1>);
  static_assert(!vortex::concepts::SensorModelLTV<SensModLTV, 1, 1, 1>);

  ASSERT_TRUE(true);
}