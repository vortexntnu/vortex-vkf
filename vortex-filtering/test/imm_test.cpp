#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

#include <vortex_filtering/filters/imm_filter.hpp>
#include <vortex_filtering/models/imm_model.hpp>

#include "gtest_assertions.hpp"

TEST(ImmModel, init)
{
  using namespace vortex::models;

  auto model_2d = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model_3d = std::make_shared<IdentityDynamicModel<3>>(1.0);

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<3>> imm_model(std::make_tuple(model_2d, model_3d), jump_mat, hold_times);

  EXPECT_EQ(typeid(*imm_model.get_model<0>()), typeid(IdentityDynamicModel<2>));
  EXPECT_EQ(typeid(*imm_model.get_model<1>()), typeid(IdentityDynamicModel<3>));
  EXPECT_EQ(typeid(imm_model.f_d<0>(1.0, Eigen::Vector2d::Zero())), typeid(Eigen::Vector2d));
  EXPECT_EQ(typeid(imm_model.f_d<1>(1.0, Eigen::Vector3d::Zero())), typeid(Eigen::Vector3d));
  EXPECT_EQ(typeid(imm_model.Q_d<0>(1.0, Eigen::Vector2d::Zero())), typeid(Eigen::Matrix2d));
  EXPECT_EQ(typeid(imm_model.Q_d<1>(1.0, Eigen::Vector3d::Zero())), typeid(Eigen::Matrix3d));
}

TEST(ImmModel, pi_mat_c)
{
  using namespace vortex::models;

  auto model1 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model2 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model3 = std::make_shared<IdentityDynamicModel<2>>(1.0);

  Eigen::Matrix3d jump_mat;
  // clang-format off
    jump_mat <<     0, 1.0/2, 1.0/2,
                1.0/3,     0, 2.0/3,
                5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;

  ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>, IdentityDynamicModel<2>> imm_model(std::make_tuple(model1, model2, model3), jump_mat, hold_times);

  Eigen::Matrix3d pi_mat_c;
  // clang-format off
    pi_mat_c << -6,   3,   3,
                 4, -12,   8,
                15,   3, -18;
  // clang-format on

  EXPECT_EQ(imm_model.get_pi_mat_c(), pi_mat_c);
}

TEST(ImmModel, pi_mat_d)
{
  using namespace vortex::models;

  auto model1 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model2 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model3 = std::make_shared<IdentityDynamicModel<2>>(1.0);

  Eigen::Matrix3d jump_mat;
  // clang-format off
    jump_mat <<     0, 1.0/2, 1.0/2,
                1.0/3,     0, 2.0/3,
                5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;

  ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>, IdentityDynamicModel<2>> imm_model(std::make_tuple(model1, model2, model3), jump_mat, hold_times);

  Eigen::Matrix3d pi_mat_d;
  // clang-format off
    pi_mat_d << 64.0/105.0 + 1.0/(42.0 * exp(21.0)) + 11.0/(30.0 * exp(15.0)) , 1.0/5.0 - 1.0/(5.0 * exp(15.0)), 4.0/21.0 - 1.0/(42.0 * exp(21.0)) - 1.0/(6.0 * exp(15.0)),
                64.0/105.0 + 6.0/(7.0 * exp(21.0)) - 22.0/(15.0 * exp(15.0))  , 1.0/5.0 + 4.0/(5.0 * exp(15.0)), 4.0/21.0 - 6.0/(7.0 * exp(21.0)) + 2.0/(3.0 * exp(15.0)),
                64.0/105.0 - 41.0/(42.0 * exp(21.0)) + 11.0/(30.0 * exp(15.0)), 1.0/5.0 - 1.0/(5.0 * exp(15.0)), 4.0/21.0 + 41.0/(42.0 * exp(21.0)) - 1.0/(6.0 * exp(15.0));
  // clang-format on

  EXPECT_EQ(isApproxEqual(imm_model.get_pi_mat_d(1.0), pi_mat_d, 1e-6), true);

  // Check that each row sums to 1
  for (int i = 0; i < pi_mat_d.rows(); i++) {
    EXPECT_EQ(pi_mat_d.row(i).sum(), 1.0);
  }
}

TEST(ImmModel, state_size)
{
  using namespace vortex::models;
  using testModel = ImmModel<IdentityDynamicModel<3>, IdentityDynamicModel<2>, IdentityDynamicModel<4>>;

  EXPECT_EQ(testModel::get_n_dim_x(), (std::array<int, 3>{3, 2, 4}));
}

TEST(ImmFilter, init)
{
  using namespace vortex::models;
  using namespace vortex::filter;

  auto model1 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model2 = std::make_shared<IdentityDynamicModel<2>>(1.0);

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>> imm_model(std::make_tuple(model1, model2), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(1.0);

  ImmFilter<ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>, IdentitySensorModel<2, 1>> imm_filter(imm_model, sensor_model);
}