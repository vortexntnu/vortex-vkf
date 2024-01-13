#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

#include <vortex_filtering/filters/imm_filter.hpp>
#include <vortex_filtering/models/imm_model.hpp>

#include "gtest_assertions.hpp"

///////////////////////////////
// IMM Model Tests
///////////////////////////////

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

TEST(ImmModel, piMatC)
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

TEST(ImmModel, piMatD)
{
  using namespace vortex::models;
  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>, IdentityDynamicModel<2>>;

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

  IMM imm_model(std::make_tuple(model1, model2, model3), jump_mat, hold_times);

  Eigen::Matrix3d pi_mat_d_true;
  // clang-format off
  pi_mat_d_true << 64.0/105.0 + 1.0/(42.0 * exp(21.0)) + 11.0/(30.0 * exp(15.0)) , 1.0/5.0 - 1.0/(5.0 * exp(15.0)), 4.0/21.0 - 1.0/(42.0 * exp(21.0)) - 1.0/(6.0 * exp(15.0)),
                   64.0/105.0 + 6.0/(7.0 * exp(21.0)) - 22.0/(15.0 * exp(15.0))  , 1.0/5.0 + 4.0/(5.0 * exp(15.0)), 4.0/21.0 - 6.0/(7.0 * exp(21.0)) + 2.0/(3.0 * exp(15.0)),
                   64.0/105.0 - 41.0/(42.0 * exp(21.0)) + 11.0/(30.0 * exp(15.0)), 1.0/5.0 - 1.0/(5.0 * exp(15.0)), 4.0/21.0 + 41.0/(42.0 * exp(21.0)) - 1.0/(6.0 * exp(15.0));
  // clang-format on

  Eigen::Matrix3d pi_mat_d = imm_model.get_pi_mat_d(1.0);

  EXPECT_EQ(isApproxEqual(pi_mat_d, pi_mat_d_true, 1e-6), true);

  // Check that each row sums to 1
  for (int i = 0; i < pi_mat_d.rows(); i++) {
    EXPECT_NEAR(pi_mat_d.row(i).sum(), 1.0, 1e-9);
  }
}

TEST(ImmModel, stateSize)
{
  using namespace vortex::models;
  using TestModel = ImmModel<IdentityDynamicModel<3>, IdentityDynamicModel<2>, IdentityDynamicModel<4>>;

  EXPECT_EQ(TestModel::get_n_dim_x(), (std::array<int, 3>{3, 2, 4}));
}

///////////////////////////////
// IMM Filter Tests
///////////////////////////////

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

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(std::make_tuple(model1, model2), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(1.0);

  ImmFilter<IdentitySensorModel<2, 1>, IMM> imm_filter(imm_model, sensor_model);
}

TEST(ImmFilter, calculateMixingProbs)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  auto model1 = std::make_shared<IdentityDynamicModel<2>>(1.0);
  auto model2 = std::make_shared<IdentityDynamicModel<2>>(1.0);

  double dt = 1.0;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(std::make_tuple(model1, model2), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(dt);

  ImmFilter<IdentitySensorModel<2,1>, IMM> imm_filter(imm_model, sensor_model);

  Eigen::Vector2d model_weights;

  model_weights << 0.5, 0.5;
  // Since the weights are equal, the mixing probabilities should be equal to the discrete time Markov chain
  Eigen::Matrix2d mixing_probs_true = imm_model.get_pi_mat_d(dt);
  Eigen::Matrix2d mixing_probs = imm_filter.calculate_mixing_probs(model_weights, dt);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);

  // When all of the weight is in the first model, the probability that the previous model was the second model should be 0
  model_weights << 1, 0;
  mixing_probs_true << 1, 1, 0, 0;
  mixing_probs = imm_filter.calculate_mixing_probs(model_weights, dt);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);

}

TEST(ImmFilter, mixing)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  auto model_high_std = std::make_shared<IdentityDynamicModel<2>>(10);
  auto model_low_std  = std::make_shared<IdentityDynamicModel<2>>(0.1);

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>;

  IMM imm_model(std::make_tuple(model_high_std, model_low_std), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(dt);

  ImmFilter<IdentitySensorModel<2, 1>, IMM> imm_filter(imm_model, sensor_model);

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  std::vector<Gauss2d> x_est_prevs = {Gauss2d::Standard(), Gauss2d::Standard()};
  std::vector<Gauss2d> moment_based_approx = imm_filter.mixing(x_est_prevs, imm_model.get_pi_mat_d(dt));



}

TEST(ImmFilter, modeMatchedFilter)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  auto const_pos = std::make_shared<ConstantPosition<2>>(0.1);
  auto const_vel = std::make_shared<ConstantVelocity<1>>(0.1);

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using IMM = ImmModel<ConstantPosition<2>, ConstantVelocity<1>>;
  using IMMFilter = ImmFilter<IdentitySensorModel<2, 2>, IMM>;

  IMM imm_model(std::make_tuple(const_pos, const_vel), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  IMMFilter imm_filter(imm_model, sensor_model);

  std::vector<Gauss2d> x_est_prevs = {Gauss2d::Standard(), {{0, 1}, Eigen::Matrix2d::Identity()}};
  Eigen::Vector2d z_meas = {1,1};

  auto [x_est_upd, x_est_pred, z_est_pred] = imm_filter.mode_matched_filter(dt, x_est_prevs, z_meas);

  EXPECT_EQ(IMM::N_MODELS, x_est_upd.size());

  // Expect the second filter to predict closer to the measurement
  EXPECT_FALSE(isApproxEqual(z_est_pred.at(0).mean(), z_meas, 0.1));
  EXPECT_TRUE(isApproxEqual(z_est_pred.at(1).mean(), z_meas, 0.1));

}

TEST(ImmFilter, updateProbabilities)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  auto const_pos = std::make_shared<ConstantPosition<2>>(1);
  auto const_vel = std::make_shared<ConstantVelocity<1>>(1);

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using IMM = ImmModel<ConstantPosition<2>, ConstantVelocity<1>>;
  using IMMFilter = ImmFilter<IdentitySensorModel<2, 2>, IMM>;

  IMM imm_model(std::make_tuple(const_pos, const_vel), jump_mat, hold_times);

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  IMMFilter imm_filter(imm_model, sensor_model);

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  Eigen::Vector2d z_meas = {1,1};

  std::vector<Gauss2d> z_preds = {Gauss2d::Standard(), {{1, 1}, Eigen::Matrix2d::Identity()}};

  Eigen::Vector2d upd_weights = imm_filter.update_probabilities(z_preds, z_meas, dt, model_weights);

  EXPECT_GT(upd_weights(1), upd_weights(0));

}