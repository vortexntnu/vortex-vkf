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

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double std = 1.0;

  double dt;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<3>>;
  IMM imm_model(jump_mat, hold_times, {std}, {std});

  EXPECT_EQ(typeid(*imm_model.get_model<0>()), typeid(IdentityDynamicModel<2>));
  EXPECT_EQ(typeid(*imm_model.get_model<1>()), typeid(IdentityDynamicModel<3>));
  EXPECT_EQ(typeid(imm_model.f_d<0>(dt, Eigen::Vector2d::Zero())), typeid(Eigen::Vector2d));
  EXPECT_EQ(typeid(imm_model.f_d<1>(dt, Eigen::Vector3d::Zero())), typeid(Eigen::Vector3d));
  EXPECT_EQ(typeid(imm_model.Q_d<0>(dt, Eigen::Vector2d::Zero())), typeid(Eigen::Matrix2d));
  EXPECT_EQ(typeid(imm_model.Q_d<1>(dt, Eigen::Vector3d::Zero())), typeid(Eigen::Matrix3d));
}

TEST(ImmModel, piMatC)
{
  using namespace vortex::models;

  Eigen::Matrix3d jump_mat;
  // clang-format off
  jump_mat <<     0, 1.0/2, 1.0/2,
              1.0/3,     0, 2.0/3,
              5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;
  double std = 1.0;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(jump_mat, hold_times, {std}, {std}, {std});

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


  Eigen::Matrix3d jump_mat;
  // clang-format off
  jump_mat <<     0, 1.0/2, 1.0/2,
              1.0/3,     0, 2.0/3,
              5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;
  double std = 1.0;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(jump_mat, hold_times, {std}, {std}, {std});

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
  using DynModT   = vortex::models::IdentityDynamicModel<2>;
  using SensModT  = vortex::models::IdentitySensorModel<2, 1>;
  using ImmModelT = vortex::models::ImmModel<DynModT, DynModT>;

  vortex::filter::ImmFilter<SensModT, ImmModelT> imm_filter;

}

TEST(ImmFilter, calculateMixingProbs)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  double dt = 1.0;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double std = 1.0;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(jump_mat, hold_times, {std}, {std});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(dt);

  ImmFilter<IdentitySensorModel<2, 1>, IMM> imm_filter;

  Eigen::Vector2d model_weights;

  model_weights << 0.5, 0.5;
  // Since the weights are equal, the mixing probabilities should be equal to the discrete time Markov chain
  Eigen::Matrix2d mixing_probs_true = imm_model.get_pi_mat_d(dt);
  Eigen::Matrix2d mixing_probs      = imm_filter.calculate_mixing_probs(imm_model, model_weights, dt);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);

  // When all of the weight is in the first model, the probability that the previous model was the second model should be 0
  model_weights << 1, 0;
  mixing_probs_true << 1, 1, 0, 0;
  mixing_probs = imm_filter.calculate_mixing_probs(imm_model, model_weights, dt);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);
}

TEST(ImmFilter, mixing)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double high_std = 10;
  double low_std = 0.1;

  using IMM = ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<2>>;
  IMM imm_model(jump_mat, hold_times, {high_std}, {low_std});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 1>>(dt);

  ImmFilter<IdentitySensorModel<2, 1>, IMM> imm_filter;

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  std::vector<Gauss2d> x_est_prevs         = {Gauss2d::Standard(), Gauss2d::Standard()};
  std::vector<Gauss2d> moment_based_approx = imm_filter.mixing(x_est_prevs, imm_model.get_pi_mat_d(dt));
}

TEST(ImmFilter, modeMatchedFilter)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  // auto const_pos = std::make_shared<ConstantPosition<2>>(0.1);
  // auto const_vel = std::make_shared<ConstantVelocity<1>>(0.1);

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using IMM       = ImmModel<ConstantPosition<2>, ConstantVelocity<1>>;
  using IMMFilter = ImmFilter<IdentitySensorModel<2, 2>, IMM>;

  double std_pos = 0.1;
  double std_vel = 0.1;

  IMM imm_model(jump_mat, hold_times, {std_pos}, {std_vel});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  IMMFilter imm_filter;

  std::vector<Gauss2d> x_est_prevs = {Gauss2d::Standard(), {{0, 1}, Eigen::Matrix2d::Identity()}};
  Eigen::Vector2d z_meas           = {1, 1};

  auto [x_est_upd, x_est_pred, z_est_pred] = imm_filter.mode_matched_filter(imm_model, sensor_model, dt, x_est_prevs, z_meas);

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
  double std_pos = 1;
  double std_vel = 1;

  using IMM       = ImmModel<ConstantPosition<2>, ConstantVelocity<1>>;
  using IMMFilter = ImmFilter<IdentitySensorModel<2, 2>, IMM>;

  IMM imm_model(jump_mat, hold_times, {std_pos}, {std_vel});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  IMMFilter imm_filter;

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  Eigen::Vector2d z_meas = {1, 1};

  std::vector<Gauss2d> z_preds = {Gauss2d::Standard(), {{1, 1}, Eigen::Matrix2d::Identity()}};

  Eigen::Vector2d upd_weights = imm_filter.update_probabilities(imm_model, dt, z_preds, z_meas, model_weights);

  EXPECT_GT(upd_weights(1), upd_weights(0));
}