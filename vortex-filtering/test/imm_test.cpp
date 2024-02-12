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
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double std = 1.0;

  double dt;


  vortex::models::ImmModel imm_model(jump_mat, hold_times, ConstantPosition(std), ConstantVelocity(std));

  EXPECT_EQ(typeid(*imm_model.get_model<0>()), typeid(ConstantPosition));
  EXPECT_EQ(typeid(*imm_model.get_model<1>()), typeid(ConstantVelocity));
  EXPECT_EQ(typeid(imm_model.f_d<0>(dt, Eigen::Vector2d::Zero())), typeid(Eigen::Vector2d));
  EXPECT_EQ(typeid(imm_model.f_d<1>(dt, Eigen::Vector3d::Zero())), typeid(Eigen::Vector3d));
  EXPECT_EQ(typeid(imm_model.Q_d<0>(dt, Eigen::Vector2d::Zero())), typeid(Eigen::Matrix2d));
  EXPECT_EQ(typeid(imm_model.Q_d<1>(dt, Eigen::Vector3d::Zero())), typeid(Eigen::Matrix3d));
}

TEST(ImmModel, piMatC)
{
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  Eigen::Matrix3d jump_mat;
  // clang-format off
  jump_mat <<     0, 1.0/2, 1.0/2,
              1.0/3,     0, 2.0/3,
              5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;
  double std = 1.0;

  vortex::models::ImmModel imm_model(jump_mat, hold_times, ConstantPosition(std), ConstantPosition(std), ConstantPosition(std));

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
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  Eigen::Matrix3d jump_mat;
  // clang-format off
  jump_mat <<     0, 1.0/2, 1.0/2,
              1.0/3,     0, 2.0/3,
              5.0/6, 1.0/6,     0;
  // clang-format on
  Eigen::Vector3d hold_times;
  hold_times << 6, 12, 18;
  double std = 1.0;

  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition, ConstantPosition>;
  ImmModelT imm_model(jump_mat, hold_times, {std}, {std}, {std});

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
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;
  using vortex::models::ConstantAcceleration;

  using TestModel = vortex::models::ImmModel<ConstantPosition, ConstantVelocity, ConstantAcceleration>;

  EXPECT_EQ(TestModel::N_DIM_x(0), (std::array<int, 3>{3, 2, 4}));
}

///////////////////////////////
// IMM Filter Tests
///////////////////////////////

TEST(ImmFilter, init)
{
  using vortex::models::ConstantPosition;
  using SensModT   = vortex::models::IdentitySensorModel<2, 1>;
  using ImmModelT  = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  using ImmFilterT = vortex::filter::ImmFilter<SensModT, ImmModelT>;

  EXPECT_EQ(ImmFilterT::N_MODELS, 2);
  EXPECT_EQ(ImmFilterT::N_DIM_z, SensModT::SensModI::N_DIM_z);
}

TEST(ImmFilter, calculateMixingProbs)
{
  using vortex::models::ConstantPosition;
  using namespace vortex::prob;

  double dt = 1.0;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double std = 1.0;

  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  ImmModelT imm_model(jump_mat, hold_times, {std}, {std});

  auto sensor_model = std::make_shared<vortex::models::IdentitySensorModel<2, 1>>(dt);

  using ImmFilterT = vortex::filter::ImmFilter<vortex::models::IdentitySensorModel<2, 1>, ImmModelT>;

  Eigen::Vector2d model_weights;

  model_weights << 0.5, 0.5;
  // Since the weights are equal, the mixing probabilities should be equal to the discrete time Markov chain
  Eigen::Matrix2d mixing_probs_true = imm_model.get_pi_mat_d(dt);
  Eigen::Matrix2d mixing_probs      = ImmFilterT::calculate_mixing_probs(imm_model.get_pi_mat_d(dt), model_weights);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);

  // When all of the weight is in the first model, the probability that the previous model was the second model should be 0
  model_weights << 1, 0;
  mixing_probs_true << 1, 1, 0, 0;
  mixing_probs = ImmFilterT::calculate_mixing_probs(imm_model.get_pi_mat_d(dt), model_weights);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);
}

TEST(ImmFilter, mixing)
{
  using vortex::models::ConstantPosition;
  using namespace vortex::prob;

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double high_std = 10;
  double low_std  = 0.1;

  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  ImmModelT imm_model(jump_mat, hold_times, {high_std}, {low_std});

  auto sensor_model = std::make_shared<vortex::models::IdentitySensorModel<2, 1>>(dt);

  using ImmFilterT = vortex::filter::ImmFilter<vortex::models::IdentitySensorModel<2, 1>, ImmModelT>;

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  std::tuple<Gauss2d, Gauss2d> x_est_prevs         = {Gauss2d::Standard(), Gauss2d::Standard()};
  std::tuple<Gauss2d, Gauss2d> moment_based_approx = ImmFilterT::mixing(x_est_prevs, imm_model.get_pi_mat_d(dt));
}

TEST(ImmFilter, modeMatchedFilter)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;

  using ImmModelT  = ImmModel<ConstantPosition, ConstantVelocity>;
  using ImmFilterT = vortex::filter::ImmFilter<IdentitySensorModel<2, 2>, ImmModelT>;

  double std_pos = 0.1;
  double std_vel = 0.1;

  ImmModelT imm_model(jump_mat, hold_times, {std_pos}, {std_vel});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);


  std::tuple<Gauss2d, Gauss4d> x_est_prevs = {Gauss2d::Standard(), {{0, 0, 0, 1}, Eigen::Matrix4d::Identity()}};
  Eigen::Vector2d z_meas           = {1, 1};

  auto [x_est_upd, x_est_pred, z_est_pred] = ImmFilterT::mode_matched_filter(imm_model, sensor_model, dt, x_est_prevs, z_meas);

  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_upd)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_pred)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, z_est_pred.size());

  // Expect the second filter to predict closer to the measurement
  EXPECT_FALSE(isApproxEqual(z_est_pred.at(0).mean(), z_meas, 0.1));
  EXPECT_TRUE(isApproxEqual(z_est_pred.at(1).mean(), z_meas, 0.1));
}

TEST(ImmFilter, updateProbabilities)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  auto const_pos = std::make_shared<ConstantPosition>(1);
  auto const_vel = std::make_shared<ConstantVelocity>(1);

  double dt = 1;

  Eigen::Matrix2d jump_mat;
  jump_mat << 0, 1, 1, 0;
  Eigen::Vector2d hold_times;
  hold_times << 1, 1;
  double std_pos = 1;
  double std_vel = 1;

  using ImmModelT = ImmModel<ConstantPosition, ConstantVelocity>;
  using ImmFilterT = vortex::filter::ImmFilter<IdentitySensorModel<2, 2>, ImmModelT>;

  ImmModelT imm_model(jump_mat, hold_times, {std_pos}, {std_vel});

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  Eigen::Vector2d z_meas = {1, 1};

  std::vector<Gauss2d> z_preds = {Gauss2d::Standard(), {{1, 1}, Eigen::Matrix2d::Identity()}};

  Eigen::Vector2d upd_weights = ImmFilterT::update_probabilities(imm_model.get_pi_mat_d(dt), z_preds, z_meas, model_weights);

  EXPECT_GT(upd_weights(1), upd_weights(0));
}