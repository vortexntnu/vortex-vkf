#include "gtest_assertions.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <numbers>
#include <vortex_filtering/filters/imm_filter.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/utils/printers.hpp>

///////////////////////////////
// IMM Model Tests
///////////////////////////////

TEST(ImmModel, initWithStateNames)
{
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};

  double std = 1.0;
  double dt  = 1.0;

  using ST = vortex::models::StateType;
  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantVelocity>;
  ImmModelT imm_model{jump_mat,
                      hold_times,
                      std::tuple{ConstantPosition(std), std::array{ST::position, ST::position}},
                      std::tuple{ConstantVelocity(std), std::array{ST::position, ST::position, ST::velocity, ST::velocity}}};

  EXPECT_EQ(typeid(imm_model.get_model<0>()), typeid(ConstantPosition));
  EXPECT_EQ(typeid(imm_model.get_model<1>()), typeid(ConstantVelocity));
  EXPECT_EQ(typeid(imm_model.f_d<0>(dt, Eigen::Vector2d::Zero())), typeid(Eigen::Vector2d));
  EXPECT_EQ(typeid(imm_model.f_d<1>(dt, Eigen::Vector4d::Zero())), typeid(Eigen::Vector4d));
}

TEST(ImmModel, piMatC)
{
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  // clang-format off
  Eigen::Matrix3d jump_mat{
    {0    , 1.0/2, 1.0/2}, 
    {1.0/3, 0    , 2.0/3}, 
    {5.0/6, 1.0/6, 0    }
  };
  // clang-format on

  Eigen::Vector3d hold_times{1.0 / 6, 1.0 / 12, 1.0 / 18};
  double std = 1.0;

  using ST        = vortex::models::StateType;
  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition, ConstantPosition>;
  ImmModelT::StateNames state_names{{ST::position, ST::position}, {ST::position, ST::position}, {ST::position, ST::position}};

  ImmModelT imm_model(jump_mat, hold_times, ConstantPosition(std), ConstantPosition(std), ConstantPosition(std), state_names);

  // clang-format off
  Eigen::Matrix3d pi_mat_c{
    {-6,  3 ,  3 },
    {4 , -12,  8 },
    {15,  3 , -18}
  };
  // clang-format on

  EXPECT_EQ(imm_model.get_pi_mat_c(), pi_mat_c);
}

TEST(ImmModel, piMatD)
{
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  // clang-format off
  Eigen::Matrix3d jump_mat{
    {0    , 1.0/2, 1.0/2}, 
    {1.0/3, 0    , 2.0/3}, 
    {5.0/6, 1.0/6, 0    }
  };
  // clang-format on
  Eigen::Vector3d hold_times{1.0 / 6, 1.0 / 12, 1.0 / 18};
  double std = 1.0;

  using ST        = vortex::models::StateType;
  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition, ConstantPosition>;
  ImmModelT::StateNames state_names{{ST::position, ST::position}, {ST::position, ST::position}, {ST::position, ST::position}};

  ImmModelT imm_model(jump_mat, hold_times, ConstantPosition(std), ConstantPosition(std), ConstantPosition(std), state_names);

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
  using vortex::models::ConstantAcceleration;
  using vortex::models::ConstantPosition;
  using vortex::models::ConstantVelocity;

  using TestModel = vortex::models::ImmModel<ConstantPosition, ConstantVelocity, ConstantAcceleration>;

  EXPECT_EQ(TestModel::N_DIMS_x, (std::array<int, 3>{2, 4, 6}));
}

///////////////////////////////
// IMM Filter Tests
///////////////////////////////

TEST(ImmFilter, init)
{
  using vortex::models::ConstantPosition;
  using SensModT   = vortex::models::IdentitySensorModel<2, 1>;
  using ImmModelT  = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, SensModT>;

  EXPECT_EQ(ImmFilterT::N_MODELS, 2u);
  EXPECT_EQ(ImmFilterT::N_DIM_z, SensModT::N_DIM_z);
}

TEST(ImmFilter, calculateMixingProbs)
{
  using vortex::models::ConstantPosition;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};

  double dt  = 1.0;
  double std = 1.0;

  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  using ST        = vortex::models::StateType;
  ImmModelT imm_model(jump_mat, hold_times, {{std}, {ST::position, ST::position}}, {{std}, {ST::position, ST::position}});

  auto sensor_model = std::make_shared<vortex::models::IdentitySensorModel<2, 1>>(dt);

  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, vortex::models::IdentitySensorModel<2, 1>>;

  Eigen::Vector2d model_weights = {0.5, 0.5};

  // Since the weights are equal, the mixing probabilities should be equal to the discrete time Markov chain
  Eigen::Matrix2d mixing_probs_true = imm_model.get_pi_mat_d(dt);
  Eigen::Matrix2d mixing_probs      = ImmFilterT::calculate_mixing_probs(imm_model.get_pi_mat_d(dt), model_weights);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);

  // When all of the weight is in the first model, the probability that the previous model was the second model should be 0
  model_weights     = {1, 0};
  mixing_probs_true = Eigen::Matrix2d{{1, 1}, {0, 0}};

  mixing_probs = ImmFilterT::calculate_mixing_probs(imm_model.get_pi_mat_d(dt), model_weights);
  EXPECT_EQ(isApproxEqual(mixing_probs, mixing_probs_true, 1e-6), true);
}

TEST(ImmFilter, mixing_two_of_the_same_model)
{
  using vortex::models::ConstantPosition;
  using vortex::prob::Gauss2d;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};

  double dt      = 1;
  double pos_std = 1;

  using ImmModelT = vortex::models::ImmModel<ConstantPosition, ConstantPosition>;
  using ST        = vortex::models::StateType;
  ImmModelT imm_model(jump_mat, hold_times, {{pos_std}, {ST::position, ST::position}}, {{pos_std}, {ST::position, ST::position}});

  auto sensor_model = std::make_shared<vortex::models::IdentitySensorModel<2, 1>>(dt);

  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, vortex::models::IdentitySensorModel<2, 1>>;

  Eigen::Vector2d model_weights{0.5, 0.5};

  Gauss2d x_est_prev_1{Gauss2d::Standard()};
  Gauss2d x_est_prev_2{{1, 0}, Eigen::Matrix2d::Identity() * 100};

  std::tuple<Gauss2d, Gauss2d> x_est_prevs{x_est_prev_1, x_est_prev_2};

  auto [x_est_1, x_est_2] = ImmFilterT::mixing(x_est_prevs, imm_model.get_pi_mat_d(dt), imm_model.get_all_state_names());

  // The high uncertainty in the second model should make it's position estimate move more towards the first
  // model than the first model moves towards the second
  std::cout << "x_est_prev_1:\n" << x_est_prev_1 << std::endl;
  std::cout << "x_est_prev_2:\n" << x_est_prev_2 << std::endl;
  std::cout << "x_est_1:\n" << x_est_1 << std::endl;
  std::cout << "x_est_2:\n" << x_est_2 << std::endl;

  EXPECT_LT((x_est_2.mean() - x_est_prev_2.mean()).norm(), (x_est_1.mean() - x_est_prev_1.mean()).norm());
}

TEST(ImmFilter, mixing_two_different_models)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};

  using ImmModelT  = ImmModel<ConstantPosition, ConstantVelocity>;
  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, IdentitySensorModel<2, 2>>;

  double dt      = 1;
  double std_pos = 0.1;
  double std_vel = 0.1;

  using ST = vortex::models::StateType;
  ImmModelT imm_model{jump_mat,
                      hold_times,
                      std::tuple{ConstantPosition(std_pos), std::array{ST::position, ST::position}},
                      std::tuple{ConstantVelocity(std_vel), std::array{ST::position, ST::position, ST::velocity, ST::velocity}}};

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  Gauss2d x_est_prev_1{Gauss2d::Standard()};
  Gauss4d x_est_prev_2{{1, 0, 0, 0}, Eigen::Matrix4d::Identity() * 100};

  std::tuple<Gauss2d, Gauss4d> x_est_prevs{x_est_prev_1, x_est_prev_2};

  // clang-format off
  using ST = vortex::models::StateType;
  vortex::models::StateMap states_min_max{
    {ST::velocity, {-100 , 100}}
  };
  // clang-format on

  auto [x_est_1, x_est_2] = ImmFilterT::mixing(x_est_prevs, imm_model.get_pi_mat_d(dt), imm_model.get_all_state_names(), states_min_max);

  std::cout << "x_est_prev_1:\n" << x_est_prev_1 << std::endl;
  std::cout << "x_est_prev_2:\n" << x_est_prev_2 << std::endl;
  std::cout << "x_est_1:\n" << x_est_1 << std::endl;
  std::cout << "x_est_2:\n" << x_est_2 << std::endl;

  // EXPECT_LT((x_est_2.mean().head<2>() - x_est_prev_2.mean().head<2>()).norm(), (x_est_1.mean() - x_est_prev_1).norm());
}

TEST(ImmFilter, modeMatchedFilter)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};

  using ImmModelT  = ImmModel<ConstantPosition, ConstantVelocity>;
  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, IdentitySensorModel<2, 2>>;

  double dt      = 1;
  double std_pos = 0.1;
  double std_vel = 0.1;

  using ST = vortex::models::StateType;
  ImmModelT imm_model{jump_mat,
                      hold_times,
                      std::tuple{ConstantPosition(std_pos), std::array{ST::position, ST::position}},
                      std::tuple{ConstantVelocity(std_vel), std::array{ST::position, ST::position, ST::velocity, ST::velocity}}};

  IdentitySensorModel<2, 2> sensor_model{dt};

  std::tuple<Gauss2d, Gauss4d> x_est_prevs = {Gauss2d::Standard(), {{0, 0, 0.9, 0}, Eigen::Matrix4d::Identity()}};
  Eigen::Vector2d z_meas                   = {1, 0};

  auto [x_est_upds, x_est_preds, z_est_preds] = ImmFilterT::mode_matched_filter(imm_model, sensor_model, dt, x_est_prevs, z_meas);

  std::cout << "x_est_upds:\n" << x_est_upds << std::endl;
  std::cout << "x_est_preds:\n" << x_est_preds << std::endl;
  std::cout << "z_est_preds:\n" << z_est_preds << std::endl;

  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_upds)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_preds)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, z_est_preds.size());

  // Expect the second filter to predict closer to the measurement
  double dist_to_meas_0 = (std::get<0>(x_est_preds).mean().head<2>() - z_meas).norm();
  double dist_to_meas_1 = (std::get<1>(x_est_preds).mean().head<2>() - z_meas).norm();

  EXPECT_LT(dist_to_meas_1, dist_to_meas_0);
}

TEST(ImmFilter, updateProbabilities)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  double dt = 1;

  Eigen::Matrix2d jump_mat{{0, 1}, {1, 0}};
  Eigen::Vector2d hold_times{1, 1};
  double std_pos = 1;
  double std_vel = 1;

  using ImmModelT  = ImmModel<ConstantPosition, ConstantVelocity>;
  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, IdentitySensorModel<2, 2>>;

  using ST = vortex::models::StateType;
  ImmModelT imm_model{jump_mat,
                      hold_times,
                      std::tuple{ConstantPosition(std_pos), std::array{ST::position, ST::position}},
                      std::tuple{ConstantVelocity(std_vel), std::array{ST::position, ST::position, ST::velocity, ST::velocity}}};

  auto sensor_model = std::make_shared<IdentitySensorModel<2, 2>>(dt);

  Eigen::Vector2d model_weights;
  model_weights << 0.5, 0.5;

  Eigen::Vector2d z_meas = {1, 1};

  std::array<Gauss2d, 2> z_preds = {Gauss2d::Standard(), {{1, 1}, Eigen::Matrix2d::Identity()}};

  Eigen::Vector2d upd_weights = ImmFilterT::update_probabilities(imm_model.get_pi_mat_d(dt), z_preds, z_meas, model_weights);

  EXPECT_GT(upd_weights(1), upd_weights(0));
}

TEST(ImmFilter, step)
{
  using namespace vortex::models;
  using namespace vortex::filter;
  using namespace vortex::prob;

  double dt = 1;
  // clang-format off
  Eigen::Matrix3d jump_mat{
    {0, 0.5, 0.5}, 
    {0.5, 0, 0.5}, 
    {0.5, 0.5, 0}};
  // clang-format on
  Eigen::Vector3d hold_times{10, 10, 10};
  double std_pos       = 0.1;
  double std_vel       = 0.1;
  double std_turn_rate = 0.1;

  using ImmModelT  = ImmModel<ConstantPosition, ConstantVelocity, CoordinatedTurn>;
  using ImmFilterT = vortex::filter::ImmFilter<ImmModelT, IdentitySensorModel<2, 2>>;

  ImmModelT imm_model{jump_mat,
                      hold_times,
                      {ConstantPosition(std_pos), ConstantPosition::StateNames},
                      {ConstantVelocity(std_vel), ConstantVelocity::StateNames},
                      {CoordinatedTurn(std_vel, std_turn_rate), CoordinatedTurn::StateNames}};

  IdentitySensorModel<2, 2> sensor_model{dt};

  Eigen::Vector3d model_weights{1 / 3.0, 1 / 3.0, 1 / 3.0};

  std::tuple<Gauss2d, Gauss4d, Gauss5d> x_est_prevs = {
      Gauss2d::Standard(), {{0, 0, 0.9, 0}, Eigen::Matrix4d::Identity()}, {{0, 0, 0.9, 0, 1}, Eigen::Matrix<double, 5, 5>::Identity()}};
  Eigen::Vector2d z_meas = {1, 0};

  StateMap states_min_max{{StateType::velocity, {-10, 10}}, {StateType::turn_rate, {-std::numbers::pi, std::numbers::pi}}};

  auto [weights_upd, x_est_upds, x_est_preds, z_est_preds] = ImmFilterT::step(imm_model, sensor_model, dt, x_est_prevs, z_meas, model_weights, states_min_max);

  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_upds)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, std::tuple_size<decltype(x_est_preds)>::value);
  EXPECT_EQ(ImmFilterT::N_MODELS, z_est_preds.size());
  EXPECT_EQ(ImmFilterT::N_MODELS, weights_upd.size());

  for (int i = 2; i < 50; i++) {
    z_meas << i, 0;
    std::tie(weights_upd, x_est_upds, std::ignore, std::ignore) =
        ImmFilterT::step(imm_model, sensor_model, dt, x_est_upds, z_meas, weights_upd, states_min_max);
  }

  std::cout << "weights_upd:\n" << weights_upd << std::endl;
  std::cout << "x_est_upds:\n" << x_est_upds << std::endl;

  // Expect the constant velocity model to have the highest probability
  EXPECT_GT(weights_upd(1), weights_upd(0));
  EXPECT_GT(weights_upd(1), weights_upd(2));
}