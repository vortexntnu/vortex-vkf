#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/filters/ipda.hpp>
#include <vortex_filtering/utils/plotting.hpp>

using ConstantVelocity    = vortex::models::ConstantVelocity;
using IdentitySensorModel = vortex::models::IdentitySensorModel<4, 2>;
using IPDA                = vortex::filter::IPDA<ConstantVelocity, IdentitySensorModel>;

TEST(IPDA, ipda_runs)
{
  IPDA::Config config;

  config.mahalanobis_threshold = 1.12;
  config.min_gate_threshold = 0.0;
  config.max_gate_threshold = 100.0;
  config.prob_of_detection = 0.8;
  config.clutter_intensity = 1.0;
  config.prob_of_survival = 0.9;

  double last_detection_probability = 0.85;

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  std::vector<Eigen::Vector2d> meas = { { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 },
                                        { 0.0, 2.0 }, { 2.0, 0.0 }, { 2.0, 2.0 } };

  ConstantVelocity dyn_model{1.0};
  IdentitySensorModel sen_model{1.0};

  auto [x_final, existence_pred, inside, outside, x_pred, z_pred, x_updated] =
      IPDA::step(dyn_model, sen_model, 1.0, x_est, meas, last_detection_probability, config);

  std::cout << "Existence probability: " << existence_pred << std::endl;
}

TEST(IPDA, get_existence_probability_is_calculating)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;
  double probability_of_survival = 0.9;
  double last_detection_probability = 0.9;

  std::vector<Eigen::Vector2d> meas = { { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 },
                                        { 0.0, 2.0 }, { 2.0, 0.0 }, { 2.0, 2.0 } };

  vortex::prob::Gauss2d z_pred;
  z_pred = vortex::prob::Gauss2d(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity() * 0.1);

  double existence_probability =
      IPDA::existence_probability(meas, probability_of_survival, last_detection_probability, prob_of_detection, clutter_intensity, z_pred);

  std::cout << "Existence probability: " << existence_probability << std::endl;
}