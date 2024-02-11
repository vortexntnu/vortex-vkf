#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/filters/ipda.hpp>
#include <vortex_filtering/plotting/utils.hpp>

using IPDA = vortex::filter::IPDA<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;

TEST(IPDA, ipda_runs)
{
  double gate_threshold = 1.12;
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;
  double probability_of_survival = 0.9;
  double last_detection_probability = 0.85;

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  std::vector<Eigen::Vector2d> meas = { { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 },
                                        { 0.0, 2.0 }, { 2.0, 0.0 }, { 2.0, 2.0 } };

  auto dyn_model = std::make_shared<vortex::models::ConstantVelocity<2>>(1.0);
  auto sen_model = std::make_shared<vortex::models::IdentitySensorModel<4, 2>>(1.0);

  auto [x_final, existence_pred, inside, outside, x_pred, z_pred, x_updated] =
      IPDA::step(x_est, meas, 1.0, dyn_model, sen_model, gate_threshold, prob_of_detection, probability_of_survival,
                 last_detection_probability, clutter_intensity);

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

  double existence_probability = IPDA::get_existence_probability(
      meas, probability_of_survival, last_detection_probability, prob_of_detection, clutter_intensity, z_pred);

  std::cout << "Existence probability: " << existence_probability << std::endl;
}