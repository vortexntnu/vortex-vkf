#include <gnuplot-iostream.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/utils/ellipse.hpp>

using ConstantVelocity = vortex::models::ConstantVelocity;
using IdentitySensorModel = vortex::models::IdentitySensorModel<4, 2>;
using PDAF                = vortex::filter::PDAF<ConstantVelocity, IdentitySensorModel>;

// testing the get_weights function
TEST(PDAF, get_weights_is_calculating)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
  Eigen::Array2Xd meas = {{0.0, 2.0}, {0.0, 1.0}};

  Eigen::VectorXd weights = PDAF::get_weights(meas, z_pred, prob_of_detection, clutter_intensity);

  std::cout << "weights: " << weights << std::endl;

  EXPECT_EQ(weights.size(), 3);
}

TEST(PDAF, if_no_clutter_first_weight_is_zero)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 0.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
  Eigen::Array2Xd meas = {{0.0, 2.0}, {0.0, 1.0}};

  Eigen::VectorXd weights = PDAF::get_weights(meas, z_pred, prob_of_detection, clutter_intensity);

  std::cout << "weights: " << weights << std::endl;

  EXPECT_EQ(weights(0), 0.0);
}

TEST(PDAF, weights_are_decreasing_with_distance)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
  Eigen::Array2Xd meas = {{2.0, 3.0, 4.0}, {1.0, 1.0, 1.0}};

  Eigen::VectorXd weights = PDAF::get_weights(meas, z_pred, prob_of_detection, clutter_intensity);

  std::cout << "weights: " << weights << std::endl;

  EXPECT_GT(weights(1), weights(2));
  EXPECT_GT(weights(2), weights(3));
}

// testing the get_weighted_average function
TEST(PDAF, get_weighted_average_is_calculating)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
  vortex::prob::Gauss4d x_pred(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  Eigen::Array2Xd meas                                     = {{0.0, 2.0}, {0.0, 1.0}};
  std::vector<vortex::prob::Gauss4d> updated_states = {
    vortex::prob::Gauss4d(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity()),
    vortex::prob::Gauss4d(Eigen::Vector4d(1.0, 1.0, 1.0, 1.0), Eigen::Matrix4d::Identity())
  };

  vortex::prob::Gauss4d weighted_average =
      PDAF::get_weighted_average(meas, updated_states, z_pred, x_pred, prob_of_detection, clutter_intensity);

  std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_y_axis)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
  vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  Eigen::Array2Xd meas = {{1.0}, {2.0}};

  std::vector<vortex::prob::Gauss4d> updated_states = { vortex::prob::Gauss4d(Eigen::Vector4d(1.0, 1.5, 0.0, 0.0),
                                                                              Eigen::Matrix4d::Identity()) };

  vortex::prob::Gauss4d weighted_average =
      PDAF::get_weighted_average(meas, updated_states, z_pred, x_pred, prob_of_detection, clutter_intensity);

  EXPECT_GT(weighted_average.mean()(1), x_pred.mean()(1));
  EXPECT_GT(weighted_average.mean()(1), z_pred.mean()(1));
  EXPECT_LT(weighted_average.mean()(1), meas(1, 0));
  EXPECT_LT(weighted_average.mean()(1), updated_states[0].mean()(1));

  std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_x_axis)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
  vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  Eigen::Array2Xd meas = {{2.0}, {1.0}};

  std::vector<vortex::prob::Gauss4d> updated_states = { vortex::prob::Gauss4d(Eigen::Vector4d(1.5, 1.0, 0.0, 0.0),
                                                                              Eigen::Matrix4d::Identity()) };

  vortex::prob::Gauss4d weighted_average =
      PDAF::get_weighted_average(meas, updated_states, z_pred, x_pred, prob_of_detection, clutter_intensity);

  EXPECT_GT(weighted_average.mean()(0), x_pred.mean()(0));
  EXPECT_GT(weighted_average.mean()(0), z_pred.mean()(0));
  EXPECT_LT(weighted_average.mean()(0), meas(0, 0));
  EXPECT_LT(weighted_average.mean()(0), updated_states[0].mean()(0));

  std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_both_axes)
{
  double prob_of_detection = 0.8;
  double clutter_intensity = 1.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
  vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
  Eigen::Array2Xd meas = {{2.0}, {2.0}};

  std::vector<vortex::prob::Gauss4d> updated_states = { vortex::prob::Gauss4d(Eigen::Vector4d(1.5, 1.5, 0.0, 0.0),
                                                                              Eigen::Matrix4d::Identity()) };

  vortex::prob::Gauss4d weighted_average =
      PDAF::get_weighted_average(meas, updated_states, z_pred, x_pred, prob_of_detection, clutter_intensity);

  EXPECT_GT(weighted_average.mean()(0), x_pred.mean()(0));
  EXPECT_GT(weighted_average.mean()(0), z_pred.mean()(0));
  EXPECT_LT(weighted_average.mean()(0), meas(0, 0));
  EXPECT_LT(weighted_average.mean()(0), updated_states[0].mean()(0));

  EXPECT_GT(weighted_average.mean()(1), x_pred.mean()(1));
  EXPECT_GT(weighted_average.mean()(1), z_pred.mean()(1));
  EXPECT_LT(weighted_average.mean()(1), meas(1, 0));
  EXPECT_LT(weighted_average.mean()(1), updated_states[0].mean()(1));

  std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

// testing the apply_gate function
TEST(PDAF, apply_gate_is_calculating)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 1.8,
                         }};

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
  // clang-format off
  Eigen::Array2Xd meas = {{0.0, 1.0, 1.0, 0.0, 2.0, 2.0}, 
                          {1.0, 0.0, 1.0, 2.0, 0.0, 2.0}};
  // clang-format on

  auto gated = PDAF::apply_gate(meas, z_pred, config);
}

TEST(PDAF, apply_gate_is_separating_correctly)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 3,
                         }};
  Eigen::Matrix2d cov;
  cov << 1.0, 0.0, 0.0, 4.0;

  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), cov);
  // clang-format off
  Eigen::Array2Xd meas = {{0.0, 4.0}, 
                          {4.0, 0.0}};
  // clang-format on

  auto gated = PDAF::apply_gate(meas, z_pred, config);

  EXPECT_TRUE(gated(0));
  EXPECT_FALSE(gated(1));

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";
  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  gp << "set object " << ++object_counter << " circle center " << z_pred.mean()(0) << "," << z_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";
  gp << "replot\n";

  vortex::utils::Ellipse prediction = vortex::plotting::gauss_to_ellipse(z_pred, mahalanobis_threshold);

  gp << "set object " << ++object_counter << " ellipse center " << prediction.x() << "," << prediction.y() << " size "
     << prediction.major_axis() << "," << prediction.minor_axis() << " angle " << prediction.angle_deg()
     << "fs empty border lc rgb 'cyan'\n";
  gp << "replot\n";
#endif
}

TEST(PDAF, apply_gate_is_separating_correctly_2)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 2.1,
                         }};
  vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{0.0, 1.0, 1.0, 0.0, 2.0, 2.0}, 
                          {1.0, 0.0, 1.0, 2.0, 0.0, 2.0}};
  // clang-format on

  auto gated = PDAF::apply_gate(meas, z_pred, config);

  EXPECT_EQ(gated.count(), 5u);

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";
  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  gp << "set object " << ++object_counter << " circle center " << z_pred.mean()(0) << "," << z_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";
  gp << "replot\n";

  vortex::utils::Ellipse prediction = vortex::plotting::gauss_to_ellipse(z_pred, mahalanobis_threshold);

  gp << "set object " << ++object_counter << " ellipse center " << prediction.x() << "," << prediction.y() << " size "
     << prediction.major_axis() << "," << prediction.minor_axis() << " angle " << prediction.angle_deg()
     << "fs empty border lc rgb 'cyan'\n";
  gp << "replot\n";
#endif
}

// testing the predict_next_state function
TEST(PDAF, predict_next_state_is_calculating)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 1.12,
                             .prob_of_detection     = 0.8,
                             .clutter_intensity     = 1.0,
                         }};

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{0.0, 1.0, 1.0, 0.0, 2.0, 2.0}, 
                          {1.0, 0.0, 1.0, 2.0, 0.0, 2.0}};

  ConstantVelocity dyn_model{ 1.0 };
  IdentitySensorModel sen_model{ 1.0 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF, predict_next_state_2)
{
  PDAF::Config config = {.pdaf =
        {
          .mahalanobis_threshold = 2.0,
          .prob_of_detection     = 0.8,
          .clutter_intensity     = 1.0,
        }
  };

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(1.0, 1.5, -2.0, 0.0), Eigen::Matrix4d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{-3.0, 0.0, -1.2, -2.0, 2.0, -1.0}, 
                          {-3.0, 0.0, 1.5, -2.0, 0.0, 1.0}};
  // clang-format on

  ConstantVelocity dyn_model{ 1.0 };
  IdentitySensorModel sen_model{ 0.5 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF, predict_next_state_3_1)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 4.0,
                             .prob_of_detection     = 0.9,
                             .clutter_intensity     = 1.0,
                         }};

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(0.5, 0.5, -0.75, -0.75), Eigen::Matrix4d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{0.0, 0.2, 0.8, 0.5, 4.2, 1.4}, 
                          {0.5, 0.2, 2.3, 0.0, 2.7, 2.5}};
  // clang-format on

  ConstantVelocity dyn_model{ 0.5 };
  IdentitySensorModel sen_model{ 1.0 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << 0.5 << "," << 0.5 << " size " << 0.05
     << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF, predict_next_state_3_2)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 4.0,
                             .prob_of_detection     = 0.9,
                             .clutter_intensity     = 1.0,
                         }};
  vortex::prob::Gauss4d x_est(Eigen::Vector4d(-0.00173734, 0.0766262, -0.614584, -0.57184),
                              Eigen::Matrix4d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{-0.5, -0.23, -2.0, 0.0, 0.14, -2.5}, 
                          {2.0 ,   0.5,  3.4, 1.3, 0.5 , 0.89}};
  // clang-format on

  ConstantVelocity dyn_model{ 0.5 };
  IdentitySensorModel sen_model{ 1.0 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  // old state from 3_1
  gp << "set object " << ++object_counter << " circle center " << 0.5 << "," << 0.5 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << 0.5 << "," << 0.5 << " to " << -0.00173734 << "," << 0.0766262
     << " nohead lc rgb 'orange-red'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF, predict_next_state_3_3)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 4.0,
                             .prob_of_detection     = 0.9,
                             .clutter_intensity     = 1.0,
                         }};
  vortex::prob::Gauss4d x_est(Eigen::Vector4d(-0.55929, 0.0694888, -0.583476, -0.26382), Eigen::Matrix4d::Identity());

  // clang-format off
  Eigen::Array2Xd meas = {{-1.5, -1.2, -0.8, -1.7, -2.0, -1.11}, 
                          { 2.5,  2.7,  2.3,  1.9,  3.0,  2.04}};
  // clang-format on

  ConstantVelocity dyn_model{ 0.5 };
  IdentitySensorModel sen_model{ 1.0 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  // old state from 3_1, 3_2
  gp << "set object " << ++object_counter << " circle center " << 0.5 << "," << 0.5 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << 0.5 << "," << 0.5 << " to " << -0.00173734 << "," << 0.0766262
     << " nohead lc rgb 'orange-red'\n";
  gp << "set object " << ++object_counter << " circle center " << -0.00173734 << "," << 0.0766262 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << -0.00173734 << "," << 0.0766262 << " to " << -0.55929 << "," << 0.0694888
     << " nohead lc rgb 'orange-red'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF, predict_next_state_3_4)
{
  PDAF::Config config = {.pdaf = {
                             .mahalanobis_threshold = 4.0,
                             .prob_of_detection     = 0.9,
                             .clutter_intensity     = 1.0,
                         }};

  vortex::prob::Gauss4d x_est(Eigen::Vector4d(-1.20613, 0.610616, -0.618037, 0.175242), Eigen::Matrix4d::Identity());
  // clang-format off
  Eigen::Array2Xd meas = {{-2.0, -1.8, -2.3, 0.6, -2.0, -1.4}, 
                          { 2.2,  2.3,  2.0, 1.5,  2.0,  2.5}};
  // clang-format on

  ConstantVelocity dyn_model{ 0.5 };
  IdentitySensorModel sen_model{ 1.0 };

  auto [x_final, x_pred, z_pred, x_updated, gated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

#if (GNUPLOT_ENABLE)
  Gnuplot gp;
  gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
  gp << "set size ratio -1\n";

  gp << "set style circle radius 0.05\n";
  gp << "plot '-' with circles title 'Samples' linecolor rgb 'red' fs transparent solid 1 noborder\n";
  gp.send1d(meas);

  int object_counter = 0;

  for (const auto& state : x_updated)
  {
    vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2, 2));
    vortex::utils::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss);

    gp << "set object " << ++object_counter << " ellipse center " << ellipse.x() << "," << ellipse.y() << " size "
       << ellipse.major_axis() << "," << ellipse.minor_axis() << " angle " << ellipse.angle_deg()
       << " back fc rgb 'skyblue' fs transparent solid 0.4 noborder\n";

    gp << "set object " << ++object_counter << " circle center " << ellipse.x() << "," << ellipse.y() << " size "
       << 0.02 << " fs empty border lc rgb 'blue'\n";
  }

  gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size "
     << 0.05 << " fs empty border lc rgb 'black'\n";
  gp << "set object " << ++object_counter << " circle center " << x_pred.mean()(0) << "," << x_pred.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'pink'\n";
  gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1)
     << " size " << 0.05 << " fs empty border lc rgb 'green'\n";

  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_pred.mean()(0) << ","
     << x_pred.mean()(1) << " nohead lc rgb 'pink'\n";
  gp << "set arrow from " << x_est.mean()(0) << "," << x_est.mean()(1) << " to " << x_final.mean()(0) << ","
     << x_final.mean()(1) << " nohead lc rgb 'green'\n";

  // old state from 3_1, 3_2, 3_3
  gp << "set object " << ++object_counter << " circle center " << 0.5 << "," << 0.5 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << 0.5 << "," << 0.5 << " to " << -0.00173734 << "," << 0.0766262
     << " nohead lc rgb 'orange-red'\n";
  gp << "set object " << ++object_counter << " circle center " << -0.00173734 << "," << 0.0766262 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << -0.00173734 << "," << 0.0766262 << " to " << -0.55929 << "," << 0.0694888
     << " nohead lc rgb 'orange-red'\n";
  gp << "set object " << ++object_counter << " circle center " << -0.55929 << "," << 0.0694888 << " size " << 0.05
     << " fs empty border lc rgb 'orange-red'\n";
  gp << "set arrow from " << -0.55929 << "," << 0.0694888 << " to " << -1.20613 << "," << 0.610616
     << " nohead lc rgb 'orange-red'\n";

  vortex::utils::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, config.mahalanobis_threshold);
  gp << "set object " << ++object_counter << " ellipse center " << gate.x() << "," << gate.y() << " size "
     << gate.major_axis() << "," << gate.minor_axis() << " angle " << gate.angle_deg()
     << " fs empty border lc rgb 'cyan'\n";

  gp << "replot\n";
#endif
}

TEST(PDAF_2, test_with_other_model)
{
  using TestDynamicModel = vortex::models::ConstantAcceleration;
  using TestSensorModel = vortex::models::IdentitySensorModel<6, 2>;
  using PDAF = vortex::filter::PDAF<TestDynamicModel, TestSensorModel>;

  PDAF::Config config;
  config.mahalanobis_threshold = 1.0;
  config.prob_of_detection = 0.8;
  config.clutter_intensity = 1.0;
  vortex::prob::Gauss6d x_est({ 0.0, 1.0, 3.0, -1.0, 0.0, 0.5 }, Eigen::MatrixXd::Identity(6, 6));
  std::vector<Eigen::Vector2d> meas = { { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 },
                                        { 0.0, 2.0 }, { 2.0, 0.0 }, { 2.0, 2.0 } };

  TestDynamicModel dyn_model{ 1.0, 1.0 };
  TestSensorModel sen_model{ 1.0 };

  auto [x_final, inside, outside, x_pred, z_pred, x_updated] =
      PDAF::step(dyn_model, sen_model, 1.0, x_est, meas, config);
  std::cout << "x_final: " << x_final.mean() << std::endl;

  // it compiles -> other template parameters are working with the PDAF class
  ASSERT_TRUE(true);
}