#include "gnuplot-iostream.h"
#include "gtest_assertions.hpp"

#include <gtest/gtest.h>
#include <random>
#include <vortex_filtering/filters/iekf.hpp>
#include <vortex_filtering/models/interfaces/dynamic_model_ltv.hpp>
#include <vortex_filtering/models/interfaces/sensor_model_ltv.hpp>
#include <vortex_filtering/probability/lie_group_gauss.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

using namespace vortex::prob;

class LieGroupGaussTest : public ::testing::Test {
protected:
  using LieGroup = manif::SE3d;
  using Tangent  = LieGroup::Tangent;
  using Vec_n    = Eigen::Vector<double, Tangent::DoF>;
  using Mat_nn   = Eigen::Matrix<double, Tangent::DoF, Tangent::DoF>;

  LieGroup mean;
  Mat_nn cov;
  LieGroupGauss<LieGroup> gauss;

  void SetUp() override
  {
    mean  = LieGroup::Identity();
    cov   = Mat_nn::Identity();
    gauss = LieGroupGauss<LieGroup>(mean, cov);
  }
};

TEST_F(LieGroupGaussTest, DefaultConstructor)
{
  LieGroupGauss<LieGroup> defaultGauss;
  EXPECT_TRUE(isApproxEqual(defaultGauss.cov(), Mat_nn::Identity(), 1e-6));
}

TEST_F(LieGroupGaussTest, ParameterizedConstructor)
{
  EXPECT_EQ(gauss.mean(), mean);
  EXPECT_EQ(gauss.cov(), cov);
}

TEST_F(LieGroupGaussTest, PdfCalculation)
{
  double pdf_value = gauss.pdf(mean);
  EXPECT_NEAR(pdf_value, 1.0 / std::sqrt(std::pow(2 * std::numbers::pi, Tangent::DoF) * cov.determinant()), 1e-6);
}

TEST_F(LieGroupGaussTest, LogPdfCalculation)
{
  double logpdf_value = gauss.logpdf(mean);
  EXPECT_NEAR(logpdf_value, -0.5 * std::log(std::pow(2 * std::numbers::pi, Tangent::DoF) * cov.determinant()), 1e-6);
}

TEST_F(LieGroupGaussTest, MahalanobisDistance)
{
  double distance = gauss.mahalanobis_distance(mean);
  EXPECT_NEAR(distance, 0.0, 1e-6);
}

TEST_F(LieGroupGaussTest, Sample)
{
  std::mt19937 gen(42); // Fixed seed for reproducibility
  LieGroup sample = gauss.sample(gen);
  EXPECT_TRUE(sample.coeffs().size() > 0); // Check if sample is non-empty
}

TEST_F(LieGroupGaussTest, StandardGaussian)
{
  auto standardGauss = LieGroupGauss<LieGroup>::Standard();
  EXPECT_EQ(standardGauss.mean(), LieGroup::Identity());
  EXPECT_EQ(standardGauss.cov(), Mat_nn::Identity());
}

TEST_F(LieGroupGaussTest, EqualityOperator)
{
  LieGroupGauss<LieGroup> gauss2(mean, cov);
  EXPECT_TRUE(gauss == gauss2);
}

TEST_F(LieGroupGaussTest, StreamOperator)
{
  std::ostringstream os;
  os << gauss;
  std::string output = os.str();
  EXPECT_TRUE(output.find("Mean:") != std::string::npos);
  EXPECT_TRUE(output.find("Covariance:") != std::string::npos);
}

// Define a simple Lie group for testing purposes
using TestLieGroup = manif::SE3d;

// Define a simple implementation of LieGroupDynamicModel for testing
class TestLieGroupDynamicModel
    : public vortex::model::interface::DynamicModel<TestLieGroup, TestLieGroup, TestLieGroup> {
public:
  Mx f(double dt, const Mx &x, const Mu &u, const Mv &v) const override
  {
    // Simple dynamics: x' = x * exp(dt * (u + v))
    return x + (dt * (u.log() + v.log()));
  }

  T::Mat_vv Q(double dt, const Mx & /*x*/) const override
  {
    // Simple constant covariance
    return T::Mat_vv::Identity() * dt;
  }
};

// Define a simple implementation of LieGroupDynamicModelLTV for testing
class TestLieGroupDynamicModelLTV
    : public vortex::model::interface::DynamicModelLTV<TestLieGroup, TestLieGroup, TestLieGroup> {
public:
  Mx f(double dt, const Mx &x, const Mu &u, const Mv &v) const override
  {
    // Simple dynamics: x' = x * exp(dt * (u + v))
    return x + (dt * (u.log() + v.log()));
  }

  T::Mat_xx J_f_x(double /*dt*/, const Mx & /*x*/) const override
  {
    // Simple Jacobian: identity matrix
    return T::Mat_xx::Identity();
  }

  T::Mat_vv Q(double dt, const Mx & /*x*/) const override
  {
    // Simple constant covariance
    return T::Mat_vv::Identity() * dt;
  }
};

// Test cases
TEST(LieGroupDynamicModelTest, TestFD)
{
  TestLieGroupDynamicModel model;
  TestLieGroup x = TestLieGroup::Identity();
  TestLieGroup u = TestLieGroup::Identity();
  TestLieGroup v = TestLieGroup::Identity();

  auto result = model.f(1.0, x, u, v);
  EXPECT_EQ(result, x + (u.log() + v.log()));
}

TEST(LieGroupDynamicModelTest, TestSampleFD)
{
  TestLieGroupDynamicModel model;
  TestLieGroup x = TestLieGroup::Identity();
  TestLieGroup u = TestLieGroup::Identity();
  std::mt19937 gen(42);

  auto result = model.sample_f(1.0, x, u, gen);
  EXPECT_NE(result, x); // Should not be equal due to noise
}

TEST(LieGroupDynamicModelLTVTest, TestFD)
{
  TestLieGroupDynamicModelLTV model;
  TestLieGroup x = TestLieGroup::Identity();
  TestLieGroup u = TestLieGroup::Identity();
  TestLieGroup v = TestLieGroup::Identity();

  auto result = model.f(1.0, x, u, v);
  EXPECT_EQ(result, x + (u.log() + v.log()));
}

TEST(LieGroupDynamicModelLTVTest, TestPredFromEst)
{
  constexpr int N_DIM_x = TestLieGroup::Tangent::DoF;
  using Mat_xx          = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
  TestLieGroupDynamicModelLTV model;
  TestLieGroup x = TestLieGroup::Identity();
  TestLieGroup u = TestLieGroup::Identity();
  vortex::prob::LieGroupGauss<TestLieGroup> x_est(TestLieGroup::Identity(), Mat_xx::Identity());

  auto result = model.pred_from_est(1.0, x_est, u);
  EXPECT_EQ(result.mean(), x + (u.log()));
}

using R2 = manif::Rn<double, 2>;

class CircleMovementModel : public vortex::model::interface::DynamicModelLTV<manif::SE2d, R2, manif::SE2d> {
public:
  using Mx = manif::SE2d;
  using Mu = R2;
  using Mv = manif::SE2d;
  using T  = vortex::Types_xuv<Mx::DoF, Mu::DoF, Mv::DoF>;

  // Constructor that initializes the process noise covariance matrix Q_d
  CircleMovementModel(double process_noise_std_x, double process_noise_std_y, double process_noise_std_theta)
  {
    // Initialize Q_d_ as a diagonal matrix with variance based on the given standard deviations
    Q_d_       = T::Mat_vv::Zero();
    Q_d_(0, 0) = std::pow(process_noise_std_x, 2);     // Variance in x
    Q_d_(1, 1) = std::pow(process_noise_std_y, 2);     // Variance in y
    Q_d_(2, 2) = std::pow(process_noise_std_theta, 2); // Variance in theta
  }

  /** Discrete-time dynamics modeling circular motion around the origin.
   * @param dt Time step
   * @param x Current state in SE(2)
   * @param u Input vector (not used here but defined in base class)
   * @param v Process noise in R_z(2) tangent space
   * @return Updated state in SE(2)
   */
  Mx f(double dt, const Mx &x, const Mu & /*u*/ = Mu::Identity(), const Mv &v = Mv::Identity()) const override
  {
    double theta       = x.angle();
    double r           = 1.0; // Radius of the circle
    double angular_vel = 0.5; // Constant angular velocity

    // Calculate the change in x and y due to circular motion
    double x_dot = -r * angular_vel * std::sin(theta);
    double y_dot = r * angular_vel * std::cos(theta);

    // Update x, y, and theta (angle) in SE2 with noise applied in the tangent space
    return manif::SE2d(
        x.x() + (x_dot + v.x()) * dt, x.y() + (y_dot + v.y()) * dt, theta + angular_vel * dt + v.angle() * dt);
  }

  /** Process noise covariance matrix Q_d */
  T::Mat_vv Q(double /*dt*/, const Mx & /*x*/) const override { return Q_d_; }

  /** Jacobian of the discrete-time dynamics with respect to the state x.
   * @param dt Time step
   * @param x Current state in SE(2)
   * @return Jacobian matrix with respect to x
   */
  T::Mat_xx J_f_x(double dt, const Mx &x) const override
  {
    double theta       = x.angle();
    double r           = 1.0;
    double angular_vel = 0.5;

    // Derivatives of x_{k+1} and y_{k+1} with respect to theta
    double d_x_next_d_theta = -r * angular_vel * std::cos(theta) * dt;
    double d_y_next_d_theta = -r * angular_vel * std::sin(theta) * dt;

    // Constructing the Jacobian matrix
    T::Mat_xx J_f_x = T::Mat_xx{
        {1.0, 0.0, d_x_next_d_theta}, // Row for ∂x_{k+1} with respect to x, y, and theta
        {0.0, 1.0, d_y_next_d_theta}, // Row for ∂y_{k+1} with respect to x, y, and theta
        {0.0, 0.0, 1.0}               // Row for ∂theta_{k+1} with respect to x, y, and theta
    };
    return J_f_x;
  }

  T::Mat_xu J_f_u(double /*dt*/, const Mx & /*x*/) const override { return T::Mat_xu::Zero(); }

  T::Mat_xv J_f_v(double dt, const Mx & /*x*/) const override
  {
    T::Vec_v v      = {dt, dt, dt};
    T::Mat_xv J_f_v = v.asDiagonal();
    return J_f_v;
  }

private:
  T::Mat_vv Q_d_; // Process noise covariance matrix
};

class LieGroupCircleModelTest : public ::testing::Test {
protected:
  CircleMovementModel model;

  LieGroupCircleModelTest()
      : model(0.1, 0.1, 0.1)
  {
  }
};

// Test the Q_d function to ensure the process noise covariance matrix is correctly initialized
TEST_F(LieGroupCircleModelTest, ProcessNoiseCovarianceMatrix)
{
  CircleMovementModel::T::Mat_vv Q_d = model.Q(0.1, manif::SE2d::Identity());

  // Check diagonal elements for the expected variances
  EXPECT_NEAR(Q_d(0, 0), 0.01, 1e-6); // Variance in x (0.1^2)
  EXPECT_NEAR(Q_d(1, 1), 0.01, 1e-6); // Variance in y (0.1^2)
  EXPECT_NEAR(Q_d(2, 2), 0.01, 1e-6); // Variance in theta (0.1^2)

  // Check that non-diagonal elements are zero
  EXPECT_NEAR(Q_d(0, 1), 0.0, 1e-6);
  EXPECT_NEAR(Q_d(0, 2), 0.0, 1e-6);
  EXPECT_NEAR(Q_d(1, 0), 0.0, 1e-6);
  EXPECT_NEAR(Q_d(1, 2), 0.0, 1e-6);
  EXPECT_NEAR(Q_d(2, 0), 0.0, 1e-6);
  EXPECT_NEAR(Q_d(2, 1), 0.0, 1e-6);
}

// Test the f_d function to ensure the state update follows a circular path
TEST_F(LieGroupCircleModelTest, StateTransitionFunction)
{
  double dt = 0.1;
  CircleMovementModel::Mx x(1.0, 0.0, 0.0); // Start at (1, 0) with theta = 0
  CircleMovementModel::Mv v(0.0, 0.0, 0.0); // No process noise

  manif::SE2d x_next = model.f(dt, x, CircleMovementModel::Mu::Identity(), v);

  // Expected next position, moving along a circle of radius 1 with angular velocity 0.5 rad/s
  double expected_x     = 1.0 + (-1.0 * 0.5 * std::sin(0.0)) * dt;
  double expected_y     = (0.0 + 1.0 * 0.5 * std::cos(0.0)) * dt;
  double expected_theta = 0.0 + 0.5 * dt;

  EXPECT_NEAR(x_next.x(), expected_x, 1e-6);
  EXPECT_NEAR(x_next.y(), expected_y, 1e-6);
  EXPECT_NEAR(x_next.angle(), expected_theta, 1e-6);
}

// Test the Jacobian J_f_x function to ensure it correctly models the derivative with respect to the state
TEST_F(LieGroupCircleModelTest, JacobianFunction)
{
  double dt = 0.1;
  manif::SE2d x(1.0, 0.0, 0.0); // Start at (1, 0) with theta = 0
  CircleMovementModel::T::Mat_xx J_f_x = model.J_f_x(dt, x);

  // Expected partial derivatives
  double theta            = 0.0;
  double r                = 1.0;
  double angular_vel      = 0.5;
  double d_x_next_d_theta = -r * angular_vel * std::cos(theta) * dt;
  double d_y_next_d_theta = -r * angular_vel * std::sin(theta) * dt;

  // Check the Jacobian values
  EXPECT_NEAR(J_f_x(0, 0), 1.0, 1e-6);
  EXPECT_NEAR(J_f_x(1, 1), 1.0, 1e-6);
  EXPECT_NEAR(J_f_x(2, 2), 1.0, 1e-6);
  EXPECT_NEAR(J_f_x(0, 2), d_x_next_d_theta, 1e-6);
  EXPECT_NEAR(J_f_x(1, 2), d_y_next_d_theta, 1e-6);
}

// Generate a trajectory based on the model
std::vector<std::tuple<double, double, double>> generateTrajectory(const CircleMovementModel &model, double dt,
                                                                   int num_steps, const manif::SE2d &initial_state,
                                                                   const CircleMovementModel::Mu &input)
{
  std::vector<std::tuple<double, double, double>> trajectory;
  std::mt19937 gen(42); // Random number generator with a fixed seed for reproducibility

  // Start with the initial state
  manif::SE2d current_state = initial_state;

  // Generate the trajectory
  for (int i = 0; i < num_steps; ++i) {
    // Record the current state (x, y, theta)
    trajectory.emplace_back(current_state.x(), current_state.y(), current_state.angle());

    // Sample the next state with noise
    current_state = model.sample_f(dt, current_state, input, gen);
  }

  return trajectory;
}

// Plot the trajectory using gnuplot-iostream
void plotTrajectory(const std::vector<std::tuple<double, double, double>> &trajectory)
{
  Gnuplot gp;
  gp << "set title 'Circular Motion Trajectory'\n";
  gp << "set xlabel 'x'\n";
  gp << "set ylabel 'y'\n";
  gp << "set grid\n";
  gp << "set size ratio -1\n"; // Keep aspect ratio for circular trajectory

  // Convert trajectory data to a format gnuplot understands (only x, y)
  std::vector<std::pair<double, double>> xy_points;
  for (const auto &[x, y, theta] : trajectory) {
    xy_points.emplace_back(x, y);
  }

  // Plot the trajectory
  gp << "plot '-' with linespoints title 'Trajectory'\n";
  gp.send1d(xy_points);
}

// Test case for generating and plotting the trajectory
TEST(CircleMovementModelTest, GenerateAndPlotTrajectory)
{
  // Initialize the model with process noise standard deviations
  CircleMovementModel model(0.1, 0.1, 0.05);

  // Define parameters for the trajectory
  double dt     = 0.1;                      // Time step
  int num_steps = 200;                      // Number of steps
  manif::SE2d initial_state(1.0, 0.0, 0.0); // Initial state at (1, 0) with theta = 0
  CircleMovementModel::Mu input;
  input.coeffs() << 0.0, 0.0; // No input

  // Generate the trajectory
  auto trajectory = generateTrajectory(model, dt, num_steps, initial_state, input);

  // Optional: Assert on the trajectory properties if needed
  ASSERT_EQ(trajectory.size(), num_steps);

  // Plot the trajectory (only for visualization, not part of the test validation)
  plotTrajectory(trajectory);
}

class LieEKFTest : public ::testing::Test {
public:
  // Define the state and input dimensions
  using Mx = CircleMovementModel::Mx;
  using Mv = CircleMovementModel::Mv;
  using Mu = CircleMovementModel::Mu;
  using Mz = manif::Rn<double, 2>;
  using Mw = manif::Rn<double, 2>;

  using Gauss_x = vortex::prob::LieGroupGauss<Mx>;

  // Define the process noise covariance matrix
  using T     = vortex::Types_xzuvw<Mx::DoF, Mz::DoF, Mu::DoF, Mv::DoF, Mw::DoF>;
  using LIEKF = vortex::filter::LIEKF<Mx, Mz, Mu, Mv, Mw>;

  class MockSensorModel : public vortex::model::interface::SensorModelLTV<Mx, Mz, Mw> {
  public:
    MockSensorModel(double std)
        : std(std)
    {
    }
    T::Mat_zx J_g_x(const Mx & /*x*/) const override { return T::Mat_zx::Identity(); }

    T::Mat_ww R_z(const Mx & /*x*/) const override { return T::Mat_ww::Identity(); }

    double std;
  };

protected:
  LieEKFTest()
      : dynamic_model(0.1, 0.1, 0.1)
      , sensor_model(0.1)
  {
  }

  CircleMovementModel dynamic_model;
  MockSensorModel sensor_model;
};

TEST_F(LieEKFTest, TestPredict)
{
  // Define the initial state and input
  Mx x_est_prev(1.0, 0.0, 0.0);
  T::Mat_xx P = T::Mat_xx::Identity();
  Gauss_x x_est(x_est_prev, P);
  Mu u = Mu::Identity();

  // Perform the prediction step
  auto [x_pred, z_pred] = LIEKF::predict(dynamic_model, sensor_model, 0.1, x_est, u);

  // Expected next state, moving along a circle of radius 1 with angular velocity 0.5 rad/s
  double expected_x     = 1.0 + (-1.0 * 0.5 * std::sin(0.0)) * 0.1;
  double expected_y     = (0.0 + 1.0 * 0.5 * std::cos(0.0)) * 0.1;
  double expected_theta = 0.0 + 0.5 * 0.1;

  // Check the predicted state
  EXPECT_NEAR(x_pred.x(), expected_x, 1e-6);
  EXPECT_NEAR(x_pred.y(), expected_y, 1e-6);
  EXPECT_NEAR(x_pred.angle(), expected_theta, 1e-6);
}

TEST_F(LieEKFTest, TestConvergenceOverTrajectory)
{
  // Define the initial state and input
  Mx x0(1.0, 0.0, 0.0);
  Mu u = Mu::Identity();

  // Define the trajectory parameters
  double dt     = 0.1;
  int num_steps = 200;

  // Initialize the Lie EKF filter
  LIEKF::Gauss_x x_est(x0, T::Mat_xx::Identity());
  std::mt19937 gen(42); // Random number generator with a fixed seed for reproducibility

  auto state_trajectory = generateTrajectory(dynamic_model, dt, num_steps, x0, u);

  std::vector<std::tuple<double, double, double>> state_estimates;
  // Generate the trajectory and perform the prediction step at each time step
  for (int i = 0; i < num_steps; ++i) {
    // Generate the measurement
    Mx x = Mx(std::get<0>(state_trajectory[i]), std::get<1>(state_trajectory[i]), std::get<2>(state_trajectory[i]));
    Mz z = sensor_model->sample_g(x, gen);
    // Perform the prediction step
    auto [x_upd, x_pred, z_pred] = LIEKF::step(dynamic_model, sensor_model, 0.1, x_est, z);
    state_estimates.emplace_back(x_upd.x(), x_upd.y(), x_upd.angle());
  }

  // Check the final state estimate
  auto final_state_estimate = state_estimates.back();
  auto final_state_truth    = state_trajectory.back();

  EXPECT_NEAR(std::get<0>(final_state_estimate), std::get<0>(final_state_truth), 0.1);

  // Optional: Plot the state trajectory and the state estimates for visualization
  plotTrajectory(state_trajectory);
  plotTrajectory(state_estimates);
}