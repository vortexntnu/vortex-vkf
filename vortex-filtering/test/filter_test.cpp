#include <gtest/gtest.h>
#include <gnuplot-iostream.h>
#include <Eigen/Dense>

#include <sstream>
#include <memory>
#include <random>
#include <algorithm>
#include <vector>
#include <string>

#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/plotting/utils.hpp>

#include "test_models.hpp"
#include "gtest_assertions.hpp"

struct SimData 
{
    std::vector<double> time;
    std::vector<Eigen::VectorXd> x_true;
    std::vector<vortex::prob::MultiVarGaussXd> x_est;
    std::vector<Eigen::VectorXd> z_meas;
    std::vector<vortex::prob::MultiVarGaussXd> z_est;
}; 

/**
 * @param test_name name of the test case
 * @param kf pointer to the kalman filter
 * @param dyn_mod_real pointer to the real dynamic model
 * @param dyn_mod_filter pointer to the dynamic model used by the filter
 * @param sens_mod_real pointer to the real sensor model
 * @param sens_mod_filter pointer to the sensor model used by the filter
 * @param num_iters number of iterations to simulate
 * @param dt time step
 * @param x0 initial state
 * @param x0_est initial state estimate
 * @param tolerance tolerance for the test
 */
struct TestParamKF
{
    std::string test_name = "";
    std::shared_ptr<vortex::filter::interface::KalmanFilterX> kf = nullptr;
    std::shared_ptr<vortex::models::interface::DynamicModelX> dyn_mod_real = nullptr;
    std::shared_ptr<vortex::models::interface::DynamicModelX> dyn_mod_filter = nullptr;
    std::shared_ptr<vortex::models::interface::SensorModelX> sens_mod_real = nullptr;
    std::shared_ptr<vortex::models::interface::SensorModelX> sens_mod_filter = nullptr;
    size_t num_iters = 0;
    double dt = 0.0;
    typename vortex::models::interface::DynamicModelX::VecX x0 = {};
    typename vortex::models::interface::DynamicModelX::GaussX x0_est = {Eigen::VectorXd::Zero(1), Eigen::MatrixXd::Zero(1,1)};
    double tolerance = 0.0;
};

class KFTest : public ::testing::TestWithParam<TestParamKF> {
protected:

    using VecX   = typename vortex::models::interface::DynamicModelX::VecX;
    using MatXX  = typename vortex::models::interface::DynamicModelX::MatXX;
    using GaussX = typename vortex::models::interface::DynamicModelX::GaussX;



    /**
     * @brief Simulate a dynamic model with a sensor model and a kalman filter
     * 
     * @tparam DynamicModelT 
     * @tparam SensorModelT 
     * @param tp TestParamKF Params for the test
     * @return SimData<DynamicModelT::N_DIM_x, SensorModelT::N_DIM_z> 
     */
    SimData simulate(TestParamKF tp)
    {
        // Random number generator
        std::random_device rd;                            
        std::mt19937 gen(rd());                             

        // Initial state
        std::vector<double> time;
        std::vector<VecX> x_true;
        std::vector<GaussX> x_est;
        std::vector<VecX> z_meas;
        std::vector<GaussX> z_est;

        // const int N_DIM_x = tp.dyn_mod_real->get_dim_x();
        const int N_DIM_z = tp.sens_mod_real->get_dim_z();
        const int N_DIM_u = tp.dyn_mod_real->get_dim_u();
        // const int N_DIM_w = tp.dyn_mod_real->get_dim_v();

        // Simulate
        time.push_back(0.0);
        x_true.push_back(tp.x0);
        x_est.push_back(tp.x0_est);
        z_meas.push_back(tp.sens_mod_real->sample_hX(tp.x0, gen));
        z_est.push_back({VecX::Zero(N_DIM_z), MatXX::Identity(N_DIM_z, N_DIM_z)});


        for (size_t i = 0; i < tp.num_iters; ++i) {
            // Simulate
            VecX u = VecX::Zero(N_DIM_u);
            VecX x_true_ip1 = tp.dyn_mod_real->sample_f_dX(tp.dt, x_true.at(i), u, gen);
            VecX z_meas_i = tp.sens_mod_real->sample_hX(x_true.at(i), gen);

            // Predict
            auto next_state = tp.kf->stepX(tp.dyn_mod_filter, tp.sens_mod_filter, tp.dt, x_est.at(i), z_meas_i, u);
            GaussX x_est_upd = std::get<0>(next_state);
            GaussX z_est_pred = std::get<2>(next_state);


            // Save data
            time.push_back(time.back() + tp.dt);
            x_true.push_back(x_true_ip1);
            x_est.push_back(x_est_upd);
            z_meas.push_back(z_meas_i);
            z_est.push_back(z_est_pred);
        }

        return {time, x_true, x_est, z_meas, z_est};
    }

};






TEST_P(KFTest, convergence)
{
    auto params = GetParam();
    SimData sim_data = simulate(params);
    // Check
    EXPECT_TRUE(isApproxEqual(sim_data.x_true.back(), sim_data.x_est.back().mean(), params.tolerance));

    // Plot results
    std::vector<double> x_true_0 = vortex::plotting::extract_state_series(sim_data.x_true, 0);
    std::vector<double> x_est_0 = vortex::plotting::extract_state_series(vortex::plotting::extract_mean_series(sim_data.x_est), 0);
    std::vector<double> z_meas_0 = vortex::plotting::extract_state_series(sim_data.z_meas, 0);
    Gnuplot gp;
    gp << "set terminal wxt size 1200,800\n";
    gp << "set title '" << params.test_name << ", x_0 vs. time'\n";
    gp << "set xlabel 't'\n";
    gp << "set ylabel 'x_0'\n";
    gp << "plot '-' with lines title 'True', '-' with lines title 'Estimate', '-' with points title 'Measurements' ps 1\n";
    gp.send1d(std::make_tuple(sim_data.time, x_true_0));
    gp.send1d(std::make_tuple(sim_data.time, x_est_0));
    gp.send1d(std::make_tuple(sim_data.time, z_meas_0));

    // Plot 1. state against 2. state if possible
    if (sim_data.x_true.at(0).size() >= 2) {
        Gnuplot gp;
        std::vector<double> x_true_1 = vortex::plotting::extract_state_series(sim_data.x_true, 1);
        std::vector<double> x_est_1 = vortex::plotting::extract_state_series(vortex::plotting::extract_mean_series(sim_data.x_est), 1);
        gp << "set terminal wxt size 1200,800\n";
        gp << "set title '" << params.test_name << ", x_0 vs. x_1'\n";
        gp << "set xlabel 'x_0'\n";
        gp << "set ylabel 'x_1'\n";
        gp << "plot '-' with lines title 'True', '-' with lines title 'Estimate'\n";
        gp.send1d(std::make_tuple(x_true_0, x_true_1));
        gp.send1d(std::make_tuple(x_est_0, x_est_1));
    }

} 

using DynModT = NonlinearModel1;
using SensModT = vortex::models::IdentitySensorModel<1,1>;
// clang-format off
TestParamKF test0 = {
    "UKFNonlin_1Init0",
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<SensModT>(1e-2),
    std::make_shared<SensModT>(1e-2),
    1000,
    0.1,
    DynModT::Vec_x::Zero(),
    DynModT::Gauss_x{DynModT::Vec_x::Zero(), DynModT::Mat_xx::Identity()*0.1},
    1e-1
};

TestParamKF test1 = {
    "UKFNonlin_1Init4",
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<SensModT>(1e-2),
    std::make_shared<SensModT>(1e-2),
    1000,
    0.1,
    DynModT::Vec_x::Ones()*4,
    DynModT::Gauss_x{DynModT::Vec_x::Ones()*4, DynModT::Mat_xx::Identity()},
    1e-2
};

TestParamKF test2 = {
    "UKFLorenzBase",
    std::make_shared<vortex::filter::UKF<3,3>>(),
    std::make_shared<LorenzAttractorCT>(1e-3),
    std::make_shared<LorenzAttractorCT>(1e-3),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1e-2),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1e-2),
    1000,
    0.01,
    Eigen::Vector3d::Zero(),
    vortex::prob::Gauss3d{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()*0.1},
    1e-1
};

TestParamKF test3 = {
    "UKFLorenzMoreProcessNoise",
    std::make_shared<vortex::filter::UKF<3,3>>(),
    std::make_shared<LorenzAttractorCT>(1),
    std::make_shared<LorenzAttractorCT>(1e-1),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1),
    1000,
    0.1,
    Eigen::Vector3d::Zero(),
    vortex::prob::Gauss3d{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()*0.1},
    1
};

TestParamKF test4 = {
    "EKFLorenzBase",
    std::make_shared<vortex::filter::EKF<3,3>>(),
    std::make_shared<LorenzAttractorCTLTV>(1e-3),
    std::make_shared<LorenzAttractorCTLTV>(1e-3),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1e-2),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1e-2),
    1000,
    0.01,
    Eigen::Vector3d::Zero(),
    vortex::prob::Gauss3d{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()*0.1},
    1
};

TestParamKF test5 = {
    "EKFLorenzMoreProcessNoise",
    std::make_shared<vortex::filter::EKF<3,3>>(),
    std::make_shared<LorenzAttractorCTLTV>(1),
    std::make_shared<LorenzAttractorCTLTV>(1e-1),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1),
    std::make_shared<vortex::models::IdentitySensorModel<3,3>>(1),
    1000,
    0.1,
    Eigen::Vector3d::Zero(),
    vortex::prob::Gauss3d{Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()*0.1},
    1
};


INSTANTIATE_TEST_SUITE_P(
    KalmanFilter, KFTest,
    testing::Values(
        test0,
        test1,
        test2,
        test3,
        test4,
        test5
    ),
    [](const testing::TestParamInfo<KFTest::ParamType>& info) {
        return info.param.test_name;
    }
);
// clang-format on
