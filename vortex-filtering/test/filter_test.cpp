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

struct TestParamKF
{
    std::shared_ptr<vortex::filter::interface::KalmanFilterX> kf;
    std::shared_ptr<vortex::models::DynamicModelX> dyn_mod_real;
    std::shared_ptr<vortex::models::DynamicModelX> dyn_mod_filter;
    std::shared_ptr<vortex::models::SensorModelX> sens_mod_real;
    std::shared_ptr<vortex::models::SensorModelX> sens_mod_filter;
    size_t num_iters;
    double dt;
    typename vortex::models::DynamicModelX::VecX x0;
    typename vortex::models::DynamicModelX::GaussX x0_est;
    double tolerance;
};

class KFTest : public ::testing::TestWithParam<TestParamKF> {
protected:

    using VecX = typename vortex::models::DynamicModelX::VecX;
    using MatXX = typename vortex::models::DynamicModelX::MatXX;
    using GaussX = typename vortex::models::DynamicModelX::GaussX;



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
            VecX x_true_ip1 = tp.dyn_mod_real->sample_f_dX(x_true.at(i), u, tp.dt, gen);
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




using DynModT = NonlinearModel1;
using SensModT = vortex::models::IdentitySensorModel<1,1>;

TEST_P(KFTest, ukf_convergence)
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
    gp << "set title 'State'\n";
    gp << "set xlabel 'Time'\n";
    gp << "set ylabel 'State'\n";
    gp << "plot '-' with lines title 'True', '-' with lines title 'Estimate', '-' with points title 'Measurements' ps 2\n";
    gp.send1d(std::make_tuple(sim_data.time, x_true_0));
    gp.send1d(std::make_tuple(sim_data.time, x_est_0));
    gp.send1d(std::make_tuple(sim_data.time, z_meas_0));

} 


TestParamKF test1 = {
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(0.001),
    std::make_shared<DynModT>(0.001),
    std::make_shared<SensModT>(0.01),
    std::make_shared<SensModT>(0.01),
    1000,
    0.1,
    DynModT::Vec_x::Zero(),
    DynModT::Gauss_x{DynModT::Vec_x::Zero(), DynModT::Mat_xx::Identity()*0.1},
    1e-1
};

TestParamKF test2 = {
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<DynModT>(1e-3),
    std::make_shared<SensModT>(1e-2),
    std::make_shared<SensModT>(1e-2),
    1000,
    0.1,
    DynModT::Vec_x::Zero(),
    DynModT::Gauss_x{DynModT::Vec_x::Zero(), DynModT::Mat_xx::Identity()*0.1},
    1e-2
};

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    Nonlin1TestSuite, KFTest,
    testing::Values(
        test1,
        test2
    )
);
// clang-format on
