#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

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

struct TestParams
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

class KFTest : public ::testing::TestWithParam<TestParams> {
protected:

    using VecX = typename vortex::models::DynamicModelX::VecX;
    using MatXX = typename vortex::models::DynamicModelX::MatXX;
    using GaussX = typename vortex::models::DynamicModelX::GaussX;



    /**
     * @brief Simulate a dynamic model with a sensor model and a kalman filter
     * 
     * @tparam DynamicModelT 
     * @tparam SensorModelT 
     * @param tp TestParams Params for the test
     * @return SimData<DynamicModelT::N_DIM_x, SensorModelT::N_DIM_z> 
     */
    SimData simulate(TestParams tp)
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
} 


TestParams test1 = {
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(0.1),
    std::make_shared<DynModT>(0.1),
    std::make_shared<SensModT>(0.1),
    std::make_shared<SensModT>(0.1),
    100,
    0.1,
    DynModT::Vec_x::Zero(),
    DynModT::Gauss_x{DynModT::Vec_x::Zero(), DynModT::Mat_xx::Identity()*0.1},
    1e-2
};

TestParams test2 = {
    std::make_shared<vortex::filter::UKF_M<DynModT, SensModT>>(),
    std::make_shared<DynModT>(0.01),
    std::make_shared<DynModT>(0.01),
    std::make_shared<SensModT>(0.01),
    std::make_shared<SensModT>(0.01),
    100,
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
// // clang-format on
