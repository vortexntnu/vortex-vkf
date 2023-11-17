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
template <int N_DIM_x, int N_DIM_z>
struct SimData 
{
    std::vector<double> time;
    std::vector<Eigen::Vector<double, N_DIM_x>> x_true;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_x>> x_est;
    std::vector<Eigen::Vector<double, N_DIM_z>> z_meas;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_z>> z_est;
}; 

struct TestParams
{
    std::shared_ptr<vortex::filters::KalmanFilterX> kf;
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

template <typename DynModT, typename SensModT>
class KFTest : public ::testing::TestWithParam<TestParams> {
protected:

    static constexpr int N_DIM_x = DynModT::N_DIM_x;
    static constexpr int N_DIM_u = DynModT::N_DIM_u;
    static constexpr int N_DIM_z = SensModT::N_DIM_z;

    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Gauss_x = vortex::prob::MultiVarGauss<N_DIM_x>;

    using Vec_u = Eigen::Vector<double, N_DIM_u>;

    using Vec_z = Eigen::Vector<double, N_DIM_z>;
    using Gauss_z = vortex::prob::MultiVarGauss<N_DIM_z>;


    /**
     * @brief Simulate a dynamic model with a sensor model and a kalman filter
     * 
     * @tparam DynamicModelT 
     * @tparam SensorModelT 
     * @param tp TestParams Params for the test
     * @return SimData<DynamicModelT::N_DIM_x, SensorModelT::N_DIM_z> 
     */
    SimData<N_DIM_x, N_DIM_z> simulate(TestParams tp)
    {
        auto kf = std::static_pointer_cast<vortex::filters::KalmanFilterI<DynModT, SensModT>>(tp.kf);
        auto dyn_mod_real = std::static_pointer_cast<DynModT>(tp.dyn_mod_real);
        auto sens_mod_real = std::static_pointer_cast<SensModT>(tp.sens_mod_real);
        auto dyn_mod_filter = std::static_pointer_cast<DynModT>(tp.dyn_mod_filter);
        auto sens_mod_filter = std::static_pointer_cast<SensModT>(tp.sens_mod_filter);


        // Random number generator
        std::random_device rd;                            
        std::mt19937 gen(rd());                             

        // Initial state
        std::vector<double> time;
        std::vector<Vec_x> x_true;
        std::vector<Gauss_x> x_est;
        std::vector<Vec_z> z_meas;
        std::vector<Gauss_z> z_est;

        // Simulate
        time.push_back(0.0);
        x_true.push_back(tp.x0);
        x_est.push_back(tp.x0_est);
        z_meas.push_back(sens_mod_real->sample_h(tp.x0, gen));
        z_est.push_back(sens_mod_real->sample_h(tp.x0_est.mean(), gen));


        for (size_t i = 0; i < tp.num_iters; ++i) {
            // Simulate
            Vec_u u = Vec_u::Zero();
            Vec_x x = dyn_mod_real->sample_f_d(x_true.at(i), u, tp.dt, gen);
            Vec_z z = sens_mod_real->sample_h(x_true.at(i), gen);

            // Predict
            auto next_state = kf->step(dyn_mod_filter, sens_mod_filter, x_est.at(i), z_meas.at(i), u, tp.dt);
            Gauss_x x_est_upd = std::get<0>(next_state);
            Gauss_z z_est_pred = std::get<2>(next_state);


            // Save data
            time.push_back(time.back() + tp.dt);
            x_true.push_back(x);
            x_est.push_back(x_est_upd);
            z_meas.push_back(z);
            z_est.push_back(z_est_pred);
        }

        return {time, x_true, x_est, z_meas, z_est};
    }

};




using DynModT = NonlinearModel1;
using SensModT = vortex::models::IdentitySensorModel<1,1>;
using Nonlin1Test = KFTest<DynModT,SensModT>;
TEST_P(Nonlin1Test, ukf_convergence)
{
    constexpr int N_DIM_x = DynModT::N_DIM_x;
    constexpr int N_DIM_z = SensModT::N_DIM_z;
    auto params = GetParam();
    SimData<N_DIM_x, N_DIM_z> sim_data = simulate(params);
    // Check
    for (size_t i = 0; i<N_DIM_x; i++) {
        EXPECT_NEAR(sim_data.x_true.back()(i), sim_data.x_est.back().mean()(i), params.tolerance) << "i = " << i;
    }
   
} 


TestParams test1 = {
    std::make_shared<vortex::filters::UKF<DynModT, SensModT>>(),
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
    std::make_shared<vortex::filters::UKF<DynModT, SensModT>>(),
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
    Nonlin1TestSuite, Nonlin1Test,
    testing::Values(
        test1,
        test2
    )
);
// // clang-format on
