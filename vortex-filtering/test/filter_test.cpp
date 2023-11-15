#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/movement_models.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

#include "test_models.hpp"
template <int N_DIM_x, int N_DIM_z>
struct SimData 
{
    std::vector<double> time;
    std::vector<Eigen::Vector<double, N_DIM_x>> x_true;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_x>> x_est;
    std::vector<Eigen::Vector<double, N_DIM_z>> z_meas;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_z>> z_est;
}; 

template <typename DynModT, typename SensModT>
struct TestParams
{
    std::shared_ptr<vortex::filters::KalmanFilterI<DynModT, SensModT>> kf;
    std::shared_ptr<DynModT> dyn_mod_real;
    std::shared_ptr<SensModT> sens_mod_real;
    std::shared_ptr<DynModT> dyn_mod_filter;
    std::shared_ptr<SensModT> sens_mod_filter;
    size_t num_iters;
    double dt;
    typename DynModT::Vec_x x0;
    typename DynModT::Gauss_x x0_est;
    double tolerance;
};

template <typename DynModT, typename SensModT>
class KFTest : public ::testing::TestWithParam<TestParams<DynModT, SensModT>> {
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
    SimData<N_DIM_x, N_DIM_z> simulate(TestParams<DynModT,SensModT> tp)
    {
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
        z_meas.push_back(tp.sens_mod_real->sample_h(tp.x0, gen));
        z_est.push_back(tp.sens_mod_real->sample_h(tp.x0_est.mean(), gen));


        for (size_t i = 0; i < tp.num_iters; ++i) {
            // Simulate
            Vec_u u = Vec_u::Zero();
            Vec_x x = tp.dyn_mod_real->sample_f_d(x_true.at(i), u, tp.dt, gen);
            Vec_z z = tp.sens_mod_real->sample_h(x_true.at(i), gen);

            // Predict
            auto next_state = tp.kf->step(tp.dyn_mod_filter, tp.sens_mod_filter, x_est.at(i), z_meas.at(i), u, tp.dt);
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

testing::AssertionResult isApproxEqual(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol)
{
    if (a.size() != b.size()) {
        return testing::AssertionFailure() << "Vectors are not the same size, "
                                           << "a: " << a.transpose()
                                           << ", b: " << b.transpose();
    }

    for (int i = 0; i < a.size(); ++i) {
        if (std::abs(a(i) - b(i)) > tol) {
            return testing::AssertionFailure() << "Vectors are not the same, "
                                               << "a: " << a.transpose()
                                               << ", b: " << b.transpose();
        }
    }

    return testing::AssertionSuccess();
}


template<typename DynModT>
using KFTestDynamicModels = KFTest<DynModT, SimpleSensorModel<DynModT::N_DIM_x, DynModT::N_DIM_x>>;

TYPED_TEST_SUITE_P(KFTestDynamicModels);

TYPED_TEST_P(KFTestDynamicModels, ukf_convergence)
{
    using DynModT = TypeParam;
    constexpr int N_DIM_x = DynModT::N_DIM_x;
    using SensModT = SimpleSensorModel<N_DIM_x, N_DIM_x>;
    constexpr int N_DIM_z = SensModT::N_DIM_z;

    std::vector<TestParams<DynModT, SensModT>> test_cases;
    test_cases.push_back({std::make_shared<vortex::filters::UKF<DynModT, SensModT>>(),
                          std::make_shared<DynModT>(0.1),
                          std::make_shared<SensModT>(0.01),
                          std::make_shared<DynModT>(0.1),
                          std::make_shared<SensModT>(0.01),
                          1000,
                          3,
                          DynModT::Vec_x::Ones(),
                          typename DynModT::Gauss_x(DynModT::Vec_x::Zero(), DynModT::Mat_xx::Identity()),
                          1e-3});

    // Simulate
    for (auto params : test_cases) {
        SimData<N_DIM_x, N_DIM_z> sim_data = this->simulate(params);

        // Check
        std::cout << "x_true: " << sim_data.x_true.back().transpose() << std::endl;
        std::cout << "x_est:  " << sim_data.x_est.back().mean().transpose() << std::endl;
        EXPECT_TRUE(isApproxEqual(sim_data.x_true.back(), sim_data.x_est.back().mean(), params.tolerance));
    }
   
} 


REGISTER_TYPED_TEST_SUITE_P(KFTestDynamicModels, 
                            ukf_convergence);


using DynamicModelsList = ::testing::Types<NonlinearModel1>;
INSTANTIATE_TYPED_TEST_SUITE_P(KFTestSuite, KFTestDynamicModels, DynamicModelsList);