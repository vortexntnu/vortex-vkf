#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/movement_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

#include "test_models.hpp"
template <int N_DIM_x, int N_DIM_z>
struct SimData 
{
    // SimData(std::vector<double> time, std::vector<Eigen::Vector<double, N_DIM_x>> x_true, std::vector<vortex::prob::MultiVarGauss<N_DIM_x>> x_est, std::vector<Eigen::Vector<double, N_DIM_z>> z_meas, std::vector<vortex::prob::MultiVarGauss<N_DIM_z>> z_est)
    //     : time(time), x_true(x_true), x_est(x_est), z_meas(z_meas), z_est(z_est) {}
    std::vector<double> time;
    std::vector<Eigen::Vector<double, N_DIM_x>> x_true;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_x>> x_est;
    std::vector<Eigen::Vector<double, N_DIM_z>> z_meas;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_z>> z_est;
}; 

template <int N_DIM_x, int N_DIM_z>
SimData<N_DIM_x, N_DIM_z> simulate(std::shared_ptr<vortex::filters::KalmanFilterBase> kf, std::shared_ptr<vortex::models::DynamicModelBaseI> dyn_mod, std::shared_ptr<vortex::models::SensorModelBaseI> sens_mod, size_t num_iters, double dt, const Eigen::Vector<double, N_DIM_x>& x0, const Eigen::Matrix<double, N_DIM_x, N_DIM_x>& P0)
{
    using Vec_x = Eigen::Vector<double, N_DIM_x>;
    using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
    using Gauss_x = vortex::prob::MultiVarGauss<N_DIM_x>;

    using Vec_u = Eigen::Vector<double, N_DIM_u>;

    using Vec_z = Eigen::Vector<double, N_DIM_z>;
    using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
    using Gauss_z = vortex::prob::MultiVarGauss<N_DIM_z>;
    // Random number generator
    std::random_device rd;                            
    std::mt19937 gen(rd());                             

    // Initial state
    vortex::prob::MultiVarGauss<N_DIM_x> x(x0, P0);

    std::vector<double> time;
    std::vector<Eigen::Vector<double, N_DIM_x>> x_true;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_x>> x_est;
    std::vector<Eigen::Vector<double, N_DIM_z>> z_true;
    std::vector<vortex::prob::MultiVarGauss<N_DIM_z>> z_est;

    for (int i = 0; i < 100; ++i) {
        // Simulate
        Eigen::Vector<double, N_DIM_u> u = Eigen::Vector<double, N_DIM_u>::Zero();
        Eigen::Vector<double, N_DIM_v> v = Eigen::Vector<double, N_DIM_v>::Zero();
        Eigen::Vector<double, N_DIM_w> w = Eigen::Vector<double, N_DIM_w>::Zero();
        x = dyn_mod->sample_f_d(x, u, dt, gen);
        Eigen::Vector<double, N_DIM_z> z = sens_mod->sample_h(x, gen);

        // Predict
        auto pred = kf->predict(dyn_mod, sens_mod, x, u, dt);
        vortex::prob::MultiVarGauss<N_DIM_x> x_est_pred = pred.first;
        vortex::prob::MultiVarGauss<N_DIM_z> z_est_pred = pred.second;

        // Update
        x = kf->update(dyn_mod, sens_mod, x_est_pred, z_est_pred, z);

        // Save data
        time.push_back(i*dt);
        x_true.push_back(x.mean());
        x_est.push_back(x_est_pred);
        z_true.push_back(z);
        z_est.push_back(z_est_pred);
    }

    return {time, x_true, x_est, z_true, z_est};
}

class KFTest : public ::testing::Test {
protected:
};