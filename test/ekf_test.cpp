#include <gtest/gtest.h>
#include <iostream>

#include <filters/ekf.hpp>
#include <probability/multi_var_gauss.hpp>
#include "test_models.hpp"

using namespace vortex::filters;
using namespace vortex::models;
using namespace vortex::prob;


using State = typename SimpleDynamicModel::State;
using Mat_xx = typename SimpleDynamicModel::Mat_xx;
using Measurement = typename SimpleSensorModel::Measurement;
const int N_DIMS_x = SimpleDynamicModel::N_DIM_x;
const int N_DIMS_z = SimpleSensorModel::N_DIM_z;

TEST(EKF, Simple) {

    SimpleDynamicModel dynamic_model;
    SimpleSensorModel sensor_model;
    EKF<SimpleDynamicModel, SimpleSensorModel> ekf(dynamic_model, sensor_model);

    // Initial state
    MultiVarGauss<N_DIMS_x> x({0, 0}, Mat_xx::Identity());

    // Predict
    auto [x_est_pred, z_est_pred] = ekf.predict(x, 0.1);


    // Update
    Measurement z = {1, 1};
    ekf.update(x_est_pred, z_est_pred, z);

    // Check that the state is close to zero
    // ASSERT_TRUE(x.isMuchSmallerThan(State::Ones()));
}
