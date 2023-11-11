#include <gtest/gtest.h>

#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/movement_models.hpp>
#include <vortex_filtering/models/sensor_models.hpp>

#include "test_models.hpp"

class KFTest : public ::testing::Test {
protected:
    using PosMeasModel = SimpleSensorModel<4,2>;
    using CVModel = vortex::models::CVModel;
    using Vec_x = typename CVModel::Vec_x;
    using Mat_xx = typename CVModel::Mat_xx;
    using Gauss_x = typename CVModel::Gauss_x;
    using Gauss_z = typename PosMeasModel::Gauss_z;
    using Vec_z = typename PosMeasModel::Vec_z;

    void SetUp() override {
        // Create dynamic model
        dynamic_model_ = std::make_shared<CVModel>(1.0);
        // Create sensor model
        sensor_model_ = std::make_shared<PosMeasModel>(1.0);
        // Create EKF
        ekf_ = std::make_shared<vortex::filters::EKF<CVModel, PosMeasModel>>(*dynamic_model_, *sensor_model_);
        // Create UKF
        ukf_ = std::make_shared<vortex::filters::UKF<CVModel, PosMeasModel>>(*dynamic_model_, *sensor_model_);
    }

    std::shared_ptr<CVModel> dynamic_model_;
    std::shared_ptr<PosMeasModel> sensor_model_;
    std::shared_ptr<vortex::filters::EKF<CVModel, PosMeasModel>> ekf_;
    std::shared_ptr<vortex::filters::UKF<CVModel, PosMeasModel>> ukf_;
};