#include <gtest/gtest.h>
#include <vortex_filtering/filters/pdaf.hpp>
#include <iostream>

using SimplePDAF = PDAF<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;

TEST(PDAF, init)
{
    SimplePDAF pdaf(0.0, 0.0, 0.0);
    EXPECT_EQ(pdaf.gate_threshold_, 0.0);
    EXPECT_EQ(pdaf.prob_of_detection_, 0.0);
    EXPECT_EQ(pdaf.clutter_intensity_, 0.0);
}

TEST(PDAF, init_with_params)
{
    SimplePDAF pdaf(1.0, 0.9, 0.1);
    EXPECT_EQ(pdaf.gate_threshold_, 1.0);
    EXPECT_EQ(pdaf.prob_of_detection_, 0.9);
    EXPECT_EQ(pdaf.clutter_intensity_, 0.1);
}


// testing the get_weights function
TEST(PDAF, get_weights_is_calculating)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::MultiVarGauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};

    Eigen::VectorXd weights = pdaf.get_weights(meas, z_pred);

    std::cout << "weights: " << weights << std::endl;

    EXPECT_EQ(weights.size(), 3);
}

TEST(PDAF, if_no_clutter_first_weight_is_zero)
{
    SimplePDAF pdaf(2.0, 0.8, 0.0);

    vortex::prob::MultiVarGauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};

    Eigen::VectorXd weights = pdaf.get_weights(meas, z_pred);

    std::cout << "weights: " << weights << std::endl;

    EXPECT_EQ(weights(0), 0.0);
} 

TEST(PDAF, weights_are_decreasing_with_distance)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::MultiVarGauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{2.0, 1.0}, {3.0, 1.0}, {4.0, 1.0}};

    Eigen::VectorXd weights = pdaf.get_weights(meas, z_pred);

    std::cout << "weights: " << weights << std::endl;

    EXPECT_GT(weights(1), weights(2));
    EXPECT_GT(weights(2), weights(3));
}


// testing the get_weighted_average function
TEST(PDAF, get_weighted_average_is_calculating)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::MultiVarGauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    vortex::prob::MultiVarGauss4d x_pred(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};
    std::vector<vortex::prob::MultiVarGauss4d> updated_states = {
        vortex::prob::MultiVarGauss4d(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity()),
        vortex::prob::MultiVarGauss4d(Eigen::Vector4d(1.0, 1.0, 1.0, 1.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::MultiVarGauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_y_axis)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::MultiVarGauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
    vortex::prob::MultiVarGauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{1.0, 2.0}};
    std::vector<vortex::prob::MultiVarGauss4d> updated_states = {
        vortex::prob::MultiVarGauss4d(Eigen::Vector4d(1.0, 1.5, 0.0, 0.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::MultiVarGauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    EXPECT_GT(weighted_average.mean()(1), x_pred.mean()(1));
    EXPECT_GT(weighted_average.mean()(1), z_pred.mean()(1));
    EXPECT_LT(weighted_average.mean()(1), meas[0](1));
    EXPECT_LT(weighted_average.mean()(1), updated_states[0].mean()(1));

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}