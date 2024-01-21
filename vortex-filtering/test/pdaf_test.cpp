#include <gtest/gtest.h>
#include <vortex_filtering/filters/pdaf.hpp>
#include <vortex_filtering/plotting/utils.hpp>
#include <iostream>
#include <gnuplot-iostream.h>

using SimplePDAF = vortex::filter::PDAF<vortex::models::ConstantVelocity<2>, vortex::models::IdentitySensorModel<4, 2>>;

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

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};

    Eigen::VectorXd weights = pdaf.get_weights(meas, z_pred);

    std::cout << "weights: " << weights << std::endl;

    EXPECT_EQ(weights.size(), 3);
}

TEST(PDAF, if_no_clutter_first_weight_is_zero)
{
    SimplePDAF pdaf(2.0, 0.8, 0.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};

    Eigen::VectorXd weights = pdaf.get_weights(meas, z_pred);

    std::cout << "weights: " << weights << std::endl;

    EXPECT_EQ(weights(0), 0.0);
} 

TEST(PDAF, weights_are_decreasing_with_distance)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
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

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    vortex::prob::Gauss4d x_pred(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 0.0}, {2.0, 1.0}};
    std::vector<vortex::prob::Gauss4d> updated_states = {
        vortex::prob::Gauss4d(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity()),
        vortex::prob::Gauss4d(Eigen::Vector4d(1.0, 1.0, 1.0, 1.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::Gauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_y_axis)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
    vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{1.0, 2.0}};
    std::vector<vortex::prob::Gauss4d> updated_states = {
        vortex::prob::Gauss4d(Eigen::Vector4d(1.0, 1.5, 0.0, 0.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::Gauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    EXPECT_GT(weighted_average.mean()(1), x_pred.mean()(1));
    EXPECT_GT(weighted_average.mean()(1), z_pred.mean()(1));
    EXPECT_LT(weighted_average.mean()(1), meas[0](1));
    EXPECT_LT(weighted_average.mean()(1), updated_states[0].mean()(1));

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_x_axis)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
    vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{2.0, 1.0}};
    std::vector<vortex::prob::Gauss4d> updated_states = {
        vortex::prob::Gauss4d(Eigen::Vector4d(1.5, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::Gauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    EXPECT_GT(weighted_average.mean()(0), x_pred.mean()(0));
    EXPECT_GT(weighted_average.mean()(0), z_pred.mean()(0));
    EXPECT_LT(weighted_average.mean()(0), meas[0](0));
    EXPECT_LT(weighted_average.mean()(0), updated_states[0].mean()(0));

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

TEST(PDAF, average_state_is_in_between_prediction_and_measurement_both_axes)
{
    SimplePDAF pdaf(2.0, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(1.0, 1.0), Eigen::Matrix2d::Identity());
    vortex::prob::Gauss4d x_pred(Eigen::Vector4d(1.0, 1.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{2.0, 2.0}};
    std::vector<vortex::prob::Gauss4d> updated_states = {
        vortex::prob::Gauss4d(Eigen::Vector4d(1.5, 1.5, 0.0, 0.0), Eigen::Matrix4d::Identity())
    };

    vortex::prob::Gauss4d weighted_average = pdaf.get_weighted_average(meas, updated_states, z_pred, x_pred);

    EXPECT_GT(weighted_average.mean()(0), x_pred.mean()(0));
    EXPECT_GT(weighted_average.mean()(0), z_pred.mean()(0));
    EXPECT_LT(weighted_average.mean()(0), meas[0](0));
    EXPECT_LT(weighted_average.mean()(0), updated_states[0].mean()(0));

    EXPECT_GT(weighted_average.mean()(1), x_pred.mean()(1));
    EXPECT_GT(weighted_average.mean()(1), z_pred.mean()(1));
    EXPECT_LT(weighted_average.mean()(1), meas[0](1));
    EXPECT_LT(weighted_average.mean()(1), updated_states[0].mean()(1));

    std::cout << "weighted average: " << weighted_average.mean() << std::endl;
}

// testing the apply_gate function
TEST(PDAF, apply_gate_is_calculating)
{

    double gate_threshold = 1.8;
    SimplePDAF pdaf(gate_threshold, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},{0.0, 2.0}, {2.0, 0.0}, {2.0, 2.0}};

    auto [inside, outside] = pdaf.apply_gate(meas, z_pred);
}

TEST(PDAF, apply_gate_is_separating_correctly)
{
    double gate_threshold = 2;
    SimplePDAF pdaf(gate_threshold, 0.8, 1.0);
    
    Eigen::Matrix2d cov;
    cov << 1.0, 0.0,
           0.0, 4.0;

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), cov);
    std::vector<Eigen::Vector2d> meas = {{0.0, 4.0}, {4.0, 0.0}};

    auto [inside, outside] = pdaf.apply_gate(meas, z_pred);

    EXPECT_EQ(inside.size(), 1u);
    EXPECT_EQ(outside.size(), 1u);
    EXPECT_EQ(inside[0], meas[0]);
    EXPECT_EQ(outside[0], meas[1]);

    Gnuplot gp;
    gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
    gp << "set style circle radius 0.05\n";
    gp << "plot '-' with circles title 'Samples' fs transparent solid 1 noborder\n";
    gp.send1d(meas);

    int object_counter = 0;

    gp << "set object " << ++object_counter << " circle center " << z_pred.mean()(0) << "," << z_pred.mean()(1) << " size " << 0.05 << " fs empty border lc rgb 'green'\n";
    gp << "replot\n";

    vortex::plotting::Ellipse prediction = vortex::plotting::gauss_to_ellipse(z_pred, 5.991);

    gp << "set object " << ++object_counter << " ellipse center " << prediction.x << "," << prediction.y << " size " << prediction.a << "," << prediction.b << " angle " << prediction.angle
        << "fs empty border lc rgb 'cyan'\n";
    gp << "replot\n";

}

TEST(PDAF, apply_gate_is_separating_correctly_2)
{
    double gate_threshold = 2.1;
    SimplePDAF pdaf(gate_threshold, 0.8, 1.0);

    vortex::prob::Gauss2d z_pred(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix2d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},{0.0, 2.0}, {2.0, 0.0}, {2.0, 2.0}};

    auto [inside, outside] = pdaf.apply_gate(meas, z_pred);

    EXPECT_EQ(inside.size(), 5u);
    EXPECT_EQ(outside.size(), 1u);
}

// testing the predict_next_state function
TEST(PDAF, predict_next_state_is_calculating)
{

    double gate_threshold = 1.0;
    SimplePDAF pdaf(gate_threshold, 0.8, 1.0);

    vortex::prob::Gauss4d x_est(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0), Eigen::Matrix4d::Identity());
    std::vector<Eigen::Vector2d> meas = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},{0.0, 2.0}, {2.0, 0.0}, {2.0, 2.0}};

    auto dyn_model = std::make_shared<vortex::models::ConstantVelocity<2>>(1.0);
    auto sen_model = std::make_shared<vortex::models::IdentitySensorModel<4, 2>>(1.0);

    auto [x_final, inside, outside, x_pred, z_pred, x_updated] = pdaf.predict_next_state(x_est, meas, 1.0, dyn_model, sen_model);
    std::cout << "x_final: " << x_final.mean() << std::endl;

    Gnuplot gp;
    gp << "set xrange [-8:8]\nset yrange [-8:8]\n";
    gp << "set style circle radius 0.05\n";
    gp << "plot '-' with circles title 'Samples' fs transparent solid 1 noborder\n";
    gp.send1d(meas);

    int object_counter = 0;

    gp << "set object " << ++object_counter << " circle center " << x_est.mean()(0) << "," << x_est.mean()(1) << " size " << 0.05 << "fs empty border lc rgb 'black'\n";
    gp << "set object " << ++object_counter << " circle center " << x_final.mean()(0) << "," << x_final.mean()(1) << " size " << 0.05 << " fs empty border lc rgb 'green'\n";
    gp << "replot\n";

    vortex::plotting::Ellipse gate = vortex::plotting::gauss_to_ellipse(z_pred, gate_threshold);

    gp << "set object " << ++object_counter << " ellipse center " << gate.x << "," << gate.y << " size " << gate.a << "," << gate.b << " angle " << gate.angle
        << "fs empty border lc rgb 'cyan'\n";
    gp << "replot\n";

    for (const auto& state: x_updated)
    {   
        vortex::prob::Gauss2d gauss(state.mean().head(2), state.cov().topLeftCorner(2,2));
        vortex::plotting::Ellipse ellipse = vortex::plotting::gauss_to_ellipse(gauss); 

        gp << "set object " << ++object_counter << " ellipse center " << ellipse.x << "," << ellipse.y << " size " << ellipse.a << "," << ellipse.b << " angle " << ellipse.angle << "fs empty border lc rgb 'blue'\n";
    };
    gp << "replot\n";

}