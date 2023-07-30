#include <nodes/kf_node.hpp>
#include <filters/KF.hpp>
#include <filters/UKF.hpp>
#include <models/LTI_model.hpp>
#include <models/temp_gyro_model.hpp>
#include <integration_methods/ERK_methods.hpp>

#include <iostream>
#include <memory>

constexpr int n_x = 7, n_y = 3, n_u = 3, n_v = 6, n_w = 3;

using namespace Models;
DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

int main(int argc, char **argv)
{
     rclcpp::init(argc, argv);

     // Create model
     auto model = std::make_shared<Models::Temp_gyro_model>();
     // Create integrator
     auto integrator = std::make_shared<Integrator::RK4<n_x,n_u,n_v>>();
     // Create filter
     State x0 = State::Zero();
     Mat_xx P0 = Mat_xx::Identity();
     auto ukf = std::make_shared<Filters::UKF<n_x,n_y,n_u,n_v,n_w>>(model, x0, P0);

     auto node = std::make_shared<Nodes::KF_node<n_x,n_y,n_u,n_v,n_w>>(ukf, 0.1s);

     rclcpp::spin(node);
     rclcpp::shutdown();
}