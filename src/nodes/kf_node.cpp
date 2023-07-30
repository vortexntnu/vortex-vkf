#include <nodes/kf_node.hpp>
#include <filters/KF.hpp>
#include <filters/UKF.hpp>
#include <models/LTI_model.hpp>
#include <models/temp_gyro_model.hpp>
#include <integration_methods/ERK_methods.hpp>

#include <iostream>
#include <memory>


int main(int argc, char **argv)
{
     using namespace Models;
     constexpr int n_x = Temp_gyro_model::_Nx;
     constexpr int n_y = Temp_gyro_model::_Ny;
     constexpr int n_u = Temp_gyro_model::_Nu;
     constexpr int n_v = Temp_gyro_model::_Nv;
     constexpr int n_w = Temp_gyro_model::_Nw;
     DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

     rclcpp::init(argc, argv);

     // Create model
     auto model = std::make_shared<Temp_gyro_model>();
     // Create integrator
     auto integrator = std::make_shared<Integrator::RK4_M<Temp_gyro_model>>();
     // Create filter
     State x0 = State::Zero();
     Mat_xx P0 = Mat_xx::Identity();
     auto ukf = std::make_shared<Filters::UKF_M<Temp_gyro_model>>(model, x0, P0);

     auto node = std::make_shared<Nodes::KF_node_M<Temp_gyro_model>>(ukf, 0.1s);

     rclcpp::spin(node);
     rclcpp::shutdown();
}