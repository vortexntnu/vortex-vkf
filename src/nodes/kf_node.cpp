#include <filters/UKF.hpp>
#include <integration_methods/ERK_methods.hpp>
#include <models/temp_gyro_model.hpp>
#include <nodes/kf_node.hpp>

#include <iostream>
#include <memory>

int main(int argc, char **argv)
{
	using namespace Models;
	constexpr int n_x = Temp_gyro_model::_n_x;
	constexpr int n_y = Temp_gyro_model::_n_y;
	constexpr int n_u = Temp_gyro_model::_n_u;
	constexpr int n_v = Temp_gyro_model::_n_v;
	constexpr int n_w = Temp_gyro_model::_n_w;
	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)

	rclcpp::init(argc, argv);

	// Create model
	auto model = std::make_shared<Temp_gyro_model>();
	// Create integrator
	auto integrator = std::make_shared<Integrator::RK4<n_x>>();
	// Create filter
	State x0  = State::Zero();
	Mat_xx P0 = Mat_xx::Identity();
	auto ukf  = std::make_shared<Filters::UKF<Temp_gyro_model>>(model, x0, P0);

	auto node = std::make_shared<Nodes::KF_node<Temp_gyro_model>>(ukf, 0.1s);
	RCLCPP_INFO(node->get_logger(), "KF node created");
	rclcpp::spin(node);
	rclcpp::shutdown();
}