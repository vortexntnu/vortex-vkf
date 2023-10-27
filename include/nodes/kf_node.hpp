#pragma once
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
using std::placeholders::_1;

#include <chrono>
#include <memory>

#include <filters/Kalman_filter_base.hpp>
#include <filters/UKF.hpp>
#include <models/model_definitions.hpp>
// #include <models/temp_gyro_model.hpp>
namespace Nodes {
using namespace Models;
using Period = Timestep;

template <class Model> class KF_node : public rclcpp::Node {
public:
	static constexpr int _n_x = Model::_n_x, _n_y = Model::_n_y, _n_u = Model::_n_u, _n_v = Model::_n_v, _n_w = Model::_n_w;
	DEFINE_MODEL_TYPES(_n_x, _n_y, _n_u, _n_v, _n_w)
	KF_node(std::shared_ptr<Filters::Kalman_filter_base<Model>> filter, Period P = 0.1s) : Node("KF_node"), _filter{filter}, _P{P}
	{
		_timer               = this->create_wall_timer(P, std::bind(&KF_node<Model>::timer_callback, this));
		_last_timestamp      = this->now();
		_has_new_measurement = false;

		subscription_ =
		    this->create_subscription<geometry_msgs::msg::PoseStamped>("measurement", 10, std::bind(&KF_node<Model>::meas_callback, this, _1));
	}

private:
	rclcpp::TimerBase::SharedPtr _timer;
	std::shared_ptr<Filters::Kalman_filter_base<Model>> _filter;
	const Period _P;
	rclcpp::Time _last_timestamp;
	Measurement _last_measurement;
	bool _has_new_measurement;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;

	void timer_callback()
	{
		// Get time since last call
		rclcpp::Time now = this->now();
		Time Ts          = (now - _last_timestamp).seconds() * 1s;
		_last_timestamp  = now;
		// Reset filter if time goes backwards (e.g. when playing a rosbag on loop)
		if (Ts < (Time)0) {
			_filter->reset();
			RCLCPP_INFO_STREAM(this->get_logger(), "Reset filter due to negative time step");
			return;
		}
		// Calculate next iterate
		State x_next         = _filter->iterate(Ts, _last_measurement);
		_has_new_measurement = false;
		// Publish
		RCLCPP_INFO_STREAM(this->get_logger(), '\n' << x_next);
	}

	void meas_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
	{
		RCLCPP_INFO_STREAM(this->get_logger(), "Measurement: " << msg->pose.position.x << ", " << msg->pose.position.y << ", " << msg->pose.position.z);
		_last_measurement << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
		_has_new_measurement = true;
	}
};

} // namespace Nodes