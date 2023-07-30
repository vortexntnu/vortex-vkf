#pragma once
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>

#include <filters/Kalman_filter_base.hpp>
#include <models/model_definitions.hpp>
#include <filters/UKF.hpp>
#include <models/temp_gyro_model.hpp>
namespace Nodes {
using namespace Models;
using Period = Timestep;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class KF_node : public rclcpp::Node {
public:
    DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	KF_node(std::shared_ptr<Filters::Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>> filter, Period P = 0.1s) : Node("KF_node"), _filter{filter}, _P{P}
    {
	    _timer = this->create_wall_timer(P, std::bind(&KF_node<n_x,n_y,n_u,n_v,n_w>::timer_callback, this));
        _last_timestamp = this->now();
    }

private:
    rclcpp::TimerBase::SharedPtr _timer;
    std::shared_ptr<Filters::Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>> _filter;
    const Period _P;
    rclcpp::Time _last_timestamp;
    Measurement  _last_measurement;

    void timer_callback()
    {
        // Get time since last call
        rclcpp::Time now = this->now();
        Time Ts = (now - _last_timestamp).seconds()*1s;
        _last_timestamp = now;
        // Reset filter if time goes backwards (e.g. when playing a rosbag on loop)
        if (Ts < (Time)0)
        {
            _filter->reset();
            RCLCPP_INFO_STREAM(this->get_logger(), "Reset filter, period is " << Ts/1ms << "ms");
            return;
        }
        // Calculate next iterate
        State x_next = _filter->iterate(Ts, _last_measurement);
        // Publish 
        RCLCPP_INFO_STREAM(this->get_logger(), '\n' << x_next);
    }
};



} // namespace Nodes