#pragma once
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <memory>

#include <kalman_filters/Kalman_filter_base.hpp>
#include <models/model_definitions.hpp>
#include <kalman_filters/UKF.hpp>

namespace Nodes {
using namespace Models;
using Period = std::chrono::milliseconds;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class KF_node : public rclcpp::Node {
public:
    DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	KF_node(Filters::Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>* filter, Period P = 100ms) : _filter{filter}, _P{P}
    {
	    _timer = this->create_wall_timer(P, std::bind(&KF_node<n_x,n_y,n_u,n_v,n_w>::timer_callback, this));
    }

private:
    rclcpp::TimerBase::SharedPtr _timer;
    Filters::Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>* _filter;
    const Period _P;
    rclcpp::Time _last_timestamp;
    Measurement  _last_measurement;

    void timer_callback()
    {
        // Get time since last call
        rclcpp::Time now = this->now();
        Timestep Ts = (now - _last_timestamp).nanoseconds()/1000; // Time in milliseconds
        _last_timestamp = now;
        // Reset filter if time goes backwards (e.g. when playing a rosbag on loop)
        if (Ts < (Timestep)0)
        {
            _filter->reset();
            RCLCPP_INFO_STREAM(this->get_logger(), "Reset filter, period is " << Ts.count() << "ms");
            return;
        }
        // Calculate next iterate
        State x_next = _filter->next_state(Ts, _last_measurement);
        // Publish 
        RCLCPP_INFO_STREAM(this->get_logger(), x_next);
    }
};


int main()
{
    return 0;
}
}