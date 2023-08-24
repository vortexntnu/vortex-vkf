#pragma once

#include <models/Model_base.hpp>
#include <models/model_definitions.hpp>
#include <integration_methods/ERK_methods.hpp>

namespace Models {
constexpr int n_x = 7, n_y = 7, n_u = 1, n_v = 6, n_w = 6;
using Base = Model_base<Integrator::RK4<n_x>, n_x, n_y, n_u, n_v, n_w>;
class Temp_gyro_model : public Base {
public:
	DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)
	using Quaternion = Eigen::Quaterniond;

	Temp_gyro_model() : Base(){};

	State f(Time t, const State &x, const Input &u = Input::Zero(), const Disturbance &v = Disturbance::Zero()) const override final
	{
		(void)t;
		(void)u;
		Quaternion q(x(0), x(1), x(2), x(3));
		Quaternion q_dot = q * Quaternion(0, v(0), v(1), v(2));
		q_dot.coeffs() *= 0.5;
		State x_dot;
		x_dot << q_dot.w(), q_dot.x(), q_dot.y(), q_dot.z(), v(3), v(4), v(5);

		return x_dot;
	}

	Measurement h(Time t, const State &x, const Input &u = Input::Zero(), const Noise &w = Noise::Zero()) const override final
	{
		(void)t;
		(void)x;
		(void)u;

		Measurement z;
		z << 0, w;
		return z;
	}
};
} // namespace Models