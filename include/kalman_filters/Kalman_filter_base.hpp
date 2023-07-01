#pragma once
#include <models/model_definitions.hpp>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v, int n_w>
class Kalman_filter_base {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

	Kalman_filter_base(State x0, Mat_xx P0) : _x{x0}, _P_xx{P0} {}

	virtual State next_state(Timestep Ts, Measurement y, Input u, Disturbance v, Noise w) = 0;
	void setState(State x_n) { _x = x_n; }
	void setCovariance(Mat_xx P_n) { _P_xx = P_n; }

protected:
	State _x;
	Mat_xx _P_xx;
};
}