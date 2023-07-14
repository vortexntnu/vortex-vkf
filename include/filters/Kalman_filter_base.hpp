#pragma once
#include <models/model_definitions.hpp>


namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class Kalman_filter_base {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

	Kalman_filter_base(State x0, Mat_xx P0) : _x0{x0}, _P0_xx{P0}, _x{x0}, _P_xx{P0} {}
	virtual ~Kalman_filter_base() {}
	
	virtual State iterate(Timestep Ts, const Measurement& y, const Input& u = Input::Zero()) = 0;
	void set_state(State x_n) { _x = x_n; }
	void set_covariance(Mat_xx P_n) { _P_xx = P_n; }
	virtual void reset() 
	{
		set_state(_x0);
		set_covariance(_P0_xx);
	}
	State get_state() const { return _x; }
	Mat_xx get_covariance() const { return _P_xx; }
protected:
	const State _x0;
	const Mat_xx _P0_xx;
	State _x;
	Mat_xx _P_xx;
};
}