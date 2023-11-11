#pragma once
#include <vortex_filtering/models/model_definitions.hpp>

namespace Filters {
using namespace Models;

template <class Model> class Kalman_filter_base {
public:
	static constexpr int _n_x = Model::_n_x, _n_y = Model::_n_y, _n_u = Model::_n_u, _n_v = Model::_n_v, _n_w = Model::_n_w;
	DEFINE_MODEL_TYPES(_n_x, _n_y, _n_u, _n_v, _n_w)

	Kalman_filter_base(State x0, Mat_xx P0) : _x0{x0}, _P0_xx{P0}, _x{x0}, _P_xx{P0} {}
	virtual ~Kalman_filter_base() {}

	virtual State iterate(Time Ts, const Measurement &y, const Input &u = Input::Zero()) = 0;
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
} // namespace Filters