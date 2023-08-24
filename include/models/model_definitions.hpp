#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>

namespace Models {
using Eigen::Matrix;
using Eigen::Vector;
using namespace std::chrono_literals;
using namespace std::chrono;
using Time     = duration<double>;
using Timestep = duration<double>;

/**
 * @brief Define the types used in the models. 
 * THIS HAS TO BE USED INSIDE THE Models NAMESPACE
 * @param n_x State size
 * @param n_y Measurement size
 * @param n_u Input size
 * @param n_v Disturbance size
 * @param n_w Noise size
 */
#define DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)                                                                                                            \
	using State       = Eigen::Vector<double, n_x>;                                                                                                              \
	using Measurement = Eigen::Vector<double, n_y>;                                                                                                              \
	using Input       = Eigen::Vector<double, n_u>;                                                                                                              \
	using Disturbance = Eigen::Vector<double, n_v>;                                                                                                              \
	using Noise       = Eigen::Vector<double, n_w>;                                                                                                              \
                                                                                                                                                               \
	using Mat_xx = Eigen::Matrix<double, n_x, n_x>;                                                                                                              \
	using Mat_xy = Eigen::Matrix<double, n_x, n_y>;                                                                                                              \
	using Mat_xu = Eigen::Matrix<double, n_x, n_u>;                                                                                                              \
	using Mat_xv = Eigen::Matrix<double, n_x, n_v>;                                                                                                              \
	using Mat_xw = Eigen::Matrix<double, n_x, n_w>;                                                                                                              \
                                                                                                                                                               \
	using Mat_yx = Eigen::Matrix<double, n_y, n_x>;                                                                                                              \
	using Mat_yy = Eigen::Matrix<double, n_y, n_y>;                                                                                                              \
	using Mat_yu = Eigen::Matrix<double, n_y, n_u>;                                                                                                              \
	using Mat_yv = Eigen::Matrix<double, n_y, n_v>;                                                                                                              \
	using Mat_yw = Eigen::Matrix<double, n_y, n_w>;                                                                                                              \
                                                                                                                                                               \
	using Mat_vx = Eigen::Matrix<double, n_v, n_x>;                                                                                                              \
	using Mat_vy = Eigen::Matrix<double, n_v, n_y>;                                                                                                              \
	using Mat_vu = Eigen::Matrix<double, n_v, n_u>;                                                                                                              \
	using Mat_vv = Eigen::Matrix<double, n_v, n_v>;                                                                                                              \
	using Mat_vw = Eigen::Matrix<double, n_v, n_w>;                                                                                                              \
                                                                                                                                                               \
	using Mat_wx = Eigen::Matrix<double, n_w, n_x>;                                                                                                              \
	using Mat_wy = Eigen::Matrix<double, n_w, n_y>;                                                                                                              \
	using Mat_wu = Eigen::Matrix<double, n_w, n_u>;                                                                                                              \
	using Mat_wv = Eigen::Matrix<double, n_w, n_v>;                                                                                                              \
	using Mat_ww = Eigen::Matrix<double, n_w, n_w>;                                                                                                              \
                                                                                                                                                               \
	using State_dot = std::function<State(Time t, const State &x)>;

} // namespace Models
