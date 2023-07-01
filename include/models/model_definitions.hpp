#pragma once
#include <chrono>
#include <eigen3/Eigen/Eigen>

namespace Models {
using Eigen::Vector;
using Eigen::Matrix;
using namespace std::chrono_literals;
using Timestep = std::chrono::milliseconds;

#define DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)     \
    using State 	  = Vector<double,n_x>;      \
	using Measurement = Vector<double,n_y>;      \
	using Input       = Vector<double,n_u>;      \
	using Disturbance = Vector<double,n_v>;      \
	using Noise       = Vector<double,n_y>;      \
	using Mat_xx      = Matrix<double,n_x,n_x>;  \
	using Mat_yy      = Matrix<double,n_y,n_y>;  \
	using Mat_vv      = Matrix<double,n_v,n_v>;  \
	using Mat_ww      = Matrix<double,n_w,n_w>;  \
	using Mat_xy 	  = Matrix<double,n_x,n_y>;  \
	using Mat_xu 	  = Matrix<double,n_x,n_u>;  \
	using Mat_xv 	  = Matrix<double,n_x,n_v>;  \
	using Mat_yx 	  = Matrix<double,n_y,n_x>;  \
	using Mat_yu 	  = Matrix<double,n_y,n_u>;  \
	using Mat_yw 	  = Matrix<double,n_y,n_w>;      
} // namespace Models
