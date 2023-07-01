#pragma once
#include <kalman_filters/Kalman_filter_base.hpp>
#include <models/Model_base.hpp>

namespace Filters {
using namespace Models;

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class UKF : public Kalman_filter_base<n_x,n_y,n_u,n_v,n_w> {
public:
	DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
	UKF(Models::Model_base<n_x,n_y,n_u,n_v,n_w> *model, State x0, Mat_xx P0) : Kalman_filter_base<n_x,n_y,n_u,n_v,n_w>(x0, P0) {}

private:
    Model_base<n_x,n_y,n_u,n_v,n_w>* model;
};
}