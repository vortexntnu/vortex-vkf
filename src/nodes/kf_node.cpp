#include <nodes/kf_node.hpp>


#include <kalman_filters/KF.hpp>
#include <kalman_filters/UKF.hpp>
#include <models/LTI_model.hpp>
#include <iostream>
int main()
{
     const int n_x{3}, n_y{1}, n_u{1}, n_v{n_x}, n_w{n_y};
     DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

     Models::LTI_model<n_x,n_y,n_u> model;
     Filters::UKF<n_x,n_y,n_u> ukf(&model, State::Zero(), Mat_vv::Zero());
}