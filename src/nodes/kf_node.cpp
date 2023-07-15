#include <nodes/kf_node.hpp>
#include <filters/KF.hpp>
#include <filters/UKF.hpp>
#include <models/LTI_model.hpp>

#include <iostream>
#include <memory>

constexpr int n_x = 1, n_y = 1, n_u = 1, n_v = 1, n_w = 1;
using namespace Models;
using namespace Filters;
constexpr int n_a = n_x+n_v+n_w; // Size of augmented state
using Mat_aa  = Matrix<double,n_a,n_a>;
using State_a = Vector<double,n_a>;
DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)

int main()
{
     // Make augmented covariance matrix
     Mat_xx P;
     Mat_vv Q;
     Mat_ww R;
     P << 1;
     Q << 3;
     R << 1;

     Mat_aa P_a;
     P_a << 	P		    , Mat_xv::Zero(), Mat_xw::Zero(),
               Mat_vx::Zero(), Q			, Mat_vw::Zero(),
               Mat_wx::Zero(), Mat_wv::Zero(), R			 ;


     std::cout << "P: \n" << P << std::endl;
     Mat_aa sqrt_P_a = P_a.llt().matrixLLT();
     std::cout << "P_a: \n" << P_a << std::endl;
     std::cout << "sqrt_P_a: \n" << sqrt_P_a << std::endl;
     std::cout << "reconstructed P_a: \n" << sqrt_P_a*sqrt_P_a.transpose() << std::endl;
}