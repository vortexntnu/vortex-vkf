#include <iostream>
#include <models/model_definitions.hpp>
#include <models/LTI_model.hpp>
using namespace Models;

int main()
{
	State x0{3};
	x0 << 1, 2, 2;
	Input u0{1};
	u0 << 1;
	Disturbance v0{1};
	v0 << 2;

	Mat A = Eigen::Matrix3d::Identity();
	Mat B = Eigen::Vector3d::Ones();
	Mat C = Eigen::Vector3d::Zero();
	Mat D = Eigen::Matrix3d::Zero();
	Mat Q = Eigen::Matrix3d::Identity();
	Mat R = Eigen::Matrix<double,1,1>::Identity();
	R * Q;
	LTI_model target{A,B,C,D,Q,R};
	State x1 = target.f(1ms, x0, u0, v0);
	std::cout << x0 << "\n\n";
	std::cout << x1 << "\n\n";
}