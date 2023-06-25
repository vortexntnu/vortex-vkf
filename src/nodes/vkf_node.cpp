#include <iostream>
#include <models/model_definitions.hpp>
#include <models/LTI_model.hpp>
using namespace Models;
using namespace std::chrono_literals;
int main()
{
	State x0{3};
	x0 << 1, 2, 2;

	Mat A = Eigen::Matrix3d::Identity();
	Mat B = Eigen::Vector3d::Ones();
	Mat C = Eigen::Matrix3d::Zero();
	Mat D = Eigen::Matrix3d::Zero();
	Mat Q = Eigen::Matrix3d::Identity();
	Mat R = Eigen::Matrix<double,1,1>::Identity();

	LTI_model target{A,B,C,D,Q,R};
	State x1 = target.f(1ms, x0);
	std::cout << x0 << "\n\n";
	std::cout << x1;
}