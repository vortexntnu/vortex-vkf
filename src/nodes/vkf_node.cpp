#include <iostream>
#include <models/model_definitions.hpp>
#include <models/LTI_model.hpp>
using namespace Models;

int main()
{
	const int n_x{3}, n_y{1}, n_u{2};

    Eigen::Vector<double,n_x> x0;
    Eigen::Vector<double,n_u> u0;
    Eigen::Vector<double,n_x> v0;

	x0 << 1, 2, 2;

	u0 << 1, 1;
	
	v0 << 2, 2, 1;

	Eigen::Matrix<double,n_x,n_x> A;
	Eigen::Matrix<double,n_x,n_u> B;
	Eigen::Matrix<double,n_y,n_x> C;
	Eigen::Matrix<double,n_x,n_x> Q;
	Eigen::Matrix<double,n_y,n_y> R;

	A = A.Identity();

	B << 1, 0, 
		 0, 0, 
		 0, 0;

	C << 1, 0, 0;	

	Q << 1, 0, 0,
		 0, 2, 0,
		 0, 0, 3;

	R << 1;


	LTI_model2<n_x, n_y, n_u> target{A,B,C,Q,R};
	Eigen::Vector3d x1 = target.f(1ms, x0, u0, v0);
	std::cout << x0 << "\n\n";
	std::cout << x1 << "\n\n";
}