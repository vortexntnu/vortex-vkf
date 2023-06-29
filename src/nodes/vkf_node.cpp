#include <iostream>
#include <models/model_definitions.hpp>
#include <models/LTI_model.hpp>
using namespace Models;

int main()
{
	const int n_x{3}, n_y{1}, n_u{2}, n_v{1};

	Eigen::Vector3d x0;
	x0 << 1, 2, 2;

	Eigen::Vector2d u0;
	u0 << 1, 1;
	
	Eigen::Vector<double,1> v0{1};
	v0 << 2;

	Eigen::Matrix<double,n_x,n_x> A;
	A = A.Identity();
	Eigen::Matrix<double,n_x,n_u> B;
	B << 1, 0, 
		 0, 0, 
		 0, 0;
	Eigen::Matrix<double,n_y,n_x> C;
	C << 1, 0, 0;
	Eigen::Matrix<double,n_x,n_v> G;
	G << 0,
	     1,
		 0;
	Eigen::Matrix<double,n_x,n_x> Q;
	Q << 1, 0, 0,
		 0, 2, 0,
		 0, 0, 3;
	Eigen::Matrix<double,n_y,n_y> R;
	R << 1;


	LTI_model<n_x, n_y, n_u, n_v> target{A,B,C,G,Q,R};
	Eigen::Vector3d x1 = target.f(1ms, x0, u0, v0);
	std::cout << x0 << "\n\n";
	std::cout << x1 << "\n\n";
}