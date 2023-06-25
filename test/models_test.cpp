#include <gtest/gtest.h>
#include <models/LTI_model.hpp>
using namespace Models;

class LTImodelTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		A = Eigen::Matrix3d::Identity();
		B = Eigen::VectorXd{3};
		C = Eigen::Vector3d::Zero();
		G = Eigen::Vector3d::Zero();
		Q = Eigen::Matrix3d::Identity();
		R = Eigen::Matrix<double,1,1>::Identity();
		B << 1,0,0;
		x = State{3};
		u = Input{1};
		v = Disturbance{1};
		w = Noise{1};
		x << 1,2,3;
		u << 4;
		v << 1;
		w << 0;
	}
	Mat A;
	Mat B;
	Mat C;
	Mat G;
	Mat Q;
	Mat R;
	State x;
	Input u;
	Disturbance v;
	Noise w;
};

TEST_F(LTImodelTest, matrixSize)
{
    ASSERT_EQ(A.cols(), x.rows()) << "Number of columns of A must match number of rows of x"; 
    ASSERT_EQ(B.cols(), u.rows()) << "Number of columns of B must match number of rows of u";
	ASSERT_EQ(G.cols(), v.rows()) << "Number of columns of G must match number of rows of v";
}


