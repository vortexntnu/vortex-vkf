#include <gtest/gtest.h>
#include <eigen3/Eigen/Eigen>

class MatrixSizeTest : public ::testing::Test {
protected:
    static constexpr int n_x{3}, n_y{1}, n_u{2}, n_v{3}, n_w{1};
	void SetUp() override
	{
        x << 1, 2, 2;
        y << 1;
        u << 1, 1;
        v << 2, 2, 1;
        w << 3;
        A = A.Identity();
        B << 1, 0, 
             0, 0, 
             0, 0;
        C << 1, 0, 0;	
        D = D.Zero();
        Q << 1, 0, 0,
             0, 2, 0,
             0, 0, 3;
        R << 1;
	}

    Eigen::Vector<double,n_x> x;
    Eigen::Vector<double,n_y> y;
    Eigen::Vector<double,n_u> u;
    Eigen::Vector<double,n_v> v;
    Eigen::Vector<double,n_w> w;

    Eigen::Matrix<double,n_x,n_x> A;
	Eigen::Matrix<double,n_x,n_u> B;
    Eigen::Matrix<double,n_y,n_x> C;
    Eigen::Matrix<double,n_y,n_u> D;
    Eigen::Matrix<double,n_v,n_v> Q;
    Eigen::Matrix<double,n_w,n_w> R;
    Eigen::Matrix<double,n_x,n_v> G;
    Eigen::Matrix<double,n_y,n_w> H;
};

TEST_F(MatrixSizeTest, matrixSize)
{
    ASSERT_EQ(A.cols(), x.rows()) << "Number of columns of A must match number of rows of x"; 
    ASSERT_EQ(B.cols(), u.rows()) << "Number of columns of B must match number of rows of u";
	ASSERT_EQ(G.cols(), v.rows()) << "Number of columns of G must match number of rows of v";

	ASSERT_EQ((A*x).size(), (x).size()) << "Size of Ax must equal x";
	ASSERT_EQ((B*u).size(), (x).size()) << "Size of Bu must equal x";
	ASSERT_EQ((G*v).size(), (x).size()) << "Size of Gv must equal x";

	ASSERT_EQ((C*x).size(), (y).size()) << "Size of Cx must equal y";
	ASSERT_EQ((D*u).size(), (y).size()) << "Size of Du must equal y";
	ASSERT_EQ((H*w).size(), (y).size()) << "Size of Hw must equal y";
}


