#include <eigen3/Eigen/Eigen>
#include <gtest/gtest.h>
#include <models/model_definitions.hpp>
using namespace Models;
class MatrixSizeTest : public ::testing::Test {
protected:
	static constexpr int n_x{3}, n_y{1}, n_u{2}, n_v{3}, n_w{1};

	Eigen::Vector<double, n_x> x;
	Eigen::Vector<double, n_y> y;
	Eigen::Vector<double, n_u> u;
	Eigen::Vector<double, n_v> v;
	Eigen::Vector<double, n_w> w;

	Eigen::Matrix<double, n_x, n_x> A;
	Eigen::Matrix<double, n_x, n_u> B;
	Eigen::Matrix<double, n_y, n_x> C;
	Eigen::Matrix<double, n_y, n_u> D;
	Eigen::Matrix<double, n_v, n_v> Q;
	Eigen::Matrix<double, n_w, n_w> R;
	Eigen::Matrix<double, n_x, n_v> G;
	Eigen::Matrix<double, n_y, n_w> H;

	template<int n_x, int n_y, int n_u, int n_v, int n_w>
	void testSizes()
	{
		DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_v, n_w)
		ASSERT_EQ(State::RowsAtCompileTime, n_x) 		<< "Number of rows of State must match n_x = " << n_x;
		ASSERT_EQ(Measurement::RowsAtCompileTime, n_y) 	<< "Number of rows of Measurement must match n_y = " << n_y;
		ASSERT_EQ(Input::RowsAtCompileTime, n_u) 		<< "Number of rows of Input must match n_u = " << n_u;
		ASSERT_EQ(Disturbance::RowsAtCompileTime, n_v) 	<< "Number of rows of Disturbance must match n_v = " << n_v;
		ASSERT_EQ(Noise::RowsAtCompileTime, n_w) 		<< "Number of rows of Noise must match n_w = " << n_w;

		ASSERT_EQ(Mat_xx::RowsAtCompileTime, n_x) 		<< "Number of rows of Mat_xx must match n_x = " << n_x;
		ASSERT_EQ(Mat_xx::ColsAtCompileTime, n_x) 		<< "Number of columns of Mat_xx must match n_x = " << n_x;
		ASSERT_EQ(Mat_xy::RowsAtCompileTime, n_x) 		<< "Number of rows of Mat_xy must match n_x = " << n_x;
		ASSERT_EQ(Mat_xy::ColsAtCompileTime, n_y) 		<< "Number of columns of Mat_xy must match n_y = " << n_y;
		ASSERT_EQ(Mat_xu::RowsAtCompileTime, n_x) 		<< "Number of rows of Mat_xu must match n_x = " << n_x;
		ASSERT_EQ(Mat_xu::ColsAtCompileTime, n_u) 		<< "Number of columns of Mat_xu must match n_u = " << n_u;
		ASSERT_EQ(Mat_xv::RowsAtCompileTime, n_x) 		<< "Number of rows of Mat_xv must match n_x = " << n_x;
		ASSERT_EQ(Mat_xv::ColsAtCompileTime, n_v) 		<< "Number of columns of Mat_xv must match n_v = " << n_v;
		ASSERT_EQ(Mat_xw::RowsAtCompileTime, n_x) 		<< "Number of rows of Mat_xw must match n_x = " << n_x;
		ASSERT_EQ(Mat_xw::ColsAtCompileTime, n_w) 		<< "Number of columns of Mat_xw must match n_w = " << n_w;

		ASSERT_EQ(Mat_yx::RowsAtCompileTime, n_y) 		<< "Number of rows of Mat_yx must match n_y = " << n_y;
		ASSERT_EQ(Mat_yx::ColsAtCompileTime, n_x) 		<< "Number of columns of Mat_yx must match n_x = " << n_x;
		ASSERT_EQ(Mat_yy::RowsAtCompileTime, n_y) 		<< "Number of rows of Mat_yy must match n_y = " << n_y;
		ASSERT_EQ(Mat_yy::ColsAtCompileTime, n_y) 		<< "Number of columns of Mat_yy must match n_y = " << n_y;
		ASSERT_EQ(Mat_yu::RowsAtCompileTime, n_y) 		<< "Number of rows of Mat_yu must match n_y = " << n_y;
		ASSERT_EQ(Mat_yu::ColsAtCompileTime, n_u) 		<< "Number of columns of Mat_yu must match n_u = " << n_u;
		ASSERT_EQ(Mat_yv::RowsAtCompileTime, n_y) 		<< "Number of rows of Mat_yv must match n_y = " << n_y;
		ASSERT_EQ(Mat_yv::ColsAtCompileTime, n_v) 		<< "Number of columns of Mat_yv must match n_v = " << n_v;
		ASSERT_EQ(Mat_yw::RowsAtCompileTime, n_y) 		<< "Number of rows of Mat_yw must match n_y = " << n_y;
		ASSERT_EQ(Mat_yw::ColsAtCompileTime, n_w) 		<< "Number of columns of Mat_yw must match n_w = " << n_w;

		ASSERT_EQ(Mat_ux::RowsAtCompileTime, n_u) 		<< "Number of rows of Mat_ux must match n_u = " << n_u;
		ASSERT_EQ(Mat_ux::ColsAtCompileTime, n_x) 		<< "Number of columns of Mat_ux must match n_x = " << n_x;
		ASSERT_EQ(Mat_uy::RowsAtCompileTime, n_u) 		<< "Number of rows of Mat_uy must match n_u = " << n_u;
		ASSERT_EQ(Mat_uy::ColsAtCompileTime, n_y) 		<< "Number of columns of Mat_uy must match n_y = " << n_y;
		ASSERT_EQ(Mat_uu::RowsAtCompileTime, n_u) 		<< "Number of rows of Mat_uu must match n_u = " << n_u;
		ASSERT_EQ(Mat_uu::ColsAtCompileTime, n_u) 		<< "Number of columns of Mat_uu must match n_u = " << n_u;
		ASSERT_EQ(Mat_uv::RowsAtCompileTime, n_u) 		<< "Number of rows of Mat_uv must match n_u = " << n_u;
		ASSERT_EQ(Mat_uv::ColsAtCompileTime, n_v) 		<< "Number of columns of Mat_uv must match n_v = " << n_v;
		ASSERT_EQ(Mat_uw::RowsAtCompileTime, n_u) 		<< "Number of rows of Mat_uw must match n_u = " << n_u;
		ASSERT_EQ(Mat_uw::ColsAtCompileTime, n_w) 		<< "Number of columns of Mat_uw must match n_w = " << n_w;

		ASSERT_EQ(Mat_vx::RowsAtCompileTime, n_v) 		<< "Number of rows of Mat_vx must match n_v = " << n_v;
		ASSERT_EQ(Mat_vx::ColsAtCompileTime, n_x) 		<< "Number of columns of Mat_vx must match n_x = " << n_x;
		ASSERT_EQ(Mat_vy::RowsAtCompileTime, n_v) 		<< "Number of rows of Mat_vy must match n_v = " << n_v;
		ASSERT_EQ(Mat_vy::ColsAtCompileTime, n_y) 		<< "Number of columns of Mat_vy must match n_y = " << n_y;
		ASSERT_EQ(Mat_vu::RowsAtCompileTime, n_v) 		<< "Number of rows of Mat_vu must match n_v = " << n_v;
		ASSERT_EQ(Mat_vu::ColsAtCompileTime, n_u) 		<< "Number of columns of Mat_vu must match n_u = " << n_u;
		ASSERT_EQ(Mat_vv::RowsAtCompileTime, n_v) 		<< "Number of rows of Mat_vv must match n_v = " << n_v;
		ASSERT_EQ(Mat_vv::ColsAtCompileTime, n_v) 		<< "Number of columns of Mat_vv must match n_v = " << n_v;
		ASSERT_EQ(Mat_vw::RowsAtCompileTime, n_v) 		<< "Number of rows of Mat_vw must match n_v = " << n_v;
		ASSERT_EQ(Mat_vw::ColsAtCompileTime, n_w) 		<< "Number of columns of Mat_vw must match n_w = " << n_w;

		ASSERT_EQ(Mat_wx::RowsAtCompileTime, n_w) 		<< "Number of rows of Mat_wx must match n_w = " << n_w;
		ASSERT_EQ(Mat_wx::ColsAtCompileTime, n_x) 		<< "Number of columns of Mat_wx must match n_x = " << n_x;
		ASSERT_EQ(Mat_wy::RowsAtCompileTime, n_w) 		<< "Number of rows of Mat_wy must match n_w = " << n_w;
		ASSERT_EQ(Mat_wy::ColsAtCompileTime, n_y) 		<< "Number of columns of Mat_wy must match n_y = " << n_y;
		ASSERT_EQ(Mat_wu::RowsAtCompileTime, n_w) 		<< "Number of rows of Mat_wu must match n_w = " << n_w;
		ASSERT_EQ(Mat_wu::ColsAtCompileTime, n_u) 		<< "Number of columns of Mat_wu must match n_u = " << n_u;
		ASSERT_EQ(Mat_wv::RowsAtCompileTime, n_w) 		<< "Number of rows of Mat_wv must match n_w = " << n_w;
		ASSERT_EQ(Mat_wv::ColsAtCompileTime, n_v) 		<< "Number of columns of Mat_wv must match n_v = " << n_v;
		ASSERT_EQ(Mat_ww::RowsAtCompileTime, n_w) 		<< "Number of rows of Mat_ww must match n_w = " << n_w;
		ASSERT_EQ(Mat_ww::ColsAtCompileTime, n_w) 		<< "Number of columns of Mat_ww must match n_w = " << n_w;


	}
};

TEST_F(MatrixSizeTest, matrixSizes)
{
	ASSERT_EQ(A.cols(), x.rows()) << "Number of columns of A must match number of rows of x";
	ASSERT_EQ(B.cols(), u.rows()) << "Number of columns of B must match number of rows of u";
	ASSERT_EQ(G.cols(), v.rows()) << "Number of columns of G must match number of rows of v";

	ASSERT_EQ((A * x).size(), (x).size()) << "Size of Ax must equal x";
	ASSERT_EQ((B * u).size(), (x).size()) << "Size of Bu must equal x";
	ASSERT_EQ((G * v).size(), (x).size()) << "Size of Gv must equal x";

	ASSERT_EQ((C * x).size(), (y).size()) << "Size of Cx must equal y";
	ASSERT_EQ((D * u).size(), (y).size()) << "Size of Du must equal y";
	ASSERT_EQ((H * w).size(), (y).size()) << "Size of Hw must equal y";
}

TEST_F(MatrixSizeTest, modelSizes)
{
	// Test different combination of sizes. Can't use for loop because of compile time constants
	testSizes<1, 1, 1, 1, 1>();
	testSizes<1, 1, 1, 1, 2>();
	testSizes<1, 1, 1, 2, 1>();
	testSizes<1, 1, 1, 2, 2>();
	testSizes<1, 1, 2, 1, 1>();
	testSizes<1, 1, 2, 1, 2>();
	testSizes<1, 1, 2, 2, 1>();
	testSizes<1, 1, 2, 2, 2>();
	testSizes<1, 2, 1, 1, 1>();
	testSizes<1, 2, 1, 1, 2>();
	testSizes<1, 2, 1, 2, 1>();
	testSizes<1, 2, 1, 2, 2>();
	testSizes<1, 2, 2, 1, 1>();
	testSizes<1, 2, 2, 1, 2>();
	testSizes<1, 2, 2, 2, 1>();
	testSizes<1, 2, 2, 2, 2>();
	testSizes<2, 1, 1, 1, 1>();
	testSizes<2, 1, 1, 1, 2>();
	testSizes<2, 1, 1, 2, 1>();
	testSizes<2, 1, 1, 2, 2>();
	testSizes<2, 1, 2, 1, 1>();
	testSizes<2, 1, 2, 1, 2>();
	testSizes<2, 1, 2, 2, 1>();
	testSizes<2, 1, 2, 2, 2>();
	testSizes<2, 2, 1, 1, 1>();
	testSizes<2, 2, 1, 1, 2>();
	testSizes<2, 2, 1, 2, 1>();
	testSizes<2, 2, 1, 2, 2>();
	testSizes<2, 2, 2, 1, 1>();
	testSizes<2, 2, 2, 1, 2>();
	testSizes<2, 2, 2, 2, 1>();
	testSizes<2, 2, 2, 2, 2>();
}