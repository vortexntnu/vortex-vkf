#include <gtest/gtest.h>
#include <models/LTI_model.hpp>
using namespace Models;
using Eigen::Matrix;
using Eigen::Vector;

class LTImodelTest : public ::testing::Test {
protected:
    static constexpr int n_x{3}, n_y{1}, n_u{2}, n_v{n_x}, n_w{n_y};
	LTImodelTest()
	{
        Matrix<double,n_x,n_x> A;
        Matrix<double,n_x,n_u> B;
        Matrix<double,n_y,n_x> C;
        Matrix<double,n_y,n_u> D;
        Matrix<double,n_v,n_v> Q;
        Matrix<double,n_w,n_w> R;

        A << .5, 0, 0,
              0, 1, 0,
              0, 0, 0;
        B << 1, 0, 
             0, 0, 
             0, 0;
        C << 1, 0, 0;	
        Q << 1, 0, 0,
             0, 2, 0,
             0, 0, 3;
        R << 1;

        model = Models::LTI_model<n_x,n_y,n_u,n_v,n_w>{A,B,C,Q,R};
	}

    Models::LTI_model<n_x,n_y,n_u,n_v,n_w> model;
};

TEST_F(LTImodelTest, f)
{
    Vector<double,n_x> x0;
    Vector<double,n_x> x1;
    Vector<double,n_u> u0;
    x0 << 1,
          0,
          0;
    x1 << 1.5,
            0,
            0;
    u0 << 1,
          0;

    ASSERT_EQ(x1, model.f(1ms,x0,u0));
}

TEST_F(LTImodelTest, h)
{
    Vector<double,n_x> x0;
    Vector<double,n_y> y0;

    x0 << 1,
          0,
          0;
    y0 << 1;

    ASSERT_EQ(y0, model.h(1ms,x0));
}
