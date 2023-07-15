#include <gtest/gtest.h>
#include <models/LTI_model.hpp>
#include <filters/EKF.hpp>
#include <models/model_definitions.hpp>
#include <random>
#include <vector>

using namespace Models;
using namespace Filters;
class EKFTest : public ::testing::Test {
protected:
    static constexpr int n_x{3}, n_y{1}, n_u{2}, n_v{n_x}, n_w{n_y};
    static constexpr size_t num_iterations{1000};
    static constexpr double COV{1};

    DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
    LTI_model<n_x,n_y,n_u>* model;
    EKF<n_x,n_y,n_u,n_v,n_w>* filter;

    EKFTest()
    {
        Mat_xx A;
        Mat_xu B;
        Mat_yx C;
        Mat_vv Q;
        Mat_ww R;
        State x0;
        Mat_xx P0;

        A << .5, 0, 0,
              0,.1, 0,
              0, 0, 0;
        B << 1, 0, 
             0, 0, 
             0, 0;
        C << 1, 0, 0;	
        Q << 1e-4, 0   , 0   ,
             0   , 1e-4, 0   ,
             0   , 0   , 1e-4;
        R << 1;
        x0 << 1,
              0,
              0;
        P0 = Mat_xx::Identity();
              
        model = new LTI_model<n_x,n_y,n_u,n_v,n_w>{A,B,C,Q,R};
        filter = new EKF<n_x,n_y,n_u,n_v,n_w>{model, x0, P0};

        // Generate random measurements
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d{0,COV};
        for (size_t i=0; i<num_iterations; i++)
        {
            Measurement y_i;
            y_i << d(gen);
            y.push_back(y_i);
        }
        // Generete vector of zero inputs
        for (size_t i=0; i<num_iterations; i++)
        {
            Input u_i;
            u_i << 0, 0;
            u.push_back(u_i);
        }
    }
    ~EKFTest()
    {
        delete filter;
        delete model;
    }
    // Vector for storing measurements
    std::vector<Measurement> y;
    // Vector for storing inputs
    std::vector<Input> u;
};

TEST_F(EKFTest, filterConverges)
{
    // Vector for storing state estimates
    std::vector<State> x;
    for (size_t i=0; i<num_iterations; i++)
    {
        x.push_back(filter->iterate(1ms, y[i], u[i]));
    }
    EXPECT_NEAR(0.0, (filter->get_state()).norm(), 1e-3) << "State estimate did not converge to zero.";
}
