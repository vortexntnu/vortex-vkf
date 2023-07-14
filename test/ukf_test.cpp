
#include <gtest/gtest.h>
#include <memory>
#include <math.h>

#include <filters/UKF.hpp>
#include <models/Model_base.hpp>
#include <integration_methods/ERK_methods.hpp>

using namespace Filters;
using namespace Models;

constexpr int n_x = 1, n_y = 1, n_u = 1;
DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)

class unlinear_model : public Model_base<n_x, n_y, n_u, 1, 1> {
public:
    unlinear_model(Mat_vv Q, Mat_ww R) : Model_base<n_x, n_y, n_u, 1, 1>(Q, R) {};

    State f(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const override final
    {
        (void)u;
        double seconds = Ts.count();
        State x_next;
        x_next << (seconds*sin(x(0))) + v(0);
        return x_next;
    }

    Measurement h(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const override final
    {
        (void)Ts;
        (void)u;
        Measurement y;
        y << x + w;
        return y;
    }
};

class UKF_test : public ::testing::Test {
protected:
    void SetUp() override
    {
        Ts = 0.1s;
        x0 << 0.0;
        P0 << 1.0;
        Q << 0.1;
        R << 0.1;
        auto model = std::make_shared<unlinear_model>(Q, R);
        ukf = std::make_shared<UKF<n_x, n_y, n_u, 1, 1>>(model, x0, P0);

    }

    void TearDown() override
    {
    }

    Timestep Ts;
    State x0;
    Mat_xx P0;
    Mat_vv Q;
    Mat_ww R;

    std::shared_ptr<UKF<n_x, n_y, n_u, 1, 1>> ukf;
};