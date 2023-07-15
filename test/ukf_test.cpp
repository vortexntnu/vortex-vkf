
#include <gtest/gtest.h>
#include <memory>
#include <math.h>
#include <random>
#include <iostream>

#include <filters/UKF.hpp>
#include <models/Model_base.hpp>

using namespace Filters;
using namespace Models;

constexpr int n_x = 1, n_y = 1, n_u = 1;
DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)

class unlinear_model : public Model_base<n_x, n_y, n_u> {
public:
    unlinear_model(Mat_vv Q, Mat_ww R) : Model_base<n_x, n_y, n_u>(Q, R) {};

    State f(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const override final
    {
        (void)u;
        State x_next;
        // x_next << x(0) + .7*x(1) + v(0), v(1);
        x_next << 0.7*x(0) + 4*sin(Ts/1s) + v(0);
        return x_next;
    }

    Measurement h(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const override final
    {
        (void)Ts;
        (void)u;
        Measurement y;
        y << x(0) + w(0);
        return y;
    }
};

class UKF_test : public ::testing::Test {
protected:
    void SetUp() override
    {
        Ts = 10ms;
        x0 = State::Zero();
        P0 = Mat_xx::Identity()*1e-10; // Trust the initial state
        Q  = Mat_vv::Identity()*1e-3;
        R  = Mat_ww::Identity()*1e-5;
        model = std::make_shared<unlinear_model>(Q, R);
        ukf = std::make_shared<UKF<n_x,n_y,n_u>>(model, x0, P0);
    }

    void TearDown() override
    {
    }

    Timestep Ts;
    State x0;
    Mat_xx P0;
    Mat_vv Q;
    Mat_ww R;

    std::shared_ptr<unlinear_model> model;
    std::shared_ptr<UKF<n_x,n_y,n_u>> ukf;
};

TEST_F(UKF_test, test_iterate)
{
    // Genereate random noise vector

    std::random_device rd;                              // random device class instance, source of 'true' randomness for initializing random seed
    std::mt19937 gen(rd());                             // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::normal_distribution<> d_disturbance{0,1e-3};   // instance of class std::normal_distribution with specific mean and stddev
    std::normal_distribution<> d_noise{0,1e-5};         // instance of class std::normal_distribution with specific mean and stddev
    
    // Simulate model with noise
    State x = x0;
    Disturbance v;
    Noise w;
    for (size_t i=0; i<1000; i++)
    {
        v << d_disturbance(gen); // get random number with normal distribution using gen as random source
        // Simulate model
        x = model->f(Ts, x, Input::Zero(), v);
        // Get measurement
        w << d_noise(gen);
        Measurement y = model->h(Ts, x, Input::Zero(), w);
        // Iterate UKF
        ukf->iterate(Ts, y);
    }  
    // Check if the state is close to the real state
    std::cout << "error: " << x(0) - ukf->get_state()(0) << std::endl;
    EXPECT_NEAR(x(0), ukf->get_state()(0), 1e-2);
}