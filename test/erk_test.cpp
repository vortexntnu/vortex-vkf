#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <memory>
#include <iostream>
#include <functional>

#include <models/model_definitions.hpp>
#include <integration_methods/ERK_methods.hpp>

using namespace Models;
using namespace Integrator;


constexpr int n_x = 2, n_y = 1, n_u = 1;
DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)
using Function_f = std::function<State(Time, State, Input, Disturbance)>;

// Make functions to test the RK methods
State sin_func(Time t, State x, Input u, Disturbance v) 
{
    (void)u;
    (void)v;
    State x_dot;
    x_dot << std::pow(std::sin(t/1s),2)*x(0), 0;
    return x_dot;
}
double sin_func_exact(double x_0, Time t) 
{
    double t_s = t/1s;
    return x_0*exp(0.5 * (t_s - sin(t_s) * cos(t_s)));
}


// van der Pol oscillator
State vdp(Time t, State x, Input u, Disturbance v) 
{
    (void)t;
    (void)v;
    State x_dot;
    x_dot << x(1), (1-std::pow(x(0),2))*x(1) - x(0) + u(0);
    return x_dot;
}


class ERKvdpTest : public ::testing::Test {
protected:
    ERKvdpTest() 
    {
        dt = 0.1ms;
        t_0 = 0s;
        x_0.setZero();
        t.push_back(t_0);
        x.push_back(x_0);
    }
    Timestep dt;
    Time t_0;
    State x_0;

    std::vector<Time> t;
    std::vector<State> x;
};

class ERKTest : public ::testing::Test {
protected:
    ERKTest() 
    {
        dt = 1ms;
        init(0s, State::Zero());
        u.setZero();
        v.setZero();
    }
    Timestep dt;

    Input u;
    Disturbance v;

    std::vector<Time> t;
    std::vector<State> x;



    void init(Time t_0, State x_0) 
    {   
        t.clear();
        x.clear();
        t.push_back(t_0);
        x.push_back(x_0);
    }

    void runIterations(std::shared_ptr<RK_method<n_x,n_u,n_x>> rk_method, Function_f f, double exact, size_t num_iters, double tolerance) 
    {
        for (size_t i = 0; i < num_iters; i++) 
        {
            t.push_back(t.back() + dt);
            x.push_back(rk_method->integrate(f, dt, t.back(), x.back(), u, v));
        }
        std::cout << "Error: " << x.back()(0) - exact << std::endl;
        EXPECT_NEAR(x.back()(0), exact, tolerance);
    }
};

TEST_F(ERKTest, RK4sinFunc) 
{
    State x0;
    x0 << 1, 0;
    init(0s, x0);
    auto rk4 = std::make_shared<RK4_method<n_x,n_u,n_x>>();
    size_t n{5000};

    // Expected error is O(dt^4)
    runIterations(rk4, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-4);
}

TEST_F(ERKTest, EulerSinFunc) 
{
    State x0;
    x0 << 1, 0;
    init(0s, x0);
    auto euler = std::make_shared<Forward_Euler<n_x,n_u,n_x>>();

    size_t n{5000};

    // Expected error is O(dt)
    runIterations(euler, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-2);

}

TEST_F(ERKTest, HeunSinFunc) 
{
    State x0;
    x0 << 1, 0;
    init(0s, x0);
    auto heun = std::make_shared<Heun<n_x,n_u,n_x>>();

    size_t n{5000};

    // Expected error is O(dt^2)
    runIterations(heun, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-2);
}

TEST_F(ERKTest, MidpointSinFunc) 
{
    State x0;
    x0 << 1, 0;
    init(0s, x0);
    auto midpoint = std::make_shared<Midpoint<n_x,n_u,n_x>>();

    size_t n{5000};

    // Expected error is O(dt^2)
    runIterations(midpoint, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-2);
}