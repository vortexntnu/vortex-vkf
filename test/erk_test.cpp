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


namespace sin_func_test {
constexpr int n_x = 1, n_y = 1, n_u = 1;
DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)
using State_dot = std::function<State(Time, const State&, const Input&, const Disturbance&)>;

// Make functions to test the RK methods
State sin_func(Time t, const State& x, const Input& u, const Disturbance& v) 
{
    (void)u;
    (void)v;
    State x_dot;
    x_dot << std::pow(std::sin(t/1s),2)*x(0);
    return x_dot;
}
double sin_func_exact(double x_0, Time t) 
{
    double t_s = t/1s;
    return x_0*exp(0.5 * (t_s - sin(t_s) * cos(t_s)));
}






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

    void runIterations(std::shared_ptr<RK_method<n_x,n_u,n_x>> rk_method, State_dot f, double exact, size_t num_iters, double tolerance) 
    {
        for (size_t i = 0; i < num_iters; i++) 
        {
            x.push_back(rk_method->integrate(f, dt, t.back(), x.back(), u, v));
            t.push_back(t.back() + dt);
        }
        std::cout << "Error: " << x.back()(0) - exact << std::endl;
        EXPECT_NEAR(x.back()(0), exact, tolerance);
    }
};

TEST_F(ERKTest, RK4sinFunc) 
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto rk4 = std::make_shared<RK4<n_x,n_u,n_x>>();
    size_t n = 5000;

    // Expected error is O(dt^4)
    runIterations(rk4, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-4);
}

TEST_F(ERKTest, EulerSinFunc) 
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto euler = std::make_shared<Forward_Euler<n_x,n_u,n_x>>();
    size_t n = 5000;

    // Expected error is O(dt)
    runIterations(euler, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-1);

}

TEST_F(ERKTest, HeunSinFunc) 
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto heun = std::make_shared<Heun<n_x,n_u,n_x>>();
    size_t n = 5000;

    // Expected error is O(dt^2)
    runIterations(heun, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-2);
}

TEST_F(ERKTest, MidpointSinFunc) 
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto midpoint = std::make_shared<Midpoint<n_x,n_u,n_x>>();
    size_t n = 5000;

    // Expected error is O(dt^2)
    runIterations(midpoint, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-5);
}


TEST_F(ERKTest, ButcherMidpointSinFunc)
{
    State x0;
    x0 << 1;
    init(0s, x0);
    constexpr int n_stages = 2;
    Eigen::Matrix<double, n_stages, n_stages> A;
    Eigen::Vector<double, n_stages> b;
    Eigen::Vector<double, n_stages> c;

    // Midpoint method
    A << 0, 0,
         1/2.0, 0;

    b << 0, 1;
    c << 0, 1/2.0;

    auto midpoint = std::make_shared<Butcher<n_x,n_u,n_x,n_stages>>(A, b, c);
    size_t n = 5000;

    // Expected error is O(dt^2)
    runIterations(midpoint, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-5);  
}

TEST_F(ERKTest, ButcherRKDPsinFunc)
{
    State x0;
    x0 << 1;
    init(0s, x0);
    constexpr int n_stages = 7;
    Eigen::Matrix<double, n_stages, n_stages> A;
    Eigen::Vector<double, n_stages> b;
    Eigen::Vector<double, n_stages> c;

    // Dormand Prince (RKDP) method
    A << 0           ,  0           , 0           ,  0        ,  0           , 0      , 0,
         1/5.0       ,  0           , 0           ,  0        ,  0           , 0      , 0,
         3/40.0      ,  9/40.0      , 0           ,  0        ,  0           , 0      , 0,
         44/45.0     , -56/15.0     , 32/9.0      ,  0        ,  0           , 0      , 0,
         19372/6561.0, -25360/2187.0, 64448/6561.0, -212/729.0,  0           , 0      , 0,
         9017/3168.0 , -355/33.0    , 46732/5247.0,  49/176.0 , -5103/18656.0, 0      , 0,
         35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0 , 11/84.0, 0;

    b << 35/384.0    ,  0           , 500/1113.0  ,  125/192.0, -2187/6784.0 , 11/84.0, 0;
    c << 0           ,  1/5.0       , 3/10.0      ,  4/5.0    ,  8/9.0       , 1      , 1;

    auto rkdp = std::make_shared<Butcher<n_x,n_u,n_x,n_stages>>(A, b, c);
    size_t n = 5000;

    // Expected error is O(dt^5)
    runIterations(rkdp, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-8);  
}

TEST_F(ERKTest, ODE45sinFuncRealTime)
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto rk45 = std::make_shared<ODE45<n_x,n_u,n_x>>(1e-5, 1000000, 1ns);

    size_t n = 5000;

    // Expected error is O(dt^5)
    runIterations(rk45, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-4);  
}

TEST_F(ERKTest, ODE45sinFuncLongStep)
{
    State x0;
    x0 << 1;
    init(0s, x0);
    auto rk45 = std::make_shared<ODE45<n_x,n_u,n_x>>(1e-6, 1000000, 1ns);
    dt = 5s;
    size_t n = 1;

    // Expected error is O(dt^5)
    runIterations(rk45, sin_func, sin_func_exact(x0(0), dt*n), n, 1e-4);
}


////////////////////////////////////////
} // namespace sin_func_test
////////////////////////////////////////


namespace van_der_pol_test {


constexpr int n_x = 2, n_y = 1, n_u = 1;
DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)

// van der Pol oscillator
State vdp(Time t, const State& x, const Input& u, const Disturbance& v) 
{
    (void)t;
    (void)v;
    State x_dot;
    x_dot << x(1), (1-std::pow(x(0),2))*x(1) - x(0) + u(0);
    return x_dot;
}


////////////////////////////////////////
} // namespace van_der_pol_test
////////////////////////////////////////
