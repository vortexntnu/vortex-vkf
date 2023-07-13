#pragma once
#include <gtest/gtest.h>
#include <math.h>

#include <filters/UKF.hpp>
#include <models/Model_base.hpp>
#include <integration_methods/ERK_methods.hpp>

using namespace Filters;
using namespace Models;

constexpr int n_x = 1, n_y = 1, n_u = 1;
class unlinear_model : public Model_base<n_x, n_y, n_u> {
public:
    DEFINE_MODEL_TYPES(n_x, n_y, n_u, n_x, n_y)
    unlinear_model() : Model_base<n_x, n_y, n_u>() {};

    State f(Timestep Ts, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const override final
    {
        (void)u;
        State x_next;
        x_next << (Ts.s()*x).sin() + v;
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