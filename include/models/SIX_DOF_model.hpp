#pragma once
#include <models/model_definitions.hpp>
#include <models/Model_base.hpp>
namespace Models {

template<int n_x, int n_y, int n_u, int n_v=n_x, int n_w=n_y>
class SIX_DOF_model : public Model_base<n_x, n_y, n_u, n_v, n_w> {
    DEFINE_MODEL_TYPES(n_x,n_y,n_u,n_v,n_w)
public:
    SIX_DOF_model() : Model_base<n_x,n_y,n_u,n_v,n_w>() {};
    ~SIX_DOF_model() {};

    /**
     * @brief Time update function f
     * 
     * @param Ts Time-step
     * @param x State
     * @param u Input
     * @param v Disturbance
     * @return State update
     */
    State f(Timestep Ts, State x, Input u = Input::Zero(), Disturbance v = Disturbance::Zero()) override final
    {
        (void)Ts;
        (void)x;
        (void)u;
        (void)v;
        State x_dot;
        return x_dot;
    }


};
}