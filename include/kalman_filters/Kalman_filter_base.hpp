#pragma once
#include <models/model_definitions.hpp>

class Kalman_filter_base
{
public:
    Kalman_filter_base(State x0, Mat P0)
    : x{x0} 
    , P_xx{P0}
    {}

    virtual State next_state(Timestep Ts, Measurement y, Input u) = 0;
    void setState(State x_n) {x=x_n;}
    void setCovariance(Mat P_n) {P_xx=P_n;}
protected:
    State x;
    Mat P_xx;
};