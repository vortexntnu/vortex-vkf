#pragma once
#include <models/LTI_model.hpp>
#include <kalman_filters/Kalman_filter_base.hpp>

namespace Filters {
using namespace Models;

class KF : public Kalman_filter_base {
public:
	KF(LTI_model *lti_model, State x0, Mat P0) : Kalman_filter_base(x0, P0), model{lti_model} {}
	virtual State next_state(Timestep Ts, Measurement y, Input u, Disturbance v, Noise w) override
    {
        
    }
protected:
    LTI_model* model;
};
}