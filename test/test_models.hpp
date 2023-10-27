#pragma once
#include <models/dynamic_model.hpp>
#include <models/sensor_model.hpp>

class SimpleDynamicModel : public vortex::models::DynamicModel<2> {
public:
    using typename DynamicModel<2>::State;
    using typename DynamicModel<2>::Mat_xx;

    // A stable state transition 
    State f_c(const State &x) const override
    {
        return -x;
    }

    Mat_xx A_c(const State &x) const override
    {
        (void)x; // unused
        return -Mat_xx::Identity();
    }

    Mat_xx Q_c(const State &x) const override
    {
        (void)x; // unused
        return Mat_xx::Identity();
    }
};

class SimpleSensorModel : public vortex::models::SensorModel<2, 2> {
public:
    using typename SensorModel<2, 2>::Measurement;
    using typename SensorModel<2, 2>::State;
    using typename SensorModel<2, 2>::Mat_xx;
    using typename SensorModel<2, 2>::Mat_zx;
    using typename SensorModel<2, 2>::Mat_zz;

    Measurement h(const State& x) const override
    {
        return x;
    }

    Mat_zx H(const State& x) const override
    {
        (void)x; // unused
        return Mat_zx::Identity();
    }
    Mat_zz R(const State& x) const override
    {
        (void)x; // unused
        return Mat_zz::Identity()*0.1;
    }

};