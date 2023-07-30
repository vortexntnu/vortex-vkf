#pragma once

#include <models/model_definitions.hpp>
#include <models/Model_base.hpp>

namespace Models {


class Temp_gyro_model : public Model_base<7,3,3,6,3> {
public:
    DEFINE_MODEL_TYPES(7,3,3,6,3)
    using Quaternion = Eigen::Quaterniond;

    Temp_gyro_model() : Model_base<7,3,3,6,3>() {};

    State f(Time t, const State& x, const Input& u = Input::Zero(), const Disturbance& v = Disturbance::Zero()) const override final
    {
        (void)t;
        (void)u;
        Quaternion q(x(0), x(1), x(2), x(3));
        Quaternion q_dot = q*Quaternion(0, v(0), v(1), v(2));
        q_dot.coeffs() *= 0.5;
        State x_dot;
        x_dot << q_dot.w(), q_dot.x(), q_dot.y(), q_dot.z(), v(3), v(4), v(5);
        
        return x_dot;
    }

    Measurement h(Time t, const State& x, const Input& u = Input::Zero(), const Noise& w = Noise::Zero()) const override final
    {
        (void)t;
        (void)x;
        (void)u;
        // 6 dof imu measurement
        Measurement z;
        z << w(0), w(1), w(2);
        return z;
        return w;
    }


};
} // namespace Models