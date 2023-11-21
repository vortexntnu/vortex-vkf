#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

class SimpleDynamicModel : public vortex::models::interface::DynamicModelCTLTV<2> {
public:
    using BaseI = vortex::models::interface::DynamicModelI<2>;
    using typename BaseI::Vec_x;
    using typename BaseI::Mat_xx;
    constexpr static int N_DIM_x = BaseI::N_DIM_x;


    Mat_xx A_c(const Vec_x& = Vec_x::Zero()) const override
    {
        return - Mat_xx::Identity();
    }

    Mat_vv Q_c(const Vec_x& = Vec_x::Zero()) const override
    {
        return Mat_xx::Identity();
    }
};



class NonlinearModel1 : public vortex::models::interface::DynamicModelI<1,1,1> {
public:
    using typename DynamicModelI<1,1,1>::Vec_x;
    using typename DynamicModelI<1,1,1>::Mat_xx;
    using typename DynamicModelI<1,1,1>::Mat_xv;
    using typename DynamicModelI<1,1,1>::Vec_v;

    NonlinearModel1(double std_dev) : cov_(std_dev*std_dev) {}

    Vec_x f_d(double, const Vec_x& x, const Vec_u& = Vec_u::Zero(), const Vec_v& v = Vec_v::Zero()) const override
    {
        Vec_x x_next;
        x_next << std::sin(x(0)) + v(0);
        return x_next;
    }

    Mat_vv Q_d(double = 0.0, const Vec_x& = Vec_x::Zero()) const override
    {
        return Mat_xx::Identity()*cov_;
    }
private:
    const double cov_;

};