#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>

class SimpleDynamicModel : public vortex::models::DynamicModelEKFI<2> {
public:
    using typename DynamicModelEKFI<2>::Vec_x;
    using typename DynamicModelEKFI<2>::Mat_xx;
    using DynamicModelEKFI<2>::N_DIM_x;

    // A stable state transition 
    Vec_x f_c(const Vec_x &x) const override
    {
        return - x;
    }

    Mat_xx A_c(const Vec_x& = Vec_x::Zero()) const override
    {
        return - Mat_xx::Identity();
    }

    Mat_xx Q_c(const Vec_x& = Vec_x::Zero()) const override
    {
        return Mat_xx::Identity();
    }
};


class VariableLengthSensorModel : public vortex::models::SensorModelEKFI<2, Eigen::Dynamic> {
public:
    using typename SensorModelEKFI<2, Eigen::Dynamic>::Vec_z;
    using typename SensorModelEKFI<2, Eigen::Dynamic>::Vec_x;
    using typename SensorModelEKFI<2, Eigen::Dynamic>::Mat_xx;
    using typename SensorModelEKFI<2, Eigen::Dynamic>::Mat_zx;
    using typename SensorModelEKFI<2, Eigen::Dynamic>::Mat_zz;
    using SensorModelEKFI::N_DIM_z;
    using SensorModelEKFI::N_DIM_x;


    VariableLengthSensorModel(int n_z) : N_z(n_z) {}

    Vec_z h(const Vec_x& x) const override
    {
        return H(x)*x;
    }

    Mat_zx H(const Vec_x&) const override
    {
        return Mat_zx::Identity(N_DIM_x, N_DIM_x);
    }
    Mat_zz R(const Vec_x&) const override
    {
        return Mat_zz::Identity(N_DIM_x, N_DIM_x)*0.1;
    }

    const int N_z;
};

class NonlinearModel1 : public vortex::models::DynamicModelI<1,1,1> {
public:
    using typename DynamicModelI<1,1,1>::Vec_x;
    using typename DynamicModelI<1,1,1>::Mat_xx;
    using typename DynamicModelI<1,1,1>::Mat_xv;
    using typename DynamicModelI<1,1,1>::Vec_v;

    NonlinearModel1(double cov) : cov_(cov) {}

    Vec_x f_d(const Vec_x& x, const Vec_u& = Vec_u::Zero(), const Vec_v& v = Vec_v::Zero(), double = 0.0) const override
    {
        Vec_x x_next;
        x_next << std::sin(x(0)) + v(0);
        return x_next;
    }

    Mat_xx Q_d(const Vec_x& = Vec_x::Zero(), double = 0.0) const override
    {
        return Mat_xx::Identity()*cov_;
    }
private:
    const double cov_;

};