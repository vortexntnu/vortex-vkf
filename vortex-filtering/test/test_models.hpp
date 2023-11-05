#pragma once
#include <models/dynamic_model.hpp>
#include <models/sensor_model.hpp>

class SimpleDynamicModel : public vortex::models::DynamicModelI<2> {
public:
    using typename DynamicModelI<2>::Vec_x;
    using typename DynamicModelI<2>::Mat_xx;
    using DynamicModelI<2>::N_DIM_x;

    // A stable state transition 
    Vec_x f_c(const Vec_x &x) const override
    {
        return -x;
    }

    Mat_xx A_c(const Vec_x &x) const override
    {
        (void)x; // unused
        return -Mat_xx::Identity();
    }

    Mat_xx Q_c(const Vec_x &x) const override
    {
        (void)x; // unused
        return Mat_xx::Identity();
    }
};

template<int n_dim_x, int n_dim_z>
class SimpleSensorModel : public vortex::models::SensorModelI<n_dim_x, n_dim_z> {
public:
    using SensModI = vortex::models::SensorModelI<n_dim_x, n_dim_z>;

    using typename SensModI::Vec_z;
    using typename SensModI::Vec_x;
    using typename SensModI::Mat_xx;
    using typename SensModI::Mat_zx;
    using typename SensModI::Mat_zz;
    using SensModI::N_DIM_x;
    using SensModI::N_DIM_z;

    Vec_z h(const Vec_x& x) const override
    {
        return H(x)*x;
    }

    Mat_zx H(const Vec_x& x) const override
    {
        (void)x; // unused
        return Mat_zx::Identity();
    }
    Mat_zz R(const Vec_x& x) const override
    {
        (void)x; // unused
        return Mat_zz::Identity()*0.1;
    }

};

class VariableLengthSensorModel : public vortex::models::SensorModelI<2, Eigen::Dynamic> {
public:
    using typename SensorModelI<2, Eigen::Dynamic>::Vec_z;
    using typename SensorModelI<2, Eigen::Dynamic>::Vec_x;
    using typename SensorModelI<2, Eigen::Dynamic>::Mat_xx;
    using typename SensorModelI<2, Eigen::Dynamic>::Mat_zx;
    using typename SensorModelI<2, Eigen::Dynamic>::Mat_zz;
    using SensorModelI::N_DIM_z;
    using SensorModelI::N_DIM_x;


    VariableLengthSensorModel(int n_z) : N_z(n_z) {}

    Vec_z h(const Vec_x& x) const override
    {
        return H(x)*x;
    }

    Mat_zx H(const Vec_x& x) const override
    {
        (void)x; // unused
        return Mat_zx::Identity(N_DIM_x, N_DIM_x);
    }
    Mat_zz R(const Vec_x& x) const override
    {
        (void)x; // unused
        return Mat_zz::Identity(N_DIM_x, N_DIM_x)*0.1;
    }

    const int N_z;
};

template<int n_dim_x>
class FirstStatesMeasuredModel : public vortex::models::SensorModelI<n_dim_x, Eigen::Dynamic> {
public:
    using SensModI = vortex::models::SensorModelI<n_dim_x, Eigen::Dynamic>;
    using SensModI::N_DIM_x;
    using typename SensModI::Vec_z;
    using typename SensModI::Vec_x;
    using typename SensModI::Mat_xx;
    using typename SensModI::Mat_zx;
    using typename SensModI::Mat_zz;

    FirstStatesMeasuredModel(int n_z, double std) : n_z_(n_z), std_(std) {}

    Vec_z h(const Vec_x& x) const override
    {
        return H(x)*x;
    }

    Mat_zx H(const Vec_x& x) const override
    {
        (void)x; // unused
        Mat_zx H = Mat_zx::Zero(n_z_, N_DIM_x);
        H.block(0, 0, n_z_, n_z_) = Mat_zz::Identity(n_z_, n_z_);

        return H;
    }

    Mat_zz R(const Vec_x& x) const override
    {
        (void)x; // unused
        return Mat_zz::Identity(n_z_, n_z_)*std_*std_;
    }

    private:
        int n_z_;
        double std_;
};