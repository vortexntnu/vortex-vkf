#pragma once
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace models {

template<int n_dim_x, int n_dim_z>
class SimpleSensorModel : public vortex::models::SensorModelEKFI<n_dim_x, n_dim_z> {
public:
    using SensModI = vortex::models::SensorModelEKFI<n_dim_x, n_dim_z>;

    using typename SensModI::Vec_z;
    using typename SensModI::Vec_x;
    using typename SensModI::Mat_xx;
    using typename SensModI::Mat_zx;
    using typename SensModI::Mat_zz;
    using SensModI::N_DIM_x;
    using SensModI::N_DIM_z;

    SimpleSensorModel(double std) : std_(std) {}
    Vec_z h(const Vec_x& x) const override
    {
        return H(x)*x;
    }

    Mat_zx H(const Vec_x&) const override
    {
        return Mat_zx::Identity();
    }
    Mat_zz R(const Vec_x&) const override
    {
        return Mat_zz::Identity()*std_*std_;
    }

private:
    const double std_;
};

} // namespace models
} // namespace vortex