#pragma once
#include <Eigen/Dense>
#include <tuple>

#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace filters {

template<typename DynamicModelT, typename SensorModelT>
class KalmanFilterBase {
public:
	static constexpr int N_DIM_x = DynamicModelT::N_DIM_x;
	static constexpr int N_DIM_u = DynamicModelT::N_DIM_u;
	static constexpr int N_DIM_z = SensorModelT::N_DIM_z;
	static constexpr int N_DIM_v = DynamicModelT::N_DIM_v;
	static constexpr int N_DIM_w = SensorModelT::N_DIM_w;

	using DynModI  = models::DynamicModelBaseI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::SensorModelBaseI<N_DIM_x, N_DIM_z, N_DIM_w>;
    using DynModIShared = std::shared_ptr<DynModI>;
    using SensModIShared = std::shared_ptr<SensModI>;


	using Vec_z = Eigen::Vector<double, N_DIM_z>;
	using Vec_u = Eigen::Vector<double, N_DIM_u>;
	using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
	using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
	
	KalmanFilterBase() = default;
	virtual ~KalmanFilterBase() = default;

	virtual std::pair<Gauss_x, Gauss_z> predict(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_u& u, double dt) = 0;

	virtual Gauss_x update(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) = 0;

	virtual std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_u& u, double dt) = 0;

};

}  // namespace filters
}  // namespace vortex