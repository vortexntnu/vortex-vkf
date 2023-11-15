#pragma once
#include <Eigen/Dense>
#include <tuple>

#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/models/dynamic_model.hpp>
#include <vortex_filtering/models/sensor_model.hpp>

namespace vortex {
namespace filters {

/** @brief Interface for Kalman filters with dynamic size dimensions.
 * The purpose of this class is to provide a common interface for Kalman filters of any dimension so that they can be used in the same way.
 * This class is not meant to be inherited from. Use KalmanFilterI instead. 
 */
class KalmanFilterX {
public:
	// Using dynamic Eigen types
	using VecX = Eigen::VectorXd;
	using MatXX = Eigen::MatrixXd;
	using GaussX = prob::MultiVarGauss<Eigen::Dynamic>;
	using DynModX = models::DynamicModelX;
	using SensModX = models::SensorModelX;
	using DynModXPtr = std::shared_ptr<DynModX>;
	using SensModXPtr = std::shared_ptr<SensModX>;

	// Constructor to initialize the dimensions
	KalmanFilterX(int dim_x, int dim_z, int dim_u, int dim_v, int dim_w)
		: dim_x_(dim_x), dim_z_(dim_z), dim_u_(dim_u), dim_v_(dim_v), dim_w_(dim_w) {}

	virtual ~KalmanFilterX() = default;

	virtual std::pair<GaussX, GaussX> predict(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_prev, const VecX& u, double dt) const = 0;

	virtual GaussX update(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_pred, const GaussX& z_est_pred, const VecX& z_meas) const = 0;

	virtual std::tuple<GaussX, GaussX, GaussX> step(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_prev, const VecX& z_meas, const VecX& u, double dt) const = 0;

protected:
	const int dim_x_;  // State dimension
	const int dim_z_;  // Measurement dimension
	const int dim_u_;  // Input dimension
	const int dim_v_;  // Process noise dimension
	const int dim_w_;  // Measurement noise dimension
};

/** @brief Interface for Kalman filters.
 * 
 * @tparam DynModT Dynamic model type.
 * @tparam SensModT Sensor model type. 
 */
template<typename DynModT, typename SensModT>
class KalmanFilterI : public KalmanFilterX {
public:
	static constexpr int N_DIM_x = DynModT::N_DIM_x;
	static constexpr int N_DIM_u = DynModT::N_DIM_u;
	static constexpr int N_DIM_z = SensModT::N_DIM_z;
	static constexpr int N_DIM_v = DynModT::N_DIM_v;
	static constexpr int N_DIM_w = SensModT::N_DIM_w;

	using DynModI  = models::DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;
    using SensModI = models::SensorModelI<N_DIM_x, N_DIM_z, N_DIM_w>;
    using DynModIShared = std::shared_ptr<DynModI>;
    using SensModIShared = std::shared_ptr<SensModI>;


	using Vec_z = Eigen::Vector<double, N_DIM_z>;
	using Vec_u = Eigen::Vector<double, N_DIM_u>;
	using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
	using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
	
	KalmanFilterI() : KalmanFilterX(N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w) {}
	virtual ~KalmanFilterI() = default;

	virtual std::pair<Gauss_x, Gauss_z> predict(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_u& u, double dt) = 0;

	virtual Gauss_x update(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_pred, const Gauss_z& z_est_pred, const Vec_z& z_meas) = 0;

	virtual std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIShared dyn_mod, SensModIShared sens_mod, const Gauss_x& x_est_prev, const Vec_z& z_meas, const Vec_u& u, double dt) = 0;

	// Override dynamic size functions to use static size functions
protected:
	std::pair<GaussX, GaussX> predict(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_prev, const VecX& u, double dt) const override {
		// cast to dynamic model type to access pred_from_est
		auto dyn_model = std::dynamic_pointer_cast<DynModT>(dyn_mod);
		// cast to sensor model type to access pred_from_est
		auto sens_model = std::dynamic_pointer_cast<SensModT>(sens_mod);

		Gauss_x x_est_pred = dyn_model->pred_from_est(x_est_prev, u, dt);
		Gauss_z z_est_pred = sens_model->pred_from_est(x_est_pred);
		return {x_est_pred, z_est_pred};
	}

	GaussX update(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_pred, const GaussX& z_est_pred, const VecX& z_meas) const override {
		// cast to sensor model type
		auto sens_model = std::dynamic_pointer_cast<SensModT>(sens_mod);

		Gauss_z z_meas_gauss = {z_meas, sens_model->R_dX(x_est_pred.mean())};
		return z_est_pred.update(z_meas_gauss);
	}

	std::tuple<GaussX, GaussX, GaussX> step(const DynModXPtr& dyn_mod, const SensModXPtr& sens_mod, const GaussX& x_est_prev, const VecX& z_meas, const VecX& u, double dt) const override {
		// cast to dynamic model type to access pred_from_est
		auto dyn_model = std::dynamic_pointer_cast<DynModT>(dyn_mod);
		// cast to sensor model type to access pred_from_est
		auto sens_model = std::dynamic_pointer_cast<SensModT>(sens_mod);

		Gauss_x x_est_pred = dyn_model->pred_from_est(x_est_prev, u, dt);
		Gauss_z z_est_pred = sens_model->pred_from_est(x_est_pred);
		Gauss_z z_meas_gauss = {z_meas, sens_model->R_dX(x_est_pred.mean())};
		Gauss_x x_est = z_est
};

}  // namespace filters
}  // namespace vortex