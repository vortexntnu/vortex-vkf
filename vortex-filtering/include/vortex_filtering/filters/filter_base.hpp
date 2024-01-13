#pragma once
#include <Eigen/Dense>
#include <memory>
#include <tuple>

#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

namespace vortex {
namespace filter {
namespace interface {

/** @brief Interface for Kalman filters with dynamic size dimensions.
 * The purpose of this class is to provide a common interface for Kalman filters of any dimension so that they can be used interchangeably.
 * This class is not meant to be inherited from. Use KalmanFilterI instead.
 */
class KalmanFilterX {
public:
  // Using dynamic Eigen types
  using VecX        = Eigen::VectorXd;
  using MatXX       = Eigen::MatrixXd;
  using GaussX      = prob::MultiVarGauss<Eigen::Dynamic>;
  using DynModX     = models::interface::DynamicModelX;
  using SensModX    = models::interface::SensorModelX;
  using DynModXPtr  = std::shared_ptr<DynModX>;
  using SensModXPtr = std::shared_ptr<SensModX>;

  /** @brief Dynamic size dimension Kalman filter constructor.
   * @param dim_x State dimension.
   * @param dim_z Measurement dimension.
   * @param dim_u Input dimension.
   * @param dim_v Process noise dimension.
   * @param dim_w Measurement noise dimension.
   */
  KalmanFilterX(int dim_x, int dim_z, int dim_u, int dim_v, int dim_w) : dim_x_(dim_x), dim_z_(dim_z), dim_u_(dim_u), dim_v_(dim_v), dim_w_(dim_w) {}

  virtual ~KalmanFilterX() = default;

  virtual std::pair<GaussX, GaussX> predictX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, double dt, const GaussX &x_est_prev,
                                             const VecX &u) const = 0;

  virtual GaussX updateX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, const GaussX &x_est_pred, const GaussX &z_est_pred,
                         const VecX &z_meas) const = 0;

  virtual std::tuple<GaussX, GaussX, GaussX> stepX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, double dt, const GaussX &x_est_prev,
                                                   const VecX &z_meas, const VecX &u) const = 0;

protected:
  const int dim_x_; // State dimension
  const int dim_z_; // Measurement dimension
  const int dim_u_; // Input dimension
  const int dim_v_; // Process noise dimension
  const int dim_w_; // Measurement noise dimension
};

/** @brief Interface for Kalman filters with static size dimensions.
 * The purpose of this class is to provide a common interface for Kalman filters of the same dimension so that they can be used interchangeably.
 * @tparam n_dim_x State dimension.
 * @tparam n_dim_z Measurement dimension.
 * @tparam n_dim_u Input dimension.
 * @tparam n_dim_v Process noise dimension.
 * @tparam n_dim_w Measurement noise dimension.
 */
template <int n_dim_x, int n_dim_z, int n_dim_u = n_dim_x, int n_dim_v = n_dim_x, int n_dim_w = n_dim_z> class KalmanFilter : public KalmanFilterX {
public:
  static constexpr int N_DIM_x = n_dim_x;
  static constexpr int N_DIM_z = n_dim_z;
  static constexpr int N_DIM_u = n_dim_u;
  static constexpr int N_DIM_v = n_dim_v;
  static constexpr int N_DIM_w = n_dim_w;

  using Vec_x = Eigen::Vector<double, N_DIM_x>;
  using Vec_z = Eigen::Vector<double, N_DIM_z>;
  using Vec_u = Eigen::Vector<double, N_DIM_u>;
  using Vec_v = Eigen::Vector<double, N_DIM_v>;
  using Vec_w = Eigen::Vector<double, N_DIM_w>;

  using Mat_xx = Eigen::Matrix<double, N_DIM_x, N_DIM_x>;
  using Mat_xz = Eigen::Matrix<double, N_DIM_x, N_DIM_z>;
  using Mat_xv = Eigen::Matrix<double, N_DIM_x, N_DIM_v>;
  using Mat_xw = Eigen::Matrix<double, N_DIM_x, N_DIM_w>;

  using Mat_zx = Eigen::Matrix<double, N_DIM_z, N_DIM_x>;
  using Mat_zz = Eigen::Matrix<double, N_DIM_z, N_DIM_z>;
  using Mat_zw = Eigen::Matrix<double, N_DIM_z, N_DIM_w>;

  using Mat_vx = Eigen::Matrix<double, N_DIM_v, N_DIM_x>;
  using Mat_vv = Eigen::Matrix<double, N_DIM_v, N_DIM_v>;
  using Mat_vw = Eigen::Matrix<double, N_DIM_v, N_DIM_w>;

  using Mat_wx = Eigen::Matrix<double, N_DIM_w, N_DIM_x>;
  using Mat_wv = Eigen::Matrix<double, N_DIM_w, N_DIM_v>;
  using Mat_ww = Eigen::Matrix<double, N_DIM_w, N_DIM_w>;

  using Gauss_x = prob::MultiVarGauss<N_DIM_x>;
  using Gauss_z = prob::MultiVarGauss<N_DIM_z>;
  using Gauss_v = prob::MultiVarGauss<N_DIM_v>;
  using Gauss_w = prob::MultiVarGauss<N_DIM_w>;

  using DynModI     = models::interface::DynamicModelI<N_DIM_x, N_DIM_u, N_DIM_v>;
  using SensModI    = models::interface::SensorModelI<N_DIM_x, N_DIM_z, N_DIM_w>;
  using DynModIPtr  = typename DynModI::SharedPtr;
  using SensModIPtr = typename SensModI::SharedPtr;

  /** @brief Static size dimension Kalman filter constructor.
   * @tparam n_dim_x State dimension.
   * @tparam n_dim_z Measurement dimension.
   * @tparam n_dim_u Input dimension.
   * @tparam n_dim_v Process noise dimension.
   * @tparam n_dim_w Measurement noise dimension.
   */
  KalmanFilter() : KalmanFilterX(N_DIM_x, N_DIM_z, N_DIM_u, N_DIM_v, N_DIM_w) {}
  virtual ~KalmanFilter() = default;

  virtual std::pair<Gauss_x, Gauss_z> predict(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x &x_est_prev, const Vec_u &u) const = 0;

  virtual Gauss_x update(DynModIPtr dyn_mod, SensModIPtr sens_mod, const Gauss_x &x_est_pred, const Gauss_z &z_est_pred, const Vec_z &z_meas) const = 0;

  virtual std::tuple<Gauss_x, Gauss_x, Gauss_z> step(DynModIPtr dyn_mod, SensModIPtr sens_mod, double dt, const Gauss_x &x_est_prev, const Vec_z &z_meas,
                                                     const Vec_u &u) const = 0;

  // Override dynamic size functions to use static size functions
protected:
  std::pair<GaussX, GaussX> predictX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, double dt, const GaussX &x_est_prev, const VecX &u) const override
  {
    DynModIPtr dyn_mod_   = std::static_pointer_cast<DynModI>(dyn_mod);
    SensModIPtr sens_mod_ = std::static_pointer_cast<SensModI>(sens_mod);
    Gauss_x x_est_prev_   = x_est_prev;
    Vec_u u_              = u;

    return predict(dyn_mod_, sens_mod_, dt, x_est_prev_, u_);
  }

  GaussX updateX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, const GaussX &x_est_pred, const GaussX &z_est_pred, const VecX &z_meas) const override
  {
    DynModIPtr dyn_mod_   = std::static_pointer_cast<DynModI>(dyn_mod);
    SensModIPtr sens_mod_ = std::static_pointer_cast<SensModI>(sens_mod);
    Gauss_x x_est_pred_   = x_est_pred;
    Gauss_z z_est_pred_   = z_est_pred;
    Vec_z z_meas_         = z_meas;

    return update(dyn_mod_, sens_mod_, x_est_pred_, z_est_pred_, z_meas_);
  }

  std::tuple<GaussX, GaussX, GaussX> stepX(const DynModXPtr &dyn_mod, const SensModXPtr &sens_mod, double dt, const GaussX &x_est_prev, const VecX &z_meas,
                                           const VecX &u) const override
  {
    DynModIPtr dyn_mod_   = std::static_pointer_cast<DynModI>(dyn_mod);
    SensModIPtr sens_mod_ = std::static_pointer_cast<SensModI>(sens_mod);
    Gauss_x x_est_prev_   = x_est_prev;
    Vec_z z_meas_         = z_meas;
    Vec_u u_              = u;

    return step(dyn_mod_, sens_mod_, dt, x_est_prev_, z_meas_, u_);
  }
};

/** @brief Kalman filter with dimensions defined by the dynamic model and sensor model.
 * @tparam DynModT Dynamic model type.
 * @tparam SensModT Sensor model type.
 */
template <typename DynModT, typename SensModT>
using KalmanFilterM = KalmanFilter<DynModT::N_DIM_x, SensModT::N_DIM_z, DynModT::N_DIM_u, DynModT::N_DIM_v, SensModT::N_DIM_w>;

} // namespace interface
} // namespace filter
} // namespace vortex