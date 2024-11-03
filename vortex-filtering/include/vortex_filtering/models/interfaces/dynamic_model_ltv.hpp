#pragma once
#include "dynamic_model.hpp"

namespace vortex::model::interface {

/**
 * @brief Linear Time Varying (LTV) Dynamic Model Interface on Lie Groups.
 *        This class defines a dynamic model on Lie groups with a linear time-varying (LTV) structure.
 *        The model is defined by the dynamics equation: [x_{k+1} = f(dt, x, u, v) = Exp(A * log(x) + B * log(u) + G *
 * log(v))]
 *
 * @tparam Type_x Lie group type of the state
 * @tparam Type_u Lie group type of the input
 * @tparam Type_v Lie group type of the process noise
 */
template <typename Type_x, typename Type_u, typename Type_v>
class DynamicModelLTV : public DynamicModel<Type_x, Type_u, Type_v> {
public:
  using Mx = Type_x; ///< Lie group type of the state
  using Mu = Type_u; ///< Lie group type of the input
  using Mv = Type_v; ///< Lie group type of the process noise

  using Tx = typename Mx::Tangent; ///< Tangent space of the state Lie group
  using Tu = typename Mu::Tangent; ///< Tangent space of the state Lie group
  using Tv = typename Mv::Tangent; ///< Tangent space of the state Lie group

  using T = Types_xuv<Mx::DoF, Mu::DoF, Mv::DoF>; ///< Type aliasing for dimensions

  using Gauss_x = vortex::prob::LieGroupGauss<Mx>; ///< Gaussian distribution on the state Lie group

  /**
   * @brief Computes the next state on the Lie group given the current state, input, and process noise.
   *        The dynamics are computed in the tangent space using Jacobians and mapped back to the Lie group.
   *
   * @param dt Time step
   * @param x Current state (Lie group element)
   * @param u Input (Lie group element)
   * @param v Process noise (Lie group element)
   * @return Mx Next state on the Lie group
   */
  virtual Mx f(double dt, const Mx &x, const Mu &u, const Mv &v) const override
  {
    typename T::Mat_xx A = J_f_x(dt, x); // Jacobian with respect to state
    typename T::Mat_xu B = J_f_u(dt, x); // Jacobian with respect to input
    typename T::Mat_xv G = J_f_v(dt, x); // Jacobian with respect to process noise

    typename T::Vec_x x_log = x.log().coeffs();
    typename T::Vec_u u_log = u.log().coeffs();
    typename T::Vec_v v_log = v.log().coeffs();

    Tx x_next = A * x_log + B * u_log + G * v_log;
    return x_next.exp();
  }

  /**
   * @brief Jacobian of the discrete-time dynamics with respect to the state.
   * @param dt Time step
   * @param x Current state (Lie group element)
   * @return T::Mat_xx Jacobian matrix in tangent space of the state
   */
  virtual T::Mat_xx J_f_x(double dt, const Mx &x) const = 0;

  /**
   * @brief Jacobian of the discrete-time dynamics with respect to the input.
   *        By default, returns a zero matrix and can be overridden in derived classes.
   *
   * @param dt Time step
   * @param x Current state (Lie group element)
   * @return T::Mat_xu Jacobian matrix in tangent space of the input
   */
  virtual T::Mat_xu J_f_u(double /*dt*/, const Mx & /*x*/) const { return T::Mat_xu::Zero(); }

  /**
   * @brief Jacobian of the discrete-time dynamics with respect to the process noise.
   *        By default, returns an identity matrix and can be overridden in derived classes.
   *
   * @param dt Time step
   * @param x Current state (Lie group element)
   * @return T::Mat_xv Jacobian matrix in tangent space of the process noise
   */
  virtual T::Mat_xv J_f_v(double /*dt*/, const Mx & /*x*/) const { return T::Mat_xv::Identity(); }

  /**
   * @brief Computes the process noise covariance matrix in the tangent space of the process noise.
   * 
   * @param dt 
   * @param x 
   * @return T::Mat_xvv
   */
  virtual T::Mat_xv Q_v(double dt, const Mx &x) const = 0;

  /**
   * @brief Computes the process noise covariance matrix in the tangent space of the state.
   * 
   * @param dt
   * @param x
   * @return T::Mat_xx
   */
  virtual T::Mat_xx Q_x(double dt, const Mx &x) const
  {
    typename T::Mat_xv Q = Q_v(dt, x);
    typename T::Mat_xv G = J_f_v(dt, x);
    return G * Q * G.transpose();
  }

  /**
   * @brief Predicts the next state distribution given a Gaussian state estimate and input.
   *        This function propagates the state estimate using the Jacobians and noise covariance.
   *
   * @param dt Time step
   * @param x_est Gaussian distribution of the state estimate
   * @param u Input (Lie group element)
   * @return Gauss_x Predicted Gaussian distribution for the next state
   */
  Gauss_x pred_from_est(double dt, const Gauss_x &x_est, const Mu &u) const
  {
    auto mean            = x_est.mean();
    typename T::Mat_xx P = x_est.cov();
    typename T::Mat_xx A = J_f_x(dt, mean);
    typename T::Mat_xv G = J_f_v(dt, mean);
    typename T::Mat_xx Q = G * this->Q(dt, mean) * G.transpose();
    return {f(dt, mean, u, Mv::Identity()), A * P * A.transpose() + Q};
  }

  /**
   * @brief Predicts the next state distribution given the current state and input, without a full covariance.
   *        This function propagates the state based on the process noise covariance and Jacobian.
   *
   * @param dt Time step
   * @param x Current state (Lie group element)
   * @param u Input (Lie group element)
   * @return Gauss_x Predicted Gaussian distribution for the next state
   */
  Gauss_x pred_from_state(double dt, const Mx &x, const Mu &u) const
  {
    typename T::Mat_xx Q_v = this->Q_d(dt, x);
    typename T::Mat_xv G   = J_f_v(dt, x);
    typename T::Mat_xx Q_x = G * Q_v * G.transpose();
    return {f(dt, x, u, Mv::Identity()), Q_x};
  }
};

template <std::size_t DIM_x, std::size_t DIM_u, std::size_t DIM_v>
using DynamicModelLTVR = DynamicModelLTV<manif::Rn<double, DIM_x>, manif::Rn<double, DIM_u>, manif::Rn<double, DIM_v>>;

} // namespace vortex::model::interface
