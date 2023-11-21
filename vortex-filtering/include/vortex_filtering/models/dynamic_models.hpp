#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>

namespace vortex {
namespace models {

// Unused
constexpr int X = 1;

template <int n_dim_x>
class IdentityDynamicModel : public interface::DynamicModelCTLTV<n_dim_x> {
public: 
    using BaseI  = interface::DynamicModelI<n_dim_x>;
    using Vec_x  = typename BaseI::Vec_x; 
    using Mat_xx = typename BaseI::Mat_xx;
    using Mat_vv = typename BaseI::Mat_vv;

    IdentityDynamicModel(double std) : Q_(Mat_xx::Identity()*std*std) {}
    IdentityDynamicModel(Mat_vv Q) : Q_(Q) {}

    Vec_x f_c(const Vec_x& x, const Vec_x& = Vec_x::Zero(), const Vec_x& = Vec_x::Zero()) const override { return x; }
    Mat_xx A_c(const Vec_x& x) const override { return Mat_xx::Identity(); }
    Mat_vv Q_c(const Vec_x& x) const override { return Q_; }

protected:
    Mat_vv Q_;
};


/** Simple dynamic model with constant velocity.
 * x = [x, y, x_dot, y_dot]
 */
class CVModel : public interface::DynamicModelCTLTV<4,X,2> {
public:
    using BaseI = interface::DynamicModelI<4,X,2>;
    using typename BaseI::Vec_x;
    using typename BaseI::Mat_xx;
    using typename BaseI::Mat_xv;
    using typename BaseI::Mat_vv;

    /**
     * @brief Constant Velocity Model in 2D
     * x = [x, y, x_dot, y_dot]
     * @param std_vel Standard deviation of velocity
     */
    CVModel(double std_vel) : std_vel_(std_vel) {}

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * @param x State (unused)
     * @return Mat_xx 
     * @note Overriding DynamicModelCTLTV::A_c
     */
    Mat_xx A_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Mat_xx A;
        // clang-format off
        A << 0, 0, 1, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,
             0, 0, 0, 0;
        // clang-format on
        return A;
    }

    /** Get the Jacobian of the continuous state transition model with respect to the process noise.
     * @param x State (unused)
     * @return Mat_xv 
     * @note Overriding DynamicModelCTLTV::G_c
     */
    Mat_xv G_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Mat_xv G;
        // clang-format off
        G << 0, 0,
             0, 0,
             1, 0,
             0, 1;
        // clang-format on
        return G;
    }

    /** Get the continuous time process noise covariance matrix.
     * @param x State (unused)
     * @return Mat_xx Process noise covariance
     * @note Overriding DynamicModelCTLTV::Q_c
     */
    Mat_vv Q_c(const Vec_x& = Vec_x::Zero()) const override
    {
        return Mat_vv::Identity()*std_vel_*std_vel_;
    }

    
private:
    double std_vel_;
};

/** Coordinated Turn Model in 2D.
 * x = [x, y, x_dot, y_dot, omega]
 */
class CTModel : public interface::DynamicModelCTLTV<5,X,3> {
public:
    using BaseI = interface::DynamicModelI<5,X,3>;
    using typename BaseI::Vec_x;
    using typename BaseI::Vec_v;

    using typename BaseI::Mat_xx;
    using typename BaseI::Mat_xv;
    using typename BaseI::Mat_vv;

    /** Coordinated Turn Model in 2D.
     * State = [x, y, x_dot, y_dot, omega]
     * @param std_vel Standard deviation of velocity
     * @param std_turn Standard deviation of turn rate
     */
    CTModel(double std_vel, double std_turn) : std_vel_(std_vel), std_turn_(std_turn) {}

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * @param x State
     * @return Mat_xx 
     * @note Overriding DynamicModelCTLTV::A_c
     */
    Mat_xx A_c(const Vec_x& x) const override
    {
        Mat_xx A;
        // clang-format off
        A << 0, 0, 1   , 0   , 0,
             0, 0, 0   , 1   , 0,
             0, 0, 0   ,-x(4), 0,
             0, 0, x(4), 0   , 0,
             0, 0, 0   , 0   , 0;
        // clang-format on
        return A;
    }

    /** Get the continuous time process noise matrix
     * @param x State (unused)
     * return Mat_xv Process noise matrix
     * @note Overriding DynamicModelCTLTV::G_c
     */
    Mat_xv G_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Mat_xv G;
        // clang-format off
        G << 0, 0, 0,
             0, 0, 0,
             1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        // clang-format on
        return G;
    }

    /** Get the continuous time process noise covariance matrix.
     * @param x State
     * @return Mat_xx Process noise covariance
     * @note Overriding DynamicModelCTLTV::Q_c
     */
    Mat_vv Q_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Vec_v D;
        D << std_vel_*std_vel_, std_vel_*std_vel_, std_turn_*std_turn_;
        return D.asDiagonal();
    }

private:
    double std_vel_;
    double std_turn_;
};

/** Constant Acceleration Model in 2D
 */
class CAModel : public interface::DynamicModelCTLTV<6,1,4> {
public:
    using BaseI = interface::DynamicModelI<6,1,4>;
    using typename BaseI::Vec_x;
    using typename BaseI::Vec_v;

    using typename BaseI::Mat_xx;
    using typename BaseI::Mat_xv;
    using typename BaseI::Mat_vv;

    /** Constant Acceleration Model in 2D
     * @param std_vel Standard deviation of velocity
     * @param std_acc Standard deviation of acceleration
     */
    CAModel(double std_vel, double std_acc) : std_vel_(std_vel), std_acc_(std_acc) {}

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * @param x State
     * @return Mat_xx 
     * @note Overriding DynamicModelCTLTV::A_c
     */
    Mat_xx A_c(const Vec_x& x) const override
    {
        (void)x; // unused
        Mat_xx A;
        // clang-format off
        A << 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0;
        // clang-format on
        return A;
    }

    Mat_xv G_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Mat_xv G;
        // clang-format off
        G << 0, 0, 0, 0,
             0, 0, 0, 0,
             1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
        // clang-format on
        return G;
    }

    /** Get the continuous time process noise covariance matrix.
     * @param x State
     * @return Mat_xx Process noise covariance
     * @note Overriding DynamicModelCTLTV::Q_c
     */
    Mat_vv Q_c(const Vec_x& = Vec_x::Zero()) const override
    {
        Vec_v D;
        D << std_vel_*std_vel_, std_vel_*std_vel_, std_acc_*std_acc_, std_acc_*std_acc_;
        return D.asDiagonal();
    }
    
private:
    double std_vel_;
    double std_acc_;
};

}  // namespace models
}  // namespace vortex