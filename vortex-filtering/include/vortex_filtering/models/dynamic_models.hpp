#pragma once
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>

namespace vortex {
namespace models {

template <int n_dim_x>
class IdentityDynamicModel : public DynamicModelCTLTV<n_dim_x> {
public: 
    using Vec_x = Eigen::Vector<double, n_dim_x>;
    using Mat_xx = Eigen::Matrix<double, n_dim_x, n_dim_x>;

    IdentityDynamicModel(double std) : Q_(Mat_xx::Identity()*std*std) {}
    IdentityDynamicModel(Mat_xx Q) : Q_(Q) {}

    Vec_x f_c(const Vec_x& x, const Vec_x& = Vec_x::Zero(), const Vec_x& = Vec_x::Zero()) const override { return x; }
    Mat_xx A_c(const Vec_x& x) const override { return Mat_xx::Identity(); }
    Mat_xx Q_c(const Vec_x& x) const override { return Q_; }

protected:
    Mat_xx Q_;
};


/** @brief Simple dynamic model with constant velocity.
 * x = [x, y, x_dot, y_dot]
 */
class CVModel : public DynamicModelCTLTV<4> {
public:
    using typename DynamicModelCTLTV<4>::Vec_x;
    using typename DynamicModelCTLTV<4>::Mat_xx;

    /**
     * @brief Constant Velocity Model in 2D
     * x = [x, y, x_dot, y_dot]
     * @param std_vel Standard deviation of velocity
     */
    CVModel(double std_vel) : std_vel_(std_vel) {}

    /** Get the continuous-time state transition model.
     * Overriding DynamicModelCTLTV::f_c
     * @param x State
     * @return Vec_x 
     */
    Vec_x f_c(const Vec_x& x, const Vec_x& = Vec_x::Zero(), const Vec_x& = Vec_x::Zero()) const override
    {
        Vec_x x_dot;
        x_dot << x(2), x(3), 0, 0;
        return x_dot;
    }

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * Overriding DynamicModelCTLTV::A_c
     * @param x State
     * @return Mat_xx 
     */
    Mat_xx A_c(const Vec_x& x) const override
    {
        (void)x; // unused
        Mat_xx A;
        A << 0, 0, 1, 0,
             0, 0, 0, 1,
             0, 0, 0, 0,
             0, 0, 0, 0;
        return A;
    }

    /** Get the continuous time process noise covariance matrix.
     * Overriding DynamicModelCTLTV::Q_c
     * @param x State
     * @return Mat_xx Process noise covariance
     */
    Mat_xx Q_c(const Vec_x& x) const override
    {
        (void)x; // unused
        Mat_xx Q;
        Q << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, std_vel_*std_vel_, 0,
             0, 0, 0, std_vel_*std_vel_;
        return Q;
    }
private:
    double std_vel_;
};

class CTModel : public DynamicModelCTLTV<5> {
public:
    using typename DynamicModelCTLTV<5>::Vec_x;
    using typename DynamicModelCTLTV<5>::Mat_xx;

    /**
     * @brief Coordinated Turn Model in 2D
     * @param std_acc Standard deviation of velocity
     * x = [x, y, x_dot, y_dot, omega]
     */
    CTModel(double std_acc, double std_turn) : std_acc_(std_acc), std_turn_(std_turn) {}

    /** Get the continuous-time state transition model.
     * Overriding DynamicModelCTLTV::f_c
     * @param x State
     * @return Vec_x 
     */
    Vec_x f_c(const Vec_x& x, const Vec_x& = Vec_x::Zero(), const Vec_x& = Vec_x::Zero()) const override
    {
        // x_ddot = -v*omega
        // y_ddot = v*omega
        Vec_x x_dot;
        x_dot << x(2), x(3), -x(3)*x(4), x(2)*x(4), 0;
        return x_dot;
    }

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * Overriding DynamicModelCTLTV::A_c
     * @param x State
     * @return Mat_xx 
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

    /** Get the continuous time process noise covariance matrix.
     * Overriding DynamicModelCTLTV::Q_c
     * @param x State
     * @return Mat_xx Process noise covariance
     */
    Mat_xx Q_c(const Vec_x& x) const override
    {
        (void)x; // unused
        Eigen::Vector3d D;
        D << std_acc_*std_acc_, std_acc_*std_acc_, std_turn_*std_turn_;
        Eigen::Matrix<double, 5,3> G;
        // clang-format off
        G << 0, 0, 0,
             0, 0, 0,
             1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        // clang-format on
        Mat_xx Q = G*D.asDiagonal()*G.transpose();
        return Q;
    }

private:
    double std_acc_;
    double std_turn_;
};


class CAModel : public DynamicModelCTLTV<6> {
public:
    using typename DynamicModelCTLTV<6>::Vec_x;
    using typename DynamicModelCTLTV<6>::Mat_xx;

    /**
     * @brief Constant Acceleration Model in 2D
     * @param std_vel Standard deviation of velocity
     * @param std_acc Standard deviation of acceleration
     */
    CAModel(double std_vel, double std_acc) : std_vel_(std_vel), std_acc_(std_acc) {}

    /** Get the continuous-time state transition model.
     * Overriding DynamicModelCTLTV::f_c
     * @param x State
     * @return Vec_x 
     */
    Vec_x f_c(const Vec_x& x, const Vec_x& = Vec_x::Zero(), const Vec_x& = Vec_x::Zero()) const override
    {
        Vec_x x_dot;
        x_dot << x(2), x(3), x(4), x(5), 0, 0;
        return x_dot;
    }

    /** Get the Jacobian of the continuous state transition model with respect to the state.
     * Overriding DynamicModelCTLTV::A_c
     * @param x State
     * @return Mat_xx 
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

    /** Get the continuous time process noise covariance matrix.
     * Overriding DynamicModelCTLTV::Q_c
     * @param x State
     * @return Mat_xx Process noise covariance
     */
    Mat_xx Q_c(const Vec_x& x) const override
    {
        assert(false); // Not implemented
        (void)x; // unused
        Mat_xx Q;
        // clang-format off
        Q << 0;
        return Q;
    }
    
private:
    double std_vel_;
    double std_acc_;
};

}  // namespace models
}  // namespace vortex