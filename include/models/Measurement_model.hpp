#pragma once
#include <eigen3/Eigen/Eigen>
#include <chrono>

namespace Model {

template<size_t n> class Measurement_model
{
public:
    using Vec = Eigen::Matrix<double,n,1>;
    using Mat = Eigen::Matrix<double,n,n>;
    /**
     * @brief Parent class for dynamic models
     */
    Measurement_model() {}

    /**
     * @brief Discrete prediction equation f:
     * Calculate the zero noise prediction at time \p Ts from \p x.
     * @param x State
     * @param Ts Time-step
     * @return The next state x_(k+1) = F x_k
     */
    virtual Vec f(std::chrono::milliseconds Ts, Vec x) const = 0;

    /**
     * @brief Covariance matrix of model:
     * Calculate the transition covariance \p Q for time \p Ts 
     * @param x State
     * @param Ts Time-step
     * @return Measuerement noise covariance matrix Q
     */
    virtual Mat R(std::chrono::milliseconds Ts, Vec x) const = 0;
};

} // End namespace Model