#pragma once
#include <gtest/gtest.h>
#include <Eigen/Dense>


testing::AssertionResult isApproxEqual(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol)
{
    if (a.size() != b.size()) {
        return testing::AssertionFailure() << "Vectors are not the same size, "
                                           << "a: " << a.transpose()
                                           << ", b: " << b.transpose();
    }
    for (int i = 0; i < a.size(); ++i) {
        if (std::abs(a(i) - b(i)) > tol) {
            return testing::AssertionFailure() << "Vectors are not the same, "
                                               << "a: " << a.transpose()
                                               << ", b: " << b.transpose();
        }
    }
    return testing::AssertionSuccess();
}

testing::AssertionResult isApproxEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return testing::AssertionFailure() << "Matrices are not the same size, "
                                           << "a: " << a
                                           << ", b: " << b;
    }
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j<a.cols(); ++j) {
            if (std::abs(a(i,j) - b(i,j)) > tol) {
                return testing::AssertionFailure() << "Matrices are not the same, "
                                                   << "a: " << a
                                                   << ", b: " << b;
            }
        }
    }
    return testing::AssertionSuccess();
}