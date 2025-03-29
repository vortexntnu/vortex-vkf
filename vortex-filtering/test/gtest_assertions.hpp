#pragma once
#include <Eigen/Dense>
#include <gtest/gtest.h>

testing::AssertionResult isApproxEqual(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, double tol);
