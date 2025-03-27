#pragma once
#include <gtest/gtest.h>
#include <Eigen/Dense>

testing::AssertionResult isApproxEqual(const Eigen::MatrixXd& a,
                                       const Eigen::MatrixXd& b,
                                       double tol);
