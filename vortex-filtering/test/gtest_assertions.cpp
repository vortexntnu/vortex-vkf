#include "gtest_assertions.hpp"

testing::AssertionResult isApproxEqual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return testing::AssertionFailure() << "Matrices are not the same size,\n"
                                           << "a: " << a
                                           << "\n, b: " << b;
    }
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j<a.cols(); ++j) {
            if (std::abs(a(i,j) - b(i,j)) > tol) {
                return testing::AssertionFailure() << "Matrices are not the same,\n"
                                                   << "a: " << a
                                                   << "\n, b: " << b;
            }
        }
    }
    return testing::AssertionSuccess();
}