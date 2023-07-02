#include <nodes/kf_node.hpp>


#include <kalman_filters/KF.hpp>

#include <iostream>
int main()
{
    Eigen::Matrix<double,3,3> A;
    Eigen::Vector<double,3> x;
    Eigen::Matrix<double,3,2> a;
    x << 0,
         1,
         2;
    a << 1,2,
         3,4,
         5,6;

    A << x,x,x;
    std::cout << A << "\n\n";

    A << x,a;
    std::cout << A << "\n\n";
}