#include <models/Dynamic_model.hpp>
#include <iostream>
using namespace std::chrono_literals;
using stateVec = Eigen::Matrix<double,7,1>;

int main()
{
    stateVec x0{stateVec::Zero()};
    Model::Landmark target{};
    stateVec x1 = target.f(1ms, x0);
    std::cout << x0 << '\n';
    std::cout << x1;
}