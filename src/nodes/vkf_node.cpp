#include <iostream>
#include <models/Dynamic_model.hpp>
using namespace std::chrono_literals;


int main()
{
    Vec x0{3};
    x0 << 1, 2, 2;
    Vec q{3};
    q << 4, 6, 2;
    Model::Landmark target{q};
    Vec x1 = target.f(1ms, x0);
    std::cout << x0 << "\n\n";
    std::cout << x1;
}