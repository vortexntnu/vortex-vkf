#include <cmath>
#include <numbers>
#include <limits>

using std::numbers::pi;

constexpr double gamma_function(double x) {
    if (x == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    if (x < 0.0) {
        return pi / (std::sin(pi * x) * gamma_function(1 - x));
    }

    double result = 1.0;
    while (x < 1.0) {
        result /= x;
        x += 1.0;
    }
    
    constexpr double p[] = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };
    
    double y = x;
    double tmp = x + 5.5;
    tmp -= (x + 0.5) * std::log(tmp);
    double ser = 1.000000000190015;
    for (int i = 0; i < 9; ++i) {
        ser += p[i] / ++y;
    }
    
    return std::exp(-tmp + std::log(2.5066282746310005 * ser / x));
}