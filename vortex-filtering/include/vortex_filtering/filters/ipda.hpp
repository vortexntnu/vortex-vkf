#ifndef IPDA_HPP
#define IPDA_HPP

#include <vector>
#include <Eigen/Dense>

class IPDAFilter {
public:
    IPDAFilter();
    double generate_new_probability(std::vector<Eigen::Vector2d> &measurements, double &probability_of_survival, float &probability_of_detection, double &standard_deviation, float &lambda);
private:
    double last_detection_probability_;
    double get_last_detection_probability();
    double sum_of_gaussian_probabilities(std::vector<Eigen::Vector2d> &measurements, double &standard_deviation);
};

IPDAFilter::IPDAFilter() {
    last_detection_probability_ = 0.0;
}

/// @brief 
/// @param measurements Measurements to iterate over
/// @param probability_of_survival How likely the object is to survive (Ps)
/// @param probability_of_detection How likely the object is to be detected (Pd)
/// @param standard_deviation Standard deviation of the measurements (Sk)
/// @param lambda Lambda value for the Poisson distribution (Lambda)
double IPDAFilter::generate_new_probability(std::vector<Eigen::Vector2d> &measurements, double &probability_of_survival, float &probability_of_detection, double &standard_deviation, float &lambda) {
    double predicted_existence_probability = probability_of_survival * last_detection_probability_;
    double l_k = 1 - probability_of_detection + probability_of_detection / lambda * sum_of_gaussian_probabilities(measurements, standard_deviation);

    return (l_k * predicted_existence_probability) / (1 - (1 - l_k) * predicted_existence_probability);
}
double IPDAFilter::sum_of_gaussian_probabilities(std::vector<Eigen::Vector2d> &measurements, double &standard_deviation) {
    double sum = 0.0;
    
    return sum;
}

double IPDAFilter::get_last_detection_probability() {
    return last_detection_probability_;
}

#endif // IPDA_HPP
