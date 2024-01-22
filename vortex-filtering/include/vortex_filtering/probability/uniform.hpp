/**
 * @file uniform.hpp
 * @author Eirik Kolås
 * @brief 
 * @version 0.1
 * @date 2024-01-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once
#include <Eigen/Dense>
#include <random>

namespace vortex::prob {

template <size_t n_dims>
class Uniform {
public:
  using Vec_n  = Eigen::Vector<double, n_dims>;
  using Mat_nn = Eigen::Matrix<double, n_dims, n_dims>;

  constexpr Uniform(Vec_n lower, Vec_n upper)
      : lower_(lower), upper_(upper)
  {}

  double pr(Vec_n x) const
  {
    for (size_t i = 0; i < n_dims; i++) {
      if (x(i) < lower_(i) || x(i) > upper_(i)) {
        return 0;
      }
    }
    return 1.0 / (upper_ - lower_).prod();
  }

  Vec_n sample(std::mt19937 &gen) const
  {
    Vec_n sample;
    for (size_t i = 0; i < n_dims; i++) {
      std::uniform_real_distribution<double> dist(lower_(i), upper_(i));
      sample(i) = dist(gen);
    }
    return sample;
  }

  Vec_n sample() const 
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    return sample(gen);
  }

  Vec_n mean() const { return (upper_ + lower_) / 2; }
  Mat_nn cov() const 
  { 
    Vec_n diff = upper_ - lower_;
    for (double &d : diff) {
      d *= d/12;
    }
    return diff.asDiagonal();
  }

  const Vec_n &lower() const { return lower_; }
  const Vec_n &upper() const { return upper_; }



  private:
    Vec_n lower_;
    Vec_n upper_;
};

} // namespace vortex::prob