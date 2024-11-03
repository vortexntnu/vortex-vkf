#pragma once
#include <manif/manif.h>
#include <vortex_filtering/probability/lie_group_gauss.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::model {
template <typename T>
concept LieGroup = requires { std::is_base_of_v<manif::LieGroupBase<T>, T>; };
} // namespace vortex::model

namespace vortex::model::interface::internal {

template <LieGroup M_In, LieGroup M_Out, LieGroup M_Noise, typename... T> class GeneralModel {
protected:
  using Mat_noise = Eigen::Matrix<typename M_Noise::Scalar, M_Noise::DoF, M_Noise::DoF>;

  virtual M_Out process(const M_In &in, const M_Noise &noise, const T&... args) const = 0;

  virtual Mat_noise noise_matrix(const M_In &in, const T&... args) const = 0;

  M_Out sample(const M_In &in, std::mt19937 &gen, T... args) const
  {
    prob::LieGroupGauss<M_Noise> noise_model(M_Noise::Identity(), noise_matrix(in, args...));
    M_Noise sampled_noise = noise_model.sample(gen);
    return process(in, sampled_noise, args...);
  }
};

} // namespace vortex::model::interface::internal