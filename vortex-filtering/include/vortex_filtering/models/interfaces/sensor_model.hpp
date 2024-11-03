#pragma once
#include "general_model.hpp"

namespace vortex::model::interface
{
template <LieGroup Type_x, LieGroup Type_z, LieGroup Type_w>
class SensorModel : public internal::GeneralModel<Type_x, Type_z, Type_w> {
public:
  using Mx = Type_x;
  using Mz = Type_z;
  using Mw = Type_w;

  using Tx = typename Mx::Tangent;
  using Tz = typename Mz::Tangent;
  using Tw = typename Mw::Tangent;

  using T = vortex::Types_xzw<Mx::DoF, Mz::DoF, Mw::DoF>;

  virtual Mz g(const Mx &x, const Mw &w) const = 0;

  virtual T::Mat_ww R_w(const Mx &x) const = 0;

  Mx sample_g(const Mx &x, std::mt19937 &gen) const { return this->sample(x, gen); }

private:
  Mz process(const Mx &x, const Mw &w) const override { return g(x, w); }
  T::Mat_ww noise_matrix(const Mx &x) const override { return R_w(x); }
};

template <std::size_t DIM_x, std::size_t DIM_z, std::size_t DIM_w>
using SensorModelR = SensorModel<manif::Rn<double, DIM_x>, manif::Rn<double, DIM_z>, manif::Rn<double, DIM_w>>;

} // namespace vortex::model::interface
