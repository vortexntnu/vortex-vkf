#pragma once
#include "general_model.hpp"

namespace vortex::model::interface {
template <LieGroup Type_x, LieGroup Type_u, LieGroup Type_v>
class DynamicModel : public internal::GeneralModel<Type_x, Type_x, Type_v, Type_u, double> {
public:
  using Mx = Type_x;
  using Mu = Type_u;
  using Mv = Type_v;

  using Tx = typename Mx::Tangent;
  using Tu = typename Mu::Tangent;
  using Tv = typename Mv::Tangent;

  using T = vortex::Types_xuv<Mx::DoF, Mu::DoF, Mv::DoF>;

  virtual Mx f(double dt, const Mx &x, const Mu &u, const Mv &v) const = 0;

  virtual T::Mat_vv Q(double dt, const Mx &x) const = 0;

  Mx sample_f(double dt, const Mx &x, const Mu &u, std::mt19937 &gen) const
  {
    return this->sample(x, gen, u, dt);
  }

private:
  Mx process(const Mx &x, const Mv &v, const Mu &u, const double &dt) const override { return f(dt, x, u, v); }
  T::Mat_vv noise_matrix(const Mx &x, const Mu &/*u*/, const double &dt) const override { return Q(dt, x); }
};

template <std::size_t DIM_x, std::size_t DIM_u, std::size_t DIM_v>
using DynamicModelR = DynamicModel<manif::Rn<double, DIM_x>, manif::Rn<double, DIM_u>, manif::Rn<double, DIM_v>>;

} // namespace vortex::model::interface