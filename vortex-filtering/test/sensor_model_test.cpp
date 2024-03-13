#include "test_models.hpp"

#include <gtest/gtest.h>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/models/sensor_models.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace simple_sensor_model_test {

using SensorModel = vortex::models::IdentitySensorModel<2, 1>;

using T = vortex::Types_xz<SensorModel::N_DIM_x, SensorModel::N_DIM_z>;

using Vec_x   = T::Vec_x;
using Vec_z   = T::Vec_z;
using Mat_xx  = T::Mat_xx;
using Mat_zz  = T::Mat_zz;
using Mat_zx  = T::Mat_zx;
using Mat_xz  = T::Mat_xz;
using Gauss_x = T::Gauss_x;
using Gauss_z = T::Gauss_z;

TEST(SensorModel, initSimpleModel)
{
  SensorModel model(0.1);
  EXPECT_EQ(model.h(Vec_x::Zero()), Vec_z::Zero());

  Vec_x x{1, 2};
  Vec_z z{1};
  EXPECT_EQ(model.h(x), z);
}

TEST(SensorModel, predictSimpleModel)
{
  SensorModel model(std::sqrt(0.1));
  Gauss_x x_est{Vec_x::Zero(), Mat_xx::Identity()};
  Gauss_z pred = model.pred_from_est(x_est);
  EXPECT_EQ(pred.mean(), Vec_z::Zero());
  EXPECT_TRUE(pred.cov().isApprox(Mat_zz::Identity() * 1.1));

  pred = model.pred_from_state(x_est.mean());
  EXPECT_TRUE(pred.cov().isApprox(Mat_zz::Identity() * 0.1));
}

} // namespace simple_sensor_model_test
