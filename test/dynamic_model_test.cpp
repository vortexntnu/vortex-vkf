#include <gtest/gtest.h>
#include <models/dynamic_model.hpp>
#include "test_models.hpp"

namespace simple_dynamic_model_test {

using State  = typename SimpleDynamicModel::State;
using Mat_xx = typename SimpleDynamicModel::Mat_xx;

TEST(DynamicModel, initSimpleModel)
{   
    SimpleDynamicModel model;
}

TEST(DynamicModel, iterateSimpleModel)
{
    SimpleDynamicModel model;
    double dt = 1.0;
    State x = State::Zero();

    for (size_t i = 0; i < 10; i++)
    {
        EXPECT_EQ(model.f_d(x, dt), std::exp(-dt) * x);
        x = model.f_d(x, dt);
    }

}

} // namespace simple_model_test