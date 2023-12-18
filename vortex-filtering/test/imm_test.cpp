#include <gtest/gtest.h>
#include <memory>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/dynamic_models.hpp>

TEST(ImmModel, init)
{
    using namespace vortex::models;

    auto model_2d = std::make_shared<IdentityDynamicModel<2>>(1.0);
    auto model_3d = std::make_shared<IdentityDynamicModel<3>>(1.0);

    Eigen::Matrix2d jump_mat;
    jump_mat << 0, 1, 1, 0;
    Eigen::Vector2d hold_times;
    hold_times << 1, 1;

    ImmModel<IdentityDynamicModel<2>, IdentityDynamicModel<3>> imm_model(std::make_tuple(model_2d, model_3d), jump_mat, hold_times);

    EXPECT_EQ(typeid(*imm_model.get_model<0>()), typeid(IdentityDynamicModel<2>));
    EXPECT_EQ(typeid(*imm_model.get_model<1>()), typeid(IdentityDynamicModel<3>));
    EXPECT_EQ(typeid(imm_model.f_d<0>(1.0, Eigen::Vector2d::Zero())), typeid(Eigen::Vector2d));
    EXPECT_EQ(typeid(imm_model.f_d<1>(1.0, Eigen::Vector3d::Zero())), typeid(Eigen::Vector3d));
    EXPECT_EQ(typeid(imm_model.Q_d<0>(1.0, Eigen::Vector2d::Zero())), typeid(Eigen::Matrix2d));
    EXPECT_EQ(typeid(imm_model.Q_d<1>(1.0, Eigen::Vector3d::Zero())), typeid(Eigen::Matrix3d));
}