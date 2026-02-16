#define EIGEN_NO_AUTOMATIC_RESIZING
#include <gtest/gtest.h>
#include <limits>
#include <vortex_filtering/filters/immipda.hpp>

class IMMIPDA : public ::testing::Test {
   protected:
    using DynMod1_ = vortex::models::ConstantPosition;
    using DynMod2_ = vortex::models::ConstantVelocity;
    using SensMod_ = vortex::models::IdentitySensorModel<2, 2>;
    using ImmModel_ = vortex::models::ImmModel<DynMod1_, DynMod2_>;
    using IMMIPDA_ = vortex::filter::IMMIPDA<ImmModel_, SensMod_>;

    using S = vortex::StateName;

    IMMIPDA()
        : imm_model_(jump_matrix, hold_times, DynMod1_(0.5), DynMod2_(0.5)),
          sensor_model_(2),
          config_{.pdaf =
                      {
                          .mahalanobis_threshold = 1.0,
                          .prob_of_detection = 0.9,
                          .clutter_intensity = 1.0,
                      },
                  .ipda =
                      {
                          .prob_of_survival = 0.9,
                          .estimate_clutter = true,
                          .update_existence_probability_on_no_detection = true,
                      },
                  .immipda = {
                      .states_min_max = {{S::position, {-100.0, 100.0}},
                                         {S::velocity, {-10.0, 10.0}}},
                  }} {};

    double dt_ = 1;
    Eigen::Matrix2d jump_matrix{{0.0, 1.0}, {1.0, 0.0}};
    Eigen::Vector2d hold_times{100.0, 100.0};

    ImmModel_ imm_model_;
    SensMod_ sensor_model_;
    IMMIPDA_::Config config_;
};

TEST_F(IMMIPDA, step) {
    using namespace vortex;

    auto [min_vel, max_vel] = config_.immipda.states_min_max.at(S::velocity);

    Eigen::Matrix4d mode_2_cov = prob::Uniform<4>{
        {-1.0, -1.0, min_vel, min_vel},
        {1.0, 1.0, max_vel, max_vel}}.cov();

    ImmModel_::GaussTuple_x x0 = {prob::Gauss2d::Standard(),
                                  {{0.0, 0.0, 1.0, 0.0}, mode_2_cov}};
    ImmModel_::Vec_n model_weights = {0.5, 0.5};

    Eigen::Array<double, 2, -1> z0 = {
        {1.0, 1.0, 1.0, 20},
        {0.1, -0.1, 0.0, 0},
    };

    IMMIPDA_::Output out = IMMIPDA_::step(
        imm_model_, sensor_model_, dt_, {x0, model_weights, 0.5}, z0, config_);

    ASSERT_EQ(out.gated_measurements.colwise().any().count(), 3);

    ASSERT_GT(out.state.mode_probabilities(0), 0.0);
    ASSERT_GT(out.state.mode_probabilities(1), 0.0);
    ASSERT_LT(out.state.mode_probabilities(0), 1.0);
    ASSERT_LT(out.state.mode_probabilities(1), 1.0);
}
