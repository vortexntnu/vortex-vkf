#include "gtest_assertions.hpp"

#include <gtest/gtest.h>
#include <vortex_filtering/models/state.hpp>

TEST(State, typeChecks)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  ASSERT_EQ(StateT::N_STATES, 4);
  ASSERT_EQ(StateT::UNIQUE_STATES_COUNT, 3);
  ASSERT_EQ(StateT::state_loc(S::position).start_index, 0);
  ASSERT_EQ(StateT::state_loc(S::velocity).start_index, 1);
  ASSERT_EQ(StateT::state_loc(S::acceleration).start_index, 3);

  ASSERT_EQ(StateT::state_loc(S::position).end_index, 1);
  ASSERT_EQ(StateT::state_loc(S::velocity).end_index, 3);
  ASSERT_EQ(StateT::state_loc(S::acceleration).end_index, 4);

  ASSERT_TRUE(StateT::has_state_name(S::position));
  ASSERT_TRUE(StateT::has_state_name(S::velocity));
  ASSERT_TRUE(StateT::has_state_name(S::acceleration));
  ASSERT_FALSE(StateT::has_state_name(S::jerk));
}

TEST(State, init)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  auto x = prob::Gauss4d::Standard();
  StateT state(x);

  ASSERT_EQ(state.mean(), x.mean());
  ASSERT_EQ(state.cov(), x.cov());
  ASSERT_EQ(state.gauss(), x);
}

TEST(State, getMean)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  auto x = prob::Gauss4d::Standard();
  StateT state(x);

  ASSERT_TRUE(isApproxEqual(state.mean_of<S::position>(), x.mean().template head<1>(), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.mean_of<S::velocity>(), x.mean().template segment<2>(1), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.mean_of<S::acceleration>(), x.mean().template tail<1>(), 1e-6));
}

TEST(State, getCov)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  auto x = prob::Gauss4d::Standard();
  StateT state(x);

  ASSERT_TRUE(isApproxEqual(state.cov_of<S::position>(), x.cov().template block<1, 1>(0, 0), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.cov_of<S::velocity>(), x.cov().template block<2, 2>(1, 1), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.cov_of<S::acceleration>(), x.cov().template block<1, 1>(3, 3), 1e-6));
}

TEST(State, setMean)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  auto x = prob::Gauss4d::Standard();
  StateT state(x);

  StateT::T::Vec_n mean = StateT::T::Vec_n::Random();
  StateT::T::Mat_nn cov = StateT::T::Mat_nn::Random();

  cov = 0.5 * (cov + cov.transpose()).eval();
  cov += StateT::T::Mat_nn::Identity() * StateT::N_STATES;

  StateT::T::Gauss_n x_new = {mean, cov};
  state.set_mean_of<S::position>(x_new.mean().template head<1>());
  state.set_mean_of<S::velocity>(x_new.mean().template segment<2>(1));
  state.set_mean_of<S::acceleration>(x_new.mean().template tail<1>());

  ASSERT_TRUE(isApproxEqual(state.mean(), x_new.mean(), 1e-6));
}

TEST(State, setCov)
{
  using namespace vortex;

  using S      = StateName;
  using StateT = State<S, S::position, S::velocity, S::velocity, S::acceleration>;

  auto x = prob::Gauss4d::Standard();
  StateT state(x);

  StateT::T::Vec_n mean = StateT::T::Vec_n::Random();
  StateT::T::Mat_nn cov = StateT::T::Mat_nn::Random();
  cov                   = 0.5 * (cov + cov.transpose()).eval();
  cov += StateT::T::Mat_nn::Identity() * StateT::N_STATES;

  StateT::T::Gauss_n x_new = {mean, cov};
  state.set_cov_of<S::position>(x_new.cov().template block<1, 1>(0, 0));
  state.set_cov_of<S::velocity>(x_new.cov().template block<2, 2>(1, 1));
  state.set_cov_of<S::acceleration>(x_new.cov().template block<1, 1>(3, 3));

  ASSERT_TRUE(isApproxEqual(state.cov_of<S::position>(), x_new.cov().template block<1, 1>(0, 0), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.cov_of<S::velocity>(), x_new.cov().template block<2, 2>(1, 1), 1e-6));
  ASSERT_TRUE(isApproxEqual(state.cov_of<S::acceleration>(), x_new.cov().template block<1, 1>(3, 3), 1e-6));

  ASSERT_FALSE(isApproxEqual(state.cov(), x_new.cov(), 1e-6));
}