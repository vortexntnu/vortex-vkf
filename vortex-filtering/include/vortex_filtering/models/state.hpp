#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <vortex_filtering/probability/multi_var_gauss.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

enum class StateName : size_t { position, velocity, acceleration, jerk, snap, crackle, pop };

template <typename R> consteval auto index_of(const R &range, StateName needle)
{
  auto it = std::ranges::find(range, needle);
  if (it == std::ranges::end(range))
    throw std::logic_error("Element not found!");
  return std::ranges::distance(std::ranges::begin(range), it);
}

template <StateName... Sn> class State : public vortex::prob::MultiVarGauss<sizeof...(Sn)> {
public:
  static constexpr size_t N_STATES = sizeof...(Sn);

  static constexpr std::array<StateName, N_STATES> STATE_NAMES = {Sn...};

  using T = vortex::Types_n<N_STATES>;

  State(const T::Vec_n &mean, const T::Mat_nn &cov)
      : vortex::prob::MultiVarGauss<N_STATES>(mean, cov)
  {
  }

  explicit State(T::Gauss_n x)
      : vortex::prob::MultiVarGauss<N_STATES>(x.mean(), x.cov())
  {
  }

  struct StateMap {
    StateName name;
    size_t start_index;
    size_t end_index;
    constexpr size_t size() const { return end_index - start_index; }
  };
  // private:
  static constexpr size_t UNIQUE_STATES = []() {
    std::array<StateName, N_STATES> state_names = STATE_NAMES;
    std::sort(state_names.begin(), state_names.end());
    auto last = std::unique(state_names.begin(), state_names.end());
    return std::distance(state_names.begin(), last);
  }();

  // static_assert(
  //     []() {
  //       size_t n_unique = 1;
  //       for (size_t i = 0; i < N_STATES; i++) {
  //         if (i > 0 && STATE_NAMES[i] == STATE_NAMES[i - 1]) {
  //           n_unique++;
  //         }
  //       }
  //       if (n_unique != UNIQUE_STATES) {
  //         return false;
  //       }
  //       return true;
  //     }(),
  //     "Groups of state names must not repeat");

  static constexpr std::array<StateName, UNIQUE_STATES> UNIQUE_STATE_NAMES = []() {
    std::array<StateName, UNIQUE_STATES> unique_state_names = {};
    size_t map_index = 0;
    for (size_t i = 1; i < N_STATES; i++) {
      if (STATE_NAMES[i] != STATE_NAMES[i - 1]) {
        unique_state_names[map_index++] = STATE_NAMES[i - 1];
      }
    }
    unique_state_names[map_index] = STATE_NAMES[N_STATES - 1];
    return unique_state_names;
  }();

  static constexpr std::array<StateMap, UNIQUE_STATES> STATE_MAP = []() {
    std::array<StateMap, UNIQUE_STATES> state_map = {};

    size_t start_index = 0;
    size_t map_index   = 0;

    for (size_t i = 1; i < N_STATES; i++) {
      if (STATE_NAMES[i] != STATE_NAMES[i - 1]) {
        state_map[map_index++] = {STATE_NAMES[i - 1], start_index, i};
        start_index            = i;
      }
    }
    state_map[map_index] = {STATE_NAMES[N_STATES - 1], start_index, N_STATES};
    return state_map;
  }();

  static constexpr bool has_state_name(StateName S) { return std::find(UNIQUE_STATE_NAMES.begin(), UNIQUE_STATE_NAMES.end(), S) != UNIQUE_STATE_NAMES.end(); }

public:
  template <StateName S>
  static constexpr StateMap state_loc()
    requires(has_state_name(S))
  {
    return STATE_MAP[index_of(UNIQUE_STATE_NAMES, S)];
  }


  template <StateName S>
    requires(has_state_name(S))
  using T_n = vortex::Types_n<state_loc<S>().size()>;

  template <StateName S>
    requires(has_state_name(S))
  T_n<S>::Vec_n mean_of() const
  {
    constexpr StateMap sm = state_loc<S>();
    return this->mean().template segment<sm.size()>(sm.start_index);
  }

  template <StateName S>
    requires(has_state_name(S))
    void set_mean_of(const T_n<S>::Vec_n &mean)
  {
    constexpr StateMap sm = state_loc<S>();
    this->mean().template segment<sm.size()>(sm.start_index) = mean;
  }

  template <StateName S>
    requires(has_state_name(S))
  T_n<S>::Mat_nn cov_of() const
  {
    constexpr StateMap sm = state_loc<S>();
    return this->cov().template block<sm.size(), sm.size()>(sm.start_index, sm.start_index);
  }

  template <StateName S>
    requires(has_state_name(S))
  void set_cov_of(const T_n<S>::Mat_nn &cov)
  {
    constexpr StateMap sm = state_loc<S>();
    this->cov().template block<sm.size(), sm.size()>(sm.start_index, sm.start_index) = cov;
  }

  T::Gauss_n gauss() const { return static_cast<typename T::Gauss_n>(*this); }
  template <StateName S>
    requires(has_state_name(S))
  T_n<S>::Gauss_n gauss_of() const
  {
    return {mean_of(S), cov_of(S)};
  }
};
