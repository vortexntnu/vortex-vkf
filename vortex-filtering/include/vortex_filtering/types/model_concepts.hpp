#pragma once
#include <concepts>
#include <vortex_filtering/models/state.hpp>
#include <vortex_filtering/types/general_concepts.hpp>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::concepts {
using std::size_t;

/**
 * @brief Concept for MultiVarGauss-like classes. Requires the following
 * functions:
 * @brief - `double pdf(Vec_n)`
 * @brief - `double logpdf(Vec_n)`
 * @brief - `Vec_n mean()`
 * @brief - `Mat_nn cov()`
 * @brief - `int size()`
 * @brief - `Vec_n sample(std::mt19937_64&)`
 * @brief - `double mahalanobis_distance(Vec_n)`
 *
 * @tparam T The MultiVarGauss-like class
 * @tparam n_dim Dimension of the state
 */
template <typename T, size_t n_dim>
concept MultiVarGaussLike = requires {
    {
        std::declval<T>().pdf(std::declval<typename Types_n<n_dim>::Vec_n>())
    } -> std::convertible_to<double>;

    {
        std::declval<T>().logpdf(std::declval<typename Types_n<n_dim>::Vec_n>())
    } -> std::convertible_to<double>;

    {
        std::declval<T>().mean()
    } -> mat_convertible_to<typename Types_n<n_dim>::Vec_n>;

    {
        std::declval<T>().cov()
    } -> mat_convertible_to<typename Types_n<n_dim>::Mat_nn>;

    { std::declval<T>().size() } -> std::convertible_to<size_t>;

    {
        std::declval<T>().sample(std::declval<std::mt19937&>())
    } -> mat_convertible_to<typename Types_n<n_dim>::Vec_n>;

    {
        std::declval<T>().mahalanobis_distance(
            std::declval<typename Types_n<n_dim>::Vec_n>())
    } -> std::convertible_to<double>;
};

template <typename T, size_t n_dim>
concept StateLike = requires {
    requires MultiVarGaussLike<T, n_dim>;

    { T::N_STATES } -> std::convertible_to<size_t>;

    {
        T::STATE_NAMES
    } -> std::convertible_to<std::array<StateName, T::N_STATES>>;

    {
        std::declval<T>().state_loc(std::declval<StateName>())
    } -> std::convertible_to<typename T::StateMap>;
};

/**
 * @brief Concept for dynamic models. Requires the following functions:
 * @brief - `Vec_x f_d(double, Vec_x, Vec_u, Vec_v)`
 * @brief - `Mat_vv Q_d(double, Vec_x)`
 *
 * @tparam DynMod The dynamic model type
 * @tparam n_dim_x Dimension of the state
 * @tparam n_dim_u Dimension of the input
 * @tparam n_dim_v Dimension of the process noise
 */
template <typename DynMod, size_t n_dim_x, size_t n_dim_u, size_t n_dim_v>
concept DynamicModel = requires {
    {
        std::declval<DynMod>().f_d(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>(),
            std::declval<typename Types_u<n_dim_u>::Vec_u>(),
            std::declval<typename Types_v<n_dim_v>::Vec_v>())
    } -> mat_convertible_to<typename Types_x<n_dim_x>::Vec_x>;

    {
        std::declval<DynMod>().Q_d(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_v<n_dim_v>::Mat_vv>;
};

/**
 * @brief Concept for dynamic models with time-varying parameters. Requires the
 * following functions:
 * @brief - `Vec_x f_d(double, Vec_x, Vec_u, Vec_v)`
 * @brief - `Mat_vv Q_d(double, Vec_x)`
 * @brief - `Mat_xx A_d(double, Vec_x)`
 * @brief - `Mat_xu B_d(double, Vec_x)`
 * @brief - `Mat_xv G_d(double, Vec_x)`
 * @brief - `Gauss_x pred_from_est(double, Gauss_x, Vec_u)`
 * @brief - `Gauss_x pred_from_state(double, Vec_x, Vec_u)`
 *
 * @tparam DynMod The dynamic model type
 * @tparam n_dim_x Dimension of the state
 * @tparam n_dim_u Dimension of the input
 * @tparam n_dim_v Dimension of the process noise
 */
template <typename DynMod, size_t n_dim_x, size_t n_dim_u, size_t n_dim_v>
concept DynamicModelLTV = requires {
    requires DynamicModel<DynMod, n_dim_x, n_dim_u,
                          n_dim_v>;  // Assuming DynamicModel is correctly
                                     // defined as shown before
    {
        std::declval<DynMod>().A_d(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_x<n_dim_x>::Mat_xx>;

    {
        std::declval<DynMod>().B_d(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_xu<n_dim_x, n_dim_u>::Mat_xu>;

    {
        std::declval<DynMod>().G_d(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_xv<n_dim_x, n_dim_v>::Mat_xv>;

    {
        std::declval<DynMod>().pred_from_est(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Gauss_x>(),
            std::declval<typename Types_u<n_dim_u>::Vec_u>())
    } -> std::convertible_to<typename Types_x<n_dim_x>::Gauss_x>;

    {
        std::declval<DynMod>().pred_from_state(
            std::declval<double>(),
            std::declval<typename Types_x<n_dim_x>::Vec_x>(),
            std::declval<typename Types_u<n_dim_u>::Vec_u>())
    } -> std::convertible_to<typename Types_x<n_dim_x>::Gauss_x>;
};

/**
 *  @brief Concept for sensor models. Requires the following functions:
 * @brief - `Vec_z h(Vec_x, Vec_w)`
 * @brief - `Mat_ww R(Vec_x)`
 *
 * @tparam SensMod The sensor model type
 * @tparam n_dim_x Dimension of the state
 * @tparam n_dim_z Dimension of the measurement
 * @tparam n_dim_w Dimension of the measurement noise
 */
template <typename SensMod, size_t n_dim_x, size_t n_dim_z, size_t n_dim_w>
concept SensorModel = requires {
    {
        std::declval<SensMod>().h(
            std::declval<typename Types_x<n_dim_x>::Vec_x>(),
            std::declval<typename Types_w<n_dim_w>::Vec_w>())
    } -> mat_convertible_to<typename Types_z<n_dim_z>::Vec_z>;
    {
        std::declval<SensMod>().R(
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_w<n_dim_w>::Mat_ww>;
};

/**
 * @brief Concept for sensor models with time-varying parameters. Requires the
 * following functions:
 * @brief - `Vec_z h(Vec_x, Vec_w)`
 * @brief - `Mat_zz R(Vec_x)`
 * @brief - `Mat_zw H(double, Vec_x)`
 * @brief - `Mat_zx C(Vec_x)`
 * @brief - `Gauss_z pred_from_est(Gauss_x)`
 * @brief - `Gauss_z pred_from_state(Vec_x)`
 */
template <typename SensMod, size_t n_dim_x, size_t n_dim_z, size_t n_dim_w>
concept SensorModelLTV = requires {
    requires SensorModel<SensMod, n_dim_x, n_dim_z, n_dim_w>;
    {
        std::declval<SensMod>().H(
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_zw<n_dim_z, n_dim_w>::Mat_zw>;
    {
        std::declval<SensMod>().C(
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> mat_convertible_to<typename Types_xz<n_dim_x, n_dim_z>::Mat_zx>;
    {
        std::declval<SensMod>().pred_from_est(
            std::declval<typename Types_x<n_dim_x>::Gauss_x>())
    } -> std::convertible_to<typename Types_z<n_dim_z>::Gauss_z>;
    {
        std::declval<SensMod>().pred_from_state(
            std::declval<typename Types_x<n_dim_x>::Vec_x>())
    } -> std::convertible_to<typename Types_z<n_dim_z>::Gauss_z>;
};

///////////////////////////////////
//
///////////////////////////////////

template <typename DynMod>
concept DynamicModelWithDefinedSizes = requires {
    { DynMod::N_DIM_x } -> std::convertible_to<size_t>;
    { DynMod::N_DIM_v } -> std::convertible_to<size_t>;
    { DynMod::N_DIM_u } -> std::convertible_to<size_t>;
    requires DynamicModel<DynMod, DynMod::N_DIM_x, DynMod::N_DIM_u,
                          DynMod::N_DIM_v>;
};

template <typename DynMod>
concept DynamicModelLTVWithDefinedSizes = requires {
    requires DynamicModelWithDefinedSizes<DynMod>;
    { DynMod::N_DIM_x } -> std::convertible_to<size_t>;
    { DynMod::N_DIM_v } -> std::convertible_to<size_t>;
    { DynMod::N_DIM_u } -> std::convertible_to<size_t>;
    requires DynamicModelLTV<DynMod, DynMod::N_DIM_x, DynMod::N_DIM_u,
                             DynMod::N_DIM_v>;
};

template <typename SensMod>
concept SensorModelWithDefinedSizes = requires {
    { SensMod::N_DIM_x } -> std::convertible_to<size_t>;
    { SensMod::N_DIM_z } -> std::convertible_to<size_t>;
    { SensMod::N_DIM_w } -> std::convertible_to<size_t>;
    requires SensorModel<SensMod, SensMod::N_DIM_x, SensMod::N_DIM_z,
                         SensMod::N_DIM_w>;
};

template <typename SensMod>
concept SensorModelLTVWithDefinedSizes = requires {
    requires SensorModelWithDefinedSizes<SensMod>;
    { SensMod::N_DIM_x } -> std::convertible_to<size_t>;
    { SensMod::N_DIM_z } -> std::convertible_to<size_t>;
    { SensMod::N_DIM_w } -> std::convertible_to<size_t>;
    requires SensorModelLTV<SensMod, SensMod::N_DIM_x, SensMod::N_DIM_z,
                            SensMod::N_DIM_w>;
};

}  // namespace vortex::concepts
