/**
 * @file imm_model.hpp
 * @author Eirik Kolås
 * @brief
 * @version 0.1
 * @date 2023-11-02
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>

namespace vortex {
namespace models {

template <typename ...DynModels> class ImmModel {
public:
	using DynModTuple = std::tuple<std::shared_ptr<DynModels>...>;
	static constexpr size_t N_MODELS_ = std::tuple_size<DynModTuple>::value;

	using Vec_n     = Eigen::Vector<double, N_MODELS_>;
	using Vec_n_row = Eigen::RowVector<double, N_MODELS_>;
	using Mat_nn    = Eigen::Matrix<double, N_MODELS_, N_MODELS_>;

	template <size_t i> using DynModT = typename std::tuple_element<i, DynModTuple>::type;
	template <size_t i> using BaseI   = typename std::tuple_element<i, std::tuple<DynModels...>>::type::BaseI;
	template <size_t i> using Vec_x   = typename BaseI<i>::Vec_x;
	template <size_t i> using Vec_u   = typename BaseI<i>::Vec_u;
	template <size_t i> using Vec_v   = typename BaseI<i>::Vec_v;
	template <size_t i> using Mat_xx  = typename BaseI<i>::Mat_xx;

	/**
	 * @brief Construct a new Imm Model object
	 * @param models Models to use. The models must have a BaseI typedef specifying the base interface.
	 * @param jump_matrix Markov jump chain matrix for the transition probabilities.
	 * I.e. the probability of switching from model i to model j is jump_matrix(i,j). Diagonal should be 0.
	 * @param hold_times Expected holding time for each state. Parameter is the mean of an exponential distribution.
	 * @note The jump matrix specifies the probability of switching to a model WHEN a switch occurs. 
	 * @note The holding times specifies HOW LONG a state is expected to be held between switches.
	 */
	ImmModel(DynModTuple models, Mat_nn jump_matrix, Vec_n hold_times)
	    : models_(std::move(models)), jump_matrix_(std::move(jump_matrix)), hold_times_(std::move(hold_times))
	{
		if (!jump_matrix_.diagonal().isZero()) {
			throw std::invalid_argument("Jump matrix diagonal should be zero");
		}

		// Normalize jump matrix
		for (int i = 0; i < jump_matrix_.rows(); i++) {
			jump_matrix_.row(i) /= jump_matrix_.row(i).sum();
		}
	}


	/**
	 * @brief Compute the continuous-time transition matrix
	 * @return Matrix Continuous-time transition matrix
	 * @note See https://en.wikipedia.org/wiki/Continuous-time_Markov_chain
	 */
	Mat_nn get_pi_mat_c() const
	{
		Vec_n t_inv = hold_times_.cwiseInverse();
		return -t_inv.asDiagonal() + t_inv * jump_matrix_;
	}

	/**
	 * @brief Compute the discrete-time transition matrix
	 * @return Matrix Discrete-time transition matrix
	 */
	Mat_nn get_pi_mat_d(double dt) const { return (get_pi_mat_c() * dt).exp(); }

	/**
	 * @brief Get the dynamic models
	 * @return tuple of shared pointers to dynamic models
	 */
	DynModTuple get_models() const { return models_; }

	/**
	 * @brief Get specific dynamic model
	 * @tparam i Index of model
	 * @return ModelT<i> shared pointer to model
	 */
	template <size_t i> DynModT<i> get_model() const { return std::get<i>(models_); }

	/**
	 * @brief f_d of specific dynamic model
	 * @tparam i Index of model
	 * @param dt Time step
	 * @param x State
	 * @param u Input (optional)
	 * @param v Noise (optional)
	 * @return Vec_x
	 */
	template <size_t i> Vec_x<i> f_d(double dt, const Vec_x<i> &x, const Vec_u<i> &u = Vec_u<i>::Zero(), const Vec_v<i> &v = Vec_v<i>::Zero()) const 
	{ 
		return std::get<i>(models_)->f_d(dt, x, u, v); 
	}

	/**
	 * @brief Q_d of specific dynamic model
	 * @tparam i Index of model
	 * @param dt Time step
	 * @param x State
	 * @return Mat_xx
	 */
	template <size_t i> Mat_xx<i> Q_d(double dt, const Vec_x<i> &x) const { return std::get<i>(models_)->Q_d(dt, x); }

private:
	DynModTuple models_;
	Mat_nn jump_matrix_;
	Vec_n hold_times_;
};

} // namespace models
} // namespace vortex