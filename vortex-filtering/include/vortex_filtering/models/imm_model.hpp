/**
 * @file imm_model.hpp
 * @author Eirik Kolås
 * @brief Container class for interacting multiple models.
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

/**
 * @brief Container class for interacting multiple models.
 * @tparam DynModels Dynamic models to use. 
 * @note The models must have a BaseI typedef specifying the base interface (e.g. `using BaseI = ...`). 
 * @note Alternatively the corresponding `DynamicModelI` parent class can be passed instead.
 */
template <typename ...DynModels> class ImmModel {
public:
	using DynModTuple    = std::tuple<DynModels...>; 
	using DynModPtrTuple = std::tuple<std::shared_ptr<DynModels>...>;
	using Gauss_xTuple   = std::tuple<typename DynModels::BaseI::Gauss_x...>;

	static constexpr int N_MODELS = std::tuple_size<DynModPtrTuple>::value;

	using Vec_n     = Eigen::Vector<double, N_MODELS>;
	using Vec_n_row = Eigen::RowVector<double, N_MODELS>;
	using Mat_nn    = Eigen::Matrix<double, N_MODELS, N_MODELS>;

	template <size_t i> using DynModT = typename std::tuple_element<i, DynModPtrTuple>::type;
	template <size_t i> using BaseI   = typename std::tuple_element<i, DynModTuple>::type::BaseI; // Get the base interface of the i'th model
	template <size_t i> using Vec_x   = typename BaseI<i>::Vec_x;
	template <size_t i> using Vec_u   = typename BaseI<i>::Vec_u;
	template <size_t i> using Vec_v   = typename BaseI<i>::Vec_v;
	template <size_t i> using Mat_xx  = typename BaseI<i>::Mat_xx;

	/**
	 * @brief Construct a new Imm Model object
	 * @tparam Models Dynamic models to use. The models must have a BaseI typedef specifying the base interface (e.g. `using BaseI = ...`).
	 * Alternatively the corresponding `DynamicModelI` parent class can be passed instead.
	 * @param models Models to use. The models must have a BaseI typedef specifying the base interface.
	 * @param jump_matrix Markov jump chain matrix for the transition probabilities.
	 * I.e. the probability of switching from model i to model j is jump_matrix(i,j). Diagonal should be 0.
	 * @param hold_times Expected holding time for each state. Parameter is the mean of an exponential distribution.
	 * @note The jump matrix specifies the probability of switching to a model WHEN a switch occurs. 
	 * @note The holding times specifies HOW LONG a state is expected to be held between switches.
	 */
	ImmModel(DynModPtrTuple models, Mat_nn jump_matrix, Vec_n hold_times)
	    : models_(std::move(models)), jump_matrix_(std::move(jump_matrix)), hold_times_(std::move(hold_times))
	{
		if (!jump_matrix_.diagonal().isZero()) {
			throw std::invalid_argument("Jump matrix diagonal should be zero");
		}

		// Check that the jump matrix is valid
		for (int i = 0; i < jump_matrix_.rows(); i++) {
			if (jump_matrix_.row(i).sum() != 1.0) {
				throw std::invalid_argument("Jump matrix row " + std::to_string(i) + " should sum to 1");
			}
		}
	}


	/**
	 * @brief Compute the continuous-time transition matrix
	 * @return Matrix Continuous-time transition matrix
	 * @note See https://en.wikipedia.org/wiki/Continuous-time_Markov_chain.
	 */
	Mat_nn get_pi_mat_c() const
	{
		Mat_nn pi_mat_c = hold_times_.replicate(1, N_MODELS);

		// Multiply the jump matrix with the hold times
		pi_mat_c = pi_mat_c.cwiseProduct(jump_matrix_);

		// Each row should sum to zero
		pi_mat_c -= hold_times_.asDiagonal();

		return pi_mat_c;
	}

	/**
	 * @brief Compute the discrete-time transition matrix
	 * @return Matrix Discrete-time transition matrix
	 */
	Mat_nn get_pi_mat_d(double dt) const { return (get_pi_mat_c() * dt).exp(); }

	/**
	 * @brief Get the dynamic models
	 * @return Reference to tuple of shared pointers to dynamic models
	 */
	DynModPtrTuple& get_models() const { return models_; }

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
		return get_model<i>()->f_d(dt, x, u, v); 
	}

	/**
	 * @brief Q_d of specific dynamic model
	 * @tparam i Index of model
	 * @param dt Time step
	 * @param x State
	 * @return Mat_xx
	 */
	template <size_t i> Mat_xx<i> Q_d(double dt, const Vec_x<i> &x) const { return get_model<i>()->Q_d(dt, x); }

	/**
	 * @brief Get the number of dimensions for the state vector of each dynamic model
	 * @return std::array<int, N_MODELS>
	 */
	static constexpr std::array<int, N_MODELS> get_n_dim_x() { return {DynModels::BaseI::N_DIM_x...}; }


private:
	const DynModPtrTuple models_;
	const Mat_nn jump_matrix_;
	const Vec_n hold_times_;
};

} // namespace models
} // namespace vortex