#include <array>
#include <memory>
#include <ranges>
#include <tuple>
#include <vector>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/imm_filter.hpp>
#include <vortex_filtering/filters/ipda.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/types/model_concepts.hpp>

namespace vortex::filter {

namespace config {
struct IMMIPDA {
  models::StateMap states_min_max;
};
} // namespace config

template <models::concepts::ImmModel ImmModelT, concepts::SensorModelWithDefinedSizes SensModT> class IMMIPDA {
public:
  template <size_t s_k>
  using T          = Types_xzuvw<ImmModelT::N_DIMS_x.at(s_k), SensModT::N_DIM_z, ImmModelT::N_DIMS_u.at(s_k), ImmModelT::N_DIMS_v.at(s_k), SensModT::N_DIM_w>;
  using ImmFilter_ = ImmFilter<ImmModelT, SensModT>;
  using Arr_nXd    = Eigen::Array<double, ImmModelT::N_MODELS, Eigen::Dynamic>;
  using Arr_nXb    = Eigen::Array<bool, ImmModelT::N_MODELS, Eigen::Dynamic>;

  template <size_t s_k> using PDAF_ = PDAF<typename ImmModelT::DynModT<s_k>, SensModT>;
  template <size_t s_k> using IPDA_ = IPDA<typename ImmModelT::DynModT<s_k>, SensModT>;

  struct Config {
    config::PDAF pdaf;
    config::IPDA ipda;
    config::IMMIPDA immipda;
  };

  struct Output {
    ImmModelT::GaussTuple_x x_final;
    ImmModelT::Vec_n mode_prob_upd;
    double existence_prob_upd;
    ImmModelT::GaussTuple_x x_preds;
    ImmFilter_::GaussArr_z z_preds;
    std::vector<typename ImmModelT::GaussTuple_x> x_upds;
    Arr_nXb gated_measurements;
  };

  IMMIPDA() = delete;

  /**
   * @brief Do one step of IMMIPDA filtering
   *
   * @param imm_model IMM model to use
   * @param sens_mod Sensor model to use
   * @param timestep Time step
   * @param x_est_prevs Previous state estimates
   * @param z_measurements Measurements
   * @param existence_prob_est Existence probability estimate
   * @param model_weights Weights of the models
   * @param config Configuration
   * @return std::tuple<ImmModelT::GaussTuple_x, ImmModelT::Vec_n, double, std::vector<Vec_z>, std::vector<Vec_z>, ImmModelT::GaussTuple_x,
   * ImmModelT::GaussTuple_z, std::vector<ImmModelT::GaussTuple_x>>
   */
  static Output step(const ImmModelT &imm_model, const SensModT &sens_mod, double timestep, const ImmModelT::GaussTuple_x &x_est_prevs,
                     const Arr_nXd &z_measurements, double existence_prob_est, const ImmModelT::Vec_n &model_weights, const Config &config)
  {

    size_t m_k = z_measurements.size();
    // Calculate IMM mixing probabilities
    auto transition_matrix = imm_model.get_pi_mat_d(timestep);
    auto mixing_probs      = ImmFilter_::calculate_mixing_probs(transition_matrix, model_weights);

    // IMM mixing
    auto moment_based_preds = ImmFilter_::mixing(x_est_prevs, mixing_probs, imm_model.get_all_state_names(), config.immipda.states_min_max);

    // IMM Kalman prediction
    std::vector<typename ImmModelT::GaussTuple_x> imm_estimates(m_k + 1);
    auto [x_est_preds, z_est_preds] = ImmFilter_::kalman_filter_predictions(imm_model, sens_mod, timestep, moment_based_preds);

    // Gate measurements
    Arr_nXb gated_measurements(ImmModelT::N_MODELS, m_k);
    [&]<size_t... s_k>(std::index_sequence<s_k...>) {
      ((gated_measurements.row(s_k) = PDAF_<s_k>::apply_gate(z_measurements, z_est_preds.at(s_k), {config.pdaf})), ...);
    }(std::make_index_sequence<ImmModelT::N_MODELS>{});

    // Split measurements
    size_t m_k_inside = (gated_measurements.colwise().any()).count();
    Arr_nXd z_meas_inside(ImmModelT::N_MODELS, m_k_inside);
    Arr_nXb gated_meas_inside(ImmModelT::N_MODELS, m_k_inside);

    for (size_t a_k = 0; const auto &z_k : z_measurements.colwise()) {
      if (gated_measurements.col(a_k).any()) {
        z_meas_inside.col(a_k)     = z_k;
        gated_meas_inside.col(a_k) = gated_measurements.col(a_k);
      }
      a_k++;
    }

    // IMM Kalman update
    imm_estimates.at(0) = x_est_preds;
    for (size_t a_k : std::views::iota(1u, m_k_inside)) {
      auto z_k              = z_meas_inside.col(a_k);
      auto gated_k          = gated_meas_inside.col(a_k);
      imm_estimates.at(a_k) = ImmFilter_::kalman_filter_updates(imm_model, sens_mod, timestep, x_est_preds, z_est_preds, z_k, gated_k);
    }

    // IMM mode probabilities
    Arr_nXd predicted_mode_probabilities(ImmModelT::N_MODELS, m_k_inside);
    for (size_t a_k : std::views::iota(0u, m_k_inside)) {
      auto z_k     = z_meas_inside.col(a_k);
      bool gated_k = gated_meas_inside.col(a_k).any();
      if (gated_k) {
        predicted_mode_probabilities.col(a_k) = ImmFilter_::update_probabilities(mixing_probs, z_est_preds, z_k, model_weights);
      }
      else {
        predicted_mode_probabilities.col(a_k) = model_weights;
      }
    }

    // Calculate the hypothesis-conditional likelihood (7.61)
    Arr_nXd hypothesis_conditional_likelyhood(ImmModelT::N_MODELS, m_k_inside);
    for (size_t s_k = 0; const auto &z_pred_s_k : z_est_preds) {
      for (size_t a_k = 0; const auto &z_k : z_meas_inside.colwise()) {
        hypothesis_conditional_likelyhood(s_k, a_k) = z_pred_s_k.pdf(z_k);
        a_k++;
      }
      s_k++;
    }

    // Calculate the hypothesis likelihoods (7.62)
    Eigen::ArrayXd hypothesis_likelihoods(m_k_inside + 1);
    for (size_t a_k = 0; a_k < m_k_inside; a_k++) {
      hypothesis_likelihoods(a_k) = predicted_mode_probabilities.col(a_k).matrix().dot(hypothesis_conditional_likelyhood.col(a_k).matrix());
    }

    // Calculate the accociation probabilities (7.56)
    Eigen::ArrayXd association_probabilities(m_k_inside + 1);
    association_probabilities = PDAF_<0>::association_probabilities(hypothesis_likelihoods, config.pdaf.prob_of_detection, config.pdaf.clutter_intensity);

    // Calculate the posterior mode probabilities (7.52)
    Arr_nXd posterior_mode_probabilities(ImmModelT::N_MODELS, m_k_inside + 1);
    posterior_mode_probabilities = hypothesis_conditional_likelyhood * predicted_mode_probabilities;
    posterior_mode_probabilities /= posterior_mode_probabilities.colwise().sum().replicate(ImmModelT::N_MODELS, 1);

    // Calculate the mode-conditional association probabilities (7.57)
    Arr_nXd mode_conditional_association_probabilities = association_probabilities * posterior_mode_probabilities;
    typename ImmModelT::Vec_n mode_prob_upd            = mode_conditional_association_probabilities.rowwise().sum();
    mode_conditional_association_probabilities /= mode_prob_upd.replicate(ImmModelT::N_MODELS, 1).array();

    // PDAF_ mixture reduction (7.58)
    typename ImmModelT::GaussTuple_x x_est_upds;

    [&]<size_t... s_k>(std::index_sequence<s_k...>) {
      std::tuple<std::vector<typename T<s_k>::Gauss_x>...> imm_estimates_s_k;

      for (const auto &imm_estimates_a_k : imm_estimates) {
        (std::get<s_k>(imm_estimates_s_k).push_back(std::get<s_k>(imm_estimates_a_k)), ...);
      }

      ((std::get<s_k>(x_est_upds) = typename T<s_k>::GaussMix_x{mode_conditional_association_probabilities.row(s_k), std::get<s_k>(imm_estimates_s_k)}.reduce()),
       ...);
    }(std::make_index_sequence<ImmModelT::N_MODELS>{});

    // Calculate existence probability
    double existence_prob_upd = IPDA_<0>::existence_prob_update(hypothesis_likelihoods, existence_prob_est, {config.pdaf, config.ipda});

    return {
        .x_final            = x_est_upds,
        .mode_prob_upd      = mode_prob_upd,
        .existence_prob_upd = existence_prob_upd,
        .x_preds            = x_est_preds,
        .z_preds            = z_est_preds,
        .x_upds             = imm_estimates,
        .gated_measurements = gated_measurements,
    };
  }
};

}  // namespace vortex::filter