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

template <concepts::SensorModelWithDefinedSizes SensModT, models::concepts::ImmModel ImmModelT> class IMMIPDA {
public:
  struct Config : public IPDA::Config {
    models::StateMap states_min_max;
  };

  template <size_t s_k>
  using T         = Types_xzuvw<ImmModelT::N_DIMS_x(s_k), SensModT::N_DIM_z, ImmModelT::N_DIMS_u(s_k), ImmModelT::N_DIMS_v(s_k), SensModT::N_DIM_w>;
  using ImmFilter = ImmFilter<SensModT, ImmModelT>;

  template <size_t s_k> using PDAF = PDAF<ImmModelT::DynModT<s_k>, SensModT>;
  template <size_t s_k> using IPDA = IPDA<ImmModelT::DynModT<s_k>, SensModT>;

  static std::tuple<ImmModelT::GaussTuple_x, ImmModelT::Vec_n, double, std::vector<Vec_z>, std::vector<Vec_z>, ImmModelT::GaussTuple_x, ImmModelT::GaussTuple_z,
                    std::vector<ImmModelT::GaussTuple_x>>
  step(const ImmModelT &imm_model, const SensModT &sens_mod, double timestep, const ImmModelT::GaussTuple_x &x_est_prevs,
       const std::vector<Vec_z> &z_measurements, double existence_prob_est, const ImmModelT::Vec_n &weights, const Config &config)
  {
    using Arr_nX = Eigen::Array<double, ImmModelT::N_MODELS, Eigen::Dynamic>;

    size_t m_k = z_measurements.size();
    // Calculate IMM mixing probabilities
    auto transition_matrix = imm_model.get_pi_mat_d(dt);
    auto mixing_probs      = ImmFilter::calculate_mixing_probs(transition_matrix, weights);

    // IMM mixing
    auto moment_based_preds = ImmFilter::mixing(x_est_prevs, mixing_probs, imm_model.get_all_state_names(), config.states_min_max);

    // IMM filtering
    std::vector<ImmModelT::GaussTuple_x> imm_estimations(m_k + 1);
    auto [x_est_preds, z_est_preds] = ImmFilter::kalman_filter_predictions(imm_model, sens_mod, dt, moment_based_preds);
    imm_estimations.at(0)           = x_est_preds;
    for (size_t a_k = 1; z_k : z_measurements) {
      auto x_est_upds           = ImmFilter::kalman_filter_updates(imm_model, sens_mod, x_est_preds, z_est_preds, z_k);
      imm_estimations.at(a_k++) = x_est_upds;
    }

    // IMM mode probabilities
    Arr_nX predicted_mode_probabilities(m_k);
    for (size_t a_k = 0; auto z_k : z_measurements) {
      predicted_mode_probabilities.col(a_k++) = ImmFilter::update_probabilities(mixing_probs, z_est_preds, z_k, weights);
    }

    // Calculate the hypothesis-conditional likelihood (7.61)
    Arr_nX hypothesis_conditional_likelyhood(ImmModelT::N_MODELS, m_k);
    for (size_t s_k = 0; const auto &z_pred_s_k : z_est_preds) {
      for (size_t a_k = 0; const auto &z_k : z_measurements) {
        hypothesis_conditional_likelyhood(s_k++, a_k++) = z_pred_s_k.pdf(z_k);
      }
    }

    // Calculate the hypothesis likelihood (7.62)
    Eigen::ArrayXd hypothesis_likelihood(m_k + 1);
    for (size_t a_k = 0; a_k < m_k; a_k++) {
      hypothesis_likelihood(a_k) = predicted_mode_probabilities.col(a_k).dot(hypothesis_conditional_likelyhood.col(a_k));
    }

    // Calculate the accociation probabilities (7.56)
    Eigen::ArrayXd association_probabilities(m_k + 1);
    association_probabilities = PDAF<0>::association_probabilities(hypothesis_likelihood, config.prob_of_detection, config.clutter_intensity);

    // Calculate the posterior mode probabilities (7.52)
    Arr_nX posterior_mode_probabilities(ImmModelT::N_MODELS, m_k + 1);
    posterior_mode_probabilities = hypothesis_conditional_likelyhood * predicted_mode_probabilities;
    posterior_mode_probabilities /= posterior_mode_probabilities.colwise().sum().replicate(ImmModelT::N_MODELS, 1);

    // Calculate the mode-conditional association probabilities (7.57)
    Arr_nX mode_conditional_association_probabilities = association_probabilities * posterior_mode_probabilities;
    ImmModelT::Vec_n mode_prob_upd                    = mode_conditional_association_probabilities.rowwise().sum();
    mode_conditional_association_probabilities /= mode_prob_upd.replicate(ImmModelT::N_MODELS, 1);

    // PDAF mixture reduction (7.58)
    ImmModelT::GaussTuple_x x_est_upds;
    std::apply(
        [&](auto &...s_k) {
          (
              {
                std::vector<T<s_k>::Gauss_x &> imm_estimations_s_k;
                for (const auto &imm_estimations_a_k : imm_estimations) {
                  imm_estimations_s_k = std::get<s_k>(imm_estimations_a_k);
                }
                std::get<s_k>(x_est_upds) = T<s_k>::GaussMix_x{imm_estimations_s_k, mode_conditional_association_probabilities.row(s_k)}.reduce();
              },
              ...);
        },
        std::make_index_sequence<ImmModelT::N_MODELS>{});

    // Calculate existence probability
    double existence_prob_upd = IPDA<0>::existence_prob_update(hypothesis_likelihood, existence_prob_est, config);

    return {x_est_upds, mode_prob_upd, existence_prob_upd, z_meas_inside, z_meas_outside, x_est_preds, z_est_preds, imm_estimations};
  }
};

}  // namespace vortex::filter