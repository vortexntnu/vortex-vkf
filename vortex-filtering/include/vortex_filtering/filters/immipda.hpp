#include <array>
#include <memory>
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
  using T                     = Types_xzuvw<ImmModelT::N_DIMS_x(s_k), SensModT::N_DIM_z, ImmModelT::N_DIMS_u(s_k), ImmModelT::N_DIMS_v(s_k), SensModT::N_DIM_w>;
  using ImmFilter             = ImmFilter<SensModT, ImmModelT>;

  static std::tuple<ImmModelT::GaussTuple_x, ImmModelT::Vec_n, double, std::vector<Vec_z>, std::vector<Vec_z>, Gauss_x, Gauss_z, std::vector<Gauss_x>>
  step(const ImmModelT &imm_model, const SensModT &sens_mod, double timestep, const ImmModelT::GaussTuple_x &x_est_prevs,
       const std::vector<Vec_z> &z_measurements, double survive_est, const ImmModelT::Vec_n &weights, const Config &config)
  {
    // Calculate IMM mixing probabilities
    /////////////////////////////////////////
    auto transition_matrix = imm_model.get_pi_mat_d(dt);
    auto mixing_probs      = ImmFilter::calculate_mixing_probs(transition_matrix, weights);

    // IMM mixing
    /////////////////////////////////////////
    auto moment_based_preds = ImmFilter::mixing(x_est_prevs, mixing_probs, imm_model.get_all_state_names(), config.states_min_max);

    std::vector<ImmModelT::GaussTuple_x> imm_estimations(z_measurements.size() + 1);
    Eigen::Array<double, ImmModelT::N_MODELS, Eigen::Dynamic> hypothesis_conditional_likelyhood(ImmModelT::N_MODELS, z_measurements.size() + 1);
    Eigen::ArrayXd hypothesis_likelihood(z_measurements.size() + 1);
    Eigen::ArrayXd association_probabilities(z_measurements.size() + 1);

    for (size_t a_k = 0; a_k < z_measurements.size(); a_k++) {
      // IMM Filtering
      ////////////////
      // TODO: change to use pdaf step
      auto [x_est_upds, x_est_preds, z_est_preds] = ImmFilter::step(imm_model, sens_mod, dt, moment_based_preds, z_measurements.at(a_k));

      if (a_k == 0) {
        imm_estimations.push_back(x_est_preds);
      }
      imm_estimations.push_back(x_est_upds);

      // IMM mode probabilities
      /////////////////////////
      auto predicted_mode_probabilities = ImmFilter::update_probabilities(mixing_probs, z_est_preds, z_measurements.at(a_k), weights);

      // Calculate the hypothesis-conditional likelihood (7.61)
      /////////////////////////////////////////////////////////
      // TODO change to use PDAF::get_weights
      if (a_k == 0) {
        hypothesis_conditional_likelyhood.col(0) = config.clutter_intensity * (1 - config.prob_of_detection)
      }
      for (size_t s_k = 0; s_k < ImmModelT::N_MODELS; s_k++) {
        hypothesis_conditional_likelyhood(s_k, a_k + 1) = z_est_preds.at(s_k).pdf(z_measurements.at(a_k));
      }

      // Calculate thehypothesis likelihood (7.62)
      ////////////////////////////////////////////
      hypothesis_likelihood(a_k) = predicted_mode_probabilities.dot(hypothesis_conditional_likelyhood.row(a_k));

      // Calculate the accociation probabilities (7.56)
      ////////////////////////////////////////////////
      association_probabilities(a_k) = config.prob_of_detection * hypothesis_likelihood(a_k)
    }
    association_probabilities /= association_probabilities.sum();

    // Calculate the posterior mode probabilities (7.52)
    ////////////////////////////////////////////////////
    Eigen::Array<double, ImmModelT::N_MODELS, Eigen::Dynamic> posterior_mode_probabilities =
        hypothesis_conditional_likelyhood.colwise() * predicted_mode_probabilities;
    posterior_mode_probabilities /= posterior_mode_probabilities.colwise().sum().replicate(ImmModelT::N_MODELS, 1);

    // Calculate the mode-conditional association probabilities (7.57)
    //////////////////////////////////////////////////////////////////
    Eigen::Array<double, ImmModelT::N_MODELS, Eigen::Dynamic> mode_conditional_association_probabilities =
        association_probabilities * posterior_mode_probabilities;
    mode_conditional_association_probabilities /= mode_conditional_association_probabilities.colwise().sum().replicate(ImmModelT::N_MODELS, 1);

    // PDAF mixture reduction (7.58)
    ////////////////////////////////
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

    // Calculate the existence probability
    //////////////////////////////////////


    return {x_est_upds, mode_conditional_association_probabilities.row().sum(), }
  }

};

}  // namespace vortex::filter