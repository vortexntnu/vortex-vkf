#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <vortex_filtering/filters/ekf.hpp>
#include <vortex_filtering/filters/ukf.hpp>
#include <vortex_filtering/models/dynamic_model_interfaces.hpp>
#include <vortex_filtering/models/imm_model.hpp>
#include <vortex_filtering/models/sensor_model_interfaces.hpp>
#include <vortex_filtering/types/model_concepts.hpp>

namespace vortex::filter {

template <concepts::SensorModelWithDefinedSizes SensModT, models::concepts::ImmModel ImmModelT> class IMMIPDA {
public:


};
    
}  // namespace vortex::filter