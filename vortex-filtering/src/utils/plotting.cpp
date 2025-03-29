#include <vortex_filtering/utils/ellipse.hpp>
#include <vortex_filtering/utils/plotting.hpp>

namespace vortex {
namespace plotting {

utils::Ellipse gauss_to_ellipse(const vortex::prob::Gauss2d &gauss, double scale_factor) { return utils::Ellipse(gauss, scale_factor); }

} // namespace plotting
} // namespace vortex
