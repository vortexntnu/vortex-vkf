#pragma once

#include <concepts>
#include <type_traits>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::concepts {

template <typename From, typename To>
concept mat_convertible_to =
    std::convertible_to<std::decay_t<From>, std::decay_t<To>> &&
    std::is_base_of_v<Eigen::MatrixBase<std::decay_t<From>>,
                      std::decay_t<From>> &&
    std::is_base_of_v<Eigen::MatrixBase<std::decay_t<To>>, std::decay_t<To>> &&
    (std::decay_t<From>::RowsAtCompileTime ==
     std::decay_t<To>::RowsAtCompileTime) &&
    (std::decay_t<From>::ColsAtCompileTime ==
     std::decay_t<To>::ColsAtCompileTime);

}  // namespace vortex::concepts
