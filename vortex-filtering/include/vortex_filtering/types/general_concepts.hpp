#pragma once

#include <concepts>
#include <type_traits>
#include <vortex_filtering/types/type_aliases.hpp>

namespace vortex::concepts {
    
template <typename From, typename To>
concept mat_convertible_to = requires {
  // Check if From is convertible to To.
  requires std::convertible_to<From, To>;
  // Check if both From and To are Eigen::Matrix types.
  requires std::is_base_of_v<Eigen::MatrixBase<From>, From> && std::is_base_of_v<Eigen::MatrixBase<To>, To>;

  // Compile-time check for fixed dimensions compatibility.
  requires(From::RowsAtCompileTime == To::RowsAtCompileTime) && (From::ColsAtCompileTime == To::ColsAtCompileTime);
};

} // namespace vortex::concepts