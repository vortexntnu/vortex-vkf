#pragma once

#include <Eigen/Dense>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

using std::size_t;

#define MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                           \
  using Mat_##t1##t2 = Eigen::Matrix<double, s1, s2>;                                                                                                          \
  using Mat_##t2##t1 = Eigen::Matrix<double, s2, s1>;

#define ONE_TYPE(t1, s1)                                                                                                                                       \
  static constexpr size_t N_DIM_##t1 = s1;                                                                                                                     \
                                                                                                                                                               \
  using Vec_##t1      = Eigen::Vector<double, s1>;                                                                                                             \
  using Mat_##t1##t1  = Eigen::Matrix<double, s1, s1>;                                                                                                         \
  using Gauss_##t1    = vortex::prob::Gauss<s1>;                                                                                                               \
  using GaussMix_##t1 = vortex::prob::GaussMix<s1>;

#define TWO_TYPES(t1, s1, t2, s2)                                                                                                                              \
  ONE_TYPE(t1, s1)                                                                                                                                             \
  ONE_TYPE(t2, s2)                                                                                                                                             \
  MATRIX_TYPES(t1, s1, t2, s2)

#define THREE_TYPES(t1, s1, t2, s2, t3, s3)                                                                                                                    \
  ONE_TYPE(t1, s1)                                                                                                                                             \
  ONE_TYPE(t2, s2)                                                                                                                                             \
  ONE_TYPE(t3, s3)                                                                                                                                             \
  MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t3, s3)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t3, s3)

#define FOUR_TYPES(t1, s1, t2, s2, t3, s3, t4, s4)                                                                                                             \
  ONE_TYPE(t1, s1)                                                                                                                                             \
  ONE_TYPE(t2, s2)                                                                                                                                             \
  ONE_TYPE(t3, s3)                                                                                                                                             \
  ONE_TYPE(t4, s4)                                                                                                                                             \
  MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t3, s3)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t4, s4)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t3, s3)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t4, s4)                                                                                                                                 \
  MATRIX_TYPES(t3, s3, t4, s4)

#define FIVE_TYPES(t1, s1, t2, s2, t3, s3, t4, s4, t5, s5)                                                                                                     \
  ONE_TYPE(t1, s1)                                                                                                                                             \
  ONE_TYPE(t2, s2)                                                                                                                                             \
  ONE_TYPE(t3, s3)                                                                                                                                             \
  ONE_TYPE(t4, s4)                                                                                                                                             \
  ONE_TYPE(t5, s5)                                                                                                                                             \
  MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t3, s3)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t4, s4)                                                                                                                                 \
  MATRIX_TYPES(t1, s1, t5, s5)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t3, s3)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t4, s4)                                                                                                                                 \
  MATRIX_TYPES(t2, s2, t5, s5)                                                                                                                                 \
  MATRIX_TYPES(t3, s3, t4, s4)                                                                                                                                 \
  MATRIX_TYPES(t3, s3, t5, s5)                                                                                                                                 \
  MATRIX_TYPES(t4, s4, t5, s5)

#define ONE_TYPE_STRUCT(t1)                                                                                                                                    \
  template <size_t n_dim_##t1> struct Types_##t1 {                                                                                                             \
    Types_##t1() = delete;                                                                                                                                     \
    ONE_TYPE(t1, n_dim_##t1)                                                                                                                                   \
  };

#define TWO_TYPES_STRUCT(t1, t2)                                                                                                                               \
  template <size_t n_dim_##t1, size_t n_dim_##t2> struct Types_##t1##t2 {                                                                                      \
    Types_##t1##t2() = delete;                                                                                                                                 \
    TWO_TYPES(t1, n_dim_##t1, t2, n_dim_##t2)                                                                                                                  \
  };

#define THREE_TYPES_STRUCT(t1, t2, t3)                                                                                                                         \
  template <size_t n_dim_##t1, size_t n_dim_##t2, size_t n_dim_##t3> struct Types_##t1##t2##t3 {                                                               \
    Types_##t1##t2##t3() = delete;                                                                                                                             \
    THREE_TYPES(t1, n_dim_##t1, t2, n_dim_##t2, t3, n_dim_##t3)                                                                                                \
  };

#define FOUR_TYPES_STRUCT(t1, t2, t3, t4)                                                                                                                      \
  template <size_t n_dim_##t1, size_t n_dim_##t2, size_t n_dim_##t3, size_t n_dim_##t4> struct Types_##t1##t2##t3##t4 {                                        \
    Types_##t1##t2##t3##t4() = delete;                                                                                                                         \
    FOUR_TYPES(t1, n_dim_##t1, t2, n_dim_##t2, t3, n_dim_##t3, t4, n_dim_##t4)                                                                                 \
  };

#define FIVE_TYPES_STRUCT(t1, t2, t3, t4, t5)                                                                                                                  \
  template <size_t n_dim_##t1, size_t n_dim_##t2, size_t n_dim_##t3, size_t n_dim_##t4, size_t n_dim_##t5> struct Types_##t1##t2##t3##t4##t5 {                 \
    Types_##t1##t2##t3##t4##t5() = delete;                                                                                                                     \
    FIVE_TYPES(t1, n_dim_##t1, t2, n_dim_##t2, t3, n_dim_##t3, t4, n_dim_##t4, t5, n_dim_##t5)                                                                 \
  };

namespace vortex {
// Hover over the types in vscode to see the expanded types

ONE_TYPE_STRUCT(x)
ONE_TYPE_STRUCT(z)
ONE_TYPE_STRUCT(u)
ONE_TYPE_STRUCT(v)
ONE_TYPE_STRUCT(w)
ONE_TYPE_STRUCT(n)

TWO_TYPES_STRUCT(x, z) // Sensor model without noise
TWO_TYPES_STRUCT(x, v) // Dynamic model without input
TWO_TYPES_STRUCT(x, u) // Dynamic model without noise
TWO_TYPES_STRUCT(x, w)

TWO_TYPES_STRUCT(z, w)

THREE_TYPES_STRUCT(x, u, v) // Dynamic model
THREE_TYPES_STRUCT(x, z, w) // Sensor model
THREE_TYPES_STRUCT(x, z, n) // For IMM Filter

FOUR_TYPES_STRUCT(x, z, w, a) // Sensor model and augmented state a

FIVE_TYPES_STRUCT(x, z, u, v, w) // Dynamic model and sensor model

} // namespace vortex

// Don't want you to use these macros outside of this file :)
#undef ONE_TYPE
#undef TWO_TYPES
#undef THREE_TYPES
#undef FOUR_TYPES
#undef FIVE_TYPES
#undef ONE_TYPE_STRUCT
#undef TWO_TYPES_STRUCT
#undef THREE_TYPES_STRUCT
#undef FOUR_TYPES_STRUCT
#undef FIVE_TYPES_STRUCT
#undef MATRIX_TYPES
