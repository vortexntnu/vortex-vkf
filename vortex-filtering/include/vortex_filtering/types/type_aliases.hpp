#pragma once

#include <Eigen/Dense>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

#define MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                         \
  using Mat_##t1##t2 = Eigen::Matrix<double, s1, s2>;                                                                                                          \
  using Mat_##t2##t1 = Eigen::Matrix<double, s2, s1>;
  
#define ONE_TYPE(t1, s1)                                                                                                                                 \
  using Vec_##t1      = Eigen::Vector<double, s1>;                                                                                                             \
  using Mat_##t1##t1  = Eigen::Matrix<double, s1, s1>;                                                                                                         \
  using Gauss_##t1    = vortex::prob::Gauss<s1>;                                                                                                               \
  using GaussMix_##t1 = vortex::prob::GaussMix<s1>;


#define TWO_TYPES(t1, s1, t2, s2)                                                                                                                        \
  ONE_TYPE(t1, s1)                                                                                                                                       \
  ONE_TYPE(t2, s2)                                                                                                                                       \
  MATRIX_TYPES(t1, s1, t2, s2)

#define THREE_TYPES(t1, s1, t2, s2, t3, s3)                                                                                                                 \
  ONE_TYPE(t1, s1)                                                                                                                                       \
  ONE_TYPE(t2, s2)                                                                                                                                       \
  ONE_TYPE(t3, s3)                                                                                                                                       \
  MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                               \
  MATRIX_TYPES(t1, s1, t3, s3)                                                                                                                               \
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

#define FIVE_TYPES(t1, s1, t2, s2, t3, s3, t4, s4, t5, s5)                                                                                                 \
  ONE_TYPE(t1, s1)                                                                                                                                       \
  ONE_TYPE(t2, s2)                                                                                                                                       \
  ONE_TYPE(t3, s3)                                                                                                                                       \
  ONE_TYPE(t4, s4)                                                                                                                                       \
  ONE_TYPE(t5, s5)                                                                                                                                       \
  MATRIX_TYPES(t1, s1, t2, s2)                                                                                                                               \
  MATRIX_TYPES(t1, s1, t3, s3)                                                                                                                               \
  MATRIX_TYPES(t1, s1, t4, s4)                                                                                                                               \
  MATRIX_TYPES(t1, s1, t5, s5)                                                                                                                               \
  MATRIX_TYPES(t2, s2, t3, s3)                                                                                                                               \
  MATRIX_TYPES(t2, s2, t4, s4)                                                                                                                               \
  MATRIX_TYPES(t2, s2, t5, s5)                                                                                                                               \
  MATRIX_TYPES(t3, s3, t4, s4)                                                                                                                               \
  MATRIX_TYPES(t3, s3, t5, s5)                                                                                                                               \
  MATRIX_TYPES(t4, s4, t5, s5)

namespace vortex {

template <size_t n_dim_x> struct Types_x {
  Types_x() = delete;
  ONE_TYPE(x, n_dim_x)
};

template <size_t n_dim_z> struct Types_z {
  Types_z() = delete;
  ONE_TYPE(z, n_dim_z)
};

template <size_t n_dim_u> struct Types_u {
  Types_u() = delete;
  ONE_TYPE(u, n_dim_u)
};

template <size_t n_dim_v> struct Types_v {
  Types_v() = delete;
  ONE_TYPE(v, n_dim_v)
};

template <size_t n_dim_w> struct Types_w {
  Types_w() = delete;
  ONE_TYPE(w, n_dim_w)
};

template <size_t n_dim_x, size_t n_dim_z> struct Types_xz {
  Types_xz() = delete;
  TWO_TYPES(x, n_dim_x, z, n_dim_z)
};

template <size_t n_dim_x, size_t n_dim_v> struct Types_xv {
  Types_xv() = delete;
  TWO_TYPES(x, n_dim_x, v, n_dim_v)
};

template <size_t n_dim_x, size_t n_dim_u> struct Types_xu {
  Types_xu() = delete;
  TWO_TYPES(x, n_dim_x, u, n_dim_u)
};

template <size_t n_dim_x, size_t n_dim_w> struct Types_xw {
  Types_xw() = delete;
  TWO_TYPES(x, n_dim_x, w, n_dim_w)
};

template <size_t n_dim_z, size_t n_dim_u> struct Types_zw {
  Types_zw() = delete;
  TWO_TYPES(z, n_dim_z, w, n_dim_u)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u> struct Types_xuv {
  Types_xuv() = delete;
  THREE_TYPES(x, n_dim_x, u, n_dim_u, v, n_dim_u)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u> struct Types_xzu {
  Types_xzu() = delete;
  THREE_TYPES(x, n_dim_x, z, n_dim_z, u, n_dim_u)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u, size_t n_dim_v> struct Types_xzuv {
  Types_xzuv() = delete;
  FOUR_TYPES(x, n_dim_x, z, n_dim_z, u, n_dim_u, v, n_dim_v)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u, size_t n_dim_v, size_t n_dim_w> struct Types_xzuvw {
  Types_xzuvw() = delete;
  FIVE_TYPES(x, n_dim_x, z, n_dim_z, u, n_dim_u, v, n_dim_v, w, n_dim_w)
};

} // namespace vortex


// Don't want you to use these macros outside of this file :)
#undef ONE_TYPE
#undef TWO_TYPES
#undef THREE_TYPES
#undef FOUR_TYPES
#undef FIVE_TYPES
#undef MATRIX_TYPES
