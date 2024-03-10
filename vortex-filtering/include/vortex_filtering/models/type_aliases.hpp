#pragma once

#include <Eigen/Dense>
#include <vortex_filtering/probability/gaussian_mixture.hpp>
#include <vortex_filtering/probability/multi_var_gauss.hpp>

#define VORTEX_TYPES_1(t1, s1)                                                                                                                                 \
  using Vec_##t1      = Eigen::Vector<double, s1>;                                                                                                             \
  using Mat_##t1##t1  = Eigen::Matrix<double, s1, s1>;                                                                                                         \
  using Gauss_##t1    = vortex::prob::Gauss<s1>;                                                                                                               \
  using GaussMix_##t1 = vortex::prob::GaussMix<s1>;

#define VORTEX_TYPES_2(t1, s1, t2, s2)                                                                                                                         \
  using Mat_##t1##t2 = Eigen::Matrix<double, s1, s2>;                                                                                                          \
  using Mat_##t2##t1 = Eigen::Matrix<double, s2, s1>;

#define VORTEX_TYPES_3(t1, s1, t2, s2, t3, s3)                                                                                                                 \
  VORTEX_TYPES_1(t1, s1)                                                                                                                                       \
  VORTEX_TYPES_1(t2, s2)                                                                                                                                       \
  VORTEX_TYPES_1(t3, s3)                                                                                                                                       \
  VORTEX_TYPES_2(t1, s1, t2, s2)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t3, s3)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t3, s3)

#define VORTEX_TYPES_4(t1, s1, t2, s2, t3, s3, t4, s4)                                                                                                         \
  VORTEX_TYPES_1(t1, s1)                                                                                                                                       \
  VORTEX_TYPES_1(t2, s2)                                                                                                                                       \
  VORTEX_TYPES_1(t3, s3)                                                                                                                                       \
  VORTEX_TYPES_1(t4, s4)                                                                                                                                       \
  VORTEX_TYPES_2(t1, s1, t2, s2)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t3, s3)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t4, s4)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t3, s3)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t4, s4)                                                                                                                               \
  VORTEX_TYPES_2(t3, s3, t4, s4)

#define VORTEX_TYPES_5(t1, s1, t2, s2, t3, s3, t4, s4, t5, s5)                                                                                                 \
  VORTEX_TYPES_1(t1, s1)                                                                                                                                       \
  VORTEX_TYPES_1(t2, s2)                                                                                                                                       \
  VORTEX_TYPES_1(t3, s3)                                                                                                                                       \
  VORTEX_TYPES_1(t4, s4)                                                                                                                                       \
  VORTEX_TYPES_1(t5, s5)                                                                                                                                       \
  VORTEX_TYPES_2(t1, s1, t2, s2)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t3, s3)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t4, s4)                                                                                                                               \
  VORTEX_TYPES_2(t1, s1, t5, s5)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t3, s3)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t4, s4)                                                                                                                               \
  VORTEX_TYPES_2(t2, s2, t5, s5)                                                                                                                               \
  VORTEX_TYPES_2(t3, s3, t4, s4)                                                                                                                               \
  VORTEX_TYPES_2(t3, s3, t5, s5)                                                                                                                               \
  VORTEX_TYPES_2(t4, s4, t5, s5)

namespace vortex {

template <size_t n_dim_x> struct Types_x {
  Types_x() = delete;
  VORTEX_TYPES_1(x, n_dim_x)
};

template <size_t n_dim_z> struct Types_z {
  Types_z() = delete;
  VORTEX_TYPES_1(z, n_dim_z)
};

template <size_t n_dim_x, size_t n_dim_z> struct Types_xz {
  Types_xz() = delete;
  VORTEX_TYPES_1(x, n_dim_x)
  VORTEX_TYPES_1(z, n_dim_z)
  VORTEX_TYPES_2(x, n_dim_x, z, n_dim_z)
};

template <size_t n_dim_x, size_t n_dim_v> struct Types_xv {
  Types_xv() = delete;
  VORTEX_TYPES_1(x, n_dim_x)
  VORTEX_TYPES_1(v, n_dim_v)
  VORTEX_TYPES_2(x, n_dim_x, v, n_dim_v)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u> struct Types_xzu {
  Types_xzu() = delete;
  VORTEX_TYPES_3(x, n_dim_x, z, n_dim_z, u, n_dim_u)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u, size_t n_dim_v> struct Types_xzuv {
  Types_xzuv() = delete;
  VORTEX_TYPES_4(x, n_dim_x, z, n_dim_z, u, n_dim_u, v, n_dim_v)
};

template <size_t n_dim_x, size_t n_dim_z, size_t n_dim_u, size_t n_dim_v, size_t n_dim_w> struct Types_xzuvw {
  Types_xzuvw() = delete;
  VORTEX_TYPES_5(x, n_dim_x, z, n_dim_z, u, n_dim_u, v, n_dim_v, w, n_dim_w)
};

} // namespace vortex

#undef VORTEX_TYPES_1
#undef VORTEX_TYPES_2
#undef VOXTEX_TYPES_3
#undef VORTEX_TYPES_4
#undef VORTEX_TYPES_5
