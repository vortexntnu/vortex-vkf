#pragma once
#include <array>
#include <iostream>
#include <tuple>

namespace detail {
template <typename... Ts, size_t... Is> std::ostream &println_tuple_impl(std::ostream &os, std::tuple<Ts...> tuple, std::index_sequence<Is...>)
{
  static_assert(sizeof...(Is) == sizeof...(Ts), "Indices must have same number of elements as tuple types!");
  static_assert(sizeof...(Ts) > 0, "Cannot insert empty tuple into stream.");
  size_t last = sizeof...(Ts) - 1; // assuming index sequence 0,...,N-1

  return ((os << std::get<Is>(tuple) << (Is != last ? "\r\n" : "")), ...);
}
} // namespace detail

template <typename... Ts> std::ostream &operator<<(std::ostream &os, const std::tuple<Ts...> &tuple)
{
  return detail::println_tuple_impl(os, tuple, std::index_sequence_for<Ts...>{});
}

template <typename T, std::size_t N> std::ostream &operator<<(std::ostream &os, const std::array<T, N> &arr)
{
  size_t last = N - 1;
  for (std::size_t i = 0; i < N; ++i) {
    os << arr[i] << (i != last ? "\r\n" : "");
  }
  return os;
}