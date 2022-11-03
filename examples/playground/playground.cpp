#include <experimental/mdspan>

#include <cstdint>
#include <tuple>
#include <iostream>
#include <type_traits>
#include <utility>

namespace stdex = std::experimental;

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple_impl(std::basic_ostream<Ch,Tr>& os,
                      const Tuple& t,
                      std::index_sequence<Is...>) {
  ((os << (Is == 0? "" : ", ") << std::get<Is>(t)), ...);
}

template <class Ch, class Tr, class... Args>
auto& operator<<(std::basic_ostream<Ch, Tr>& os, const std::tuple<Args...>& t) {
  os << "(";
  print_tuple_impl(os, t, std::index_sequence_for<Args...>{});
  return os << ")";
}

int main() {
  double array[] = {
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0,
    -1.0, -2.0, -3.0,
    -4.0, -5.0, -6.0,
    -7.0, -8.0, -9.0,
    -10.0, -11.0, -12.0
  };

  using tensor3d_view =
      stdex::mdspan<double, stdex::dextents<std::size_t, 3>, stdex::layout_right>;
  auto s = tensor3d_view(array, 2, 4, 3);
  std::cout
    << "rank = " << s.rank() << std::endl
    << "extent(0) = " << s.extent(0) << std::endl
    << "extent(1) = " << s.extent(1) << std::endl
    << "extent(2) = " << s.extent(2) << std::endl;
  for (size_t i = 0; i < s.extent(0); ++i) {
    for (size_t j = 0; j < s.extent(1); ++j) {
      for (size_t k = 0; k < s.extent(2); ++k) {
        std::cout << std::make_tuple(i, j, k) << std::endl;
      }
    }
  }

  return 0;
}