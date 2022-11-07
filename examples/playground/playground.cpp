#include <experimental/mdspan>

#include <cstdint>
#include <tuple>
#include <iostream>
#include <type_traits>
#include <utility>

namespace stdex = std::experimental;

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple_impl(std::basic_ostream<Ch, Tr>& os,
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

// Inspired by https://github.com/kokkos/mdspan/issues/202#issuecomment-1296879082
template <typename IndexType, std::size_t... Extents, typename DestMDSpan, typename SrcMDSpan,
          std::size_t FirstExtentIndex, std::size_t... RestExtentIndexes,
          typename... ConstructedIndexPack>
void copy_impl(const stdex::extents<IndexType, Extents...>& ext, DestMDSpan dest, SrcMDSpan src,
               std::index_sequence<FirstExtentIndex, RestExtentIndexes...>,
               ConstructedIndexPack... element_index) {
  for (IndexType i = 0; i < ext.extent(FirstExtentIndex); i++)
    if constexpr(sizeof...(RestExtentIndexes) > 0) {
      copy_impl(ext, dest, src, std::index_sequence<RestExtentIndexes...>{},
                element_index..., i);
    } else {
      dest(element_index..., i) = src(element_index..., i);
    }
}

template <class IndexType, std::size_t... Extents, typename DestMDSpan, typename SrcMDSpan>
void copy_impl(stdex::extents<IndexType, Extents...> ext, DestMDSpan dest, SrcMDSpan src) {
  copy_impl(ext, dest, src, std::make_index_sequence<sizeof...(Extents)>{});
}

template <typename DestMDSpan, typename SrcMDSpan>
void copy(DestMDSpan dest, SrcMDSpan src) {
  static_assert(src.rank() == dest.rank());
  static_assert(
      std::is_same_v<typename SrcMDSpan::extents_type, typename DestMDSpan::extents_type>);
  copy_impl(src.extents(), dest, src);
}

int main() {
  double array[] = {  // 2x4x3
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    10.0, 11.0, 12.0,
    -1.0, -2.0, -3.0,
    -4.0, -5.0, -6.0,
    -7.0, -8.0, -9.0,
    -10.0, -11.0, -12.0
  };

  double array2[24];

  using tensor3d_c_layout =
      stdex::mdspan<double, stdex::dextents<std::size_t, 3>, stdex::layout_right>;
  using tensor3d_f_layout =
      stdex::mdspan<double, stdex::dextents<std::size_t, 3>, stdex::layout_left>;
  auto src = tensor3d_c_layout(array, 2, 4, 3);
  auto dest = tensor3d_f_layout(array2, 2, 4, 3);
  copy(dest, src);
  for (double e : array2) {
    std::cout << e << ", ";
  }
  std::cout << std::endl;
  for (std::size_t i = 0; i < dest.extent(0); ++i) {
    for (std::size_t j = 0; j < dest.extent(1); ++j) {
      for (std::size_t k = 0; k < dest.extent(2); ++k) {
        std::cout << dest(i, j, k) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}