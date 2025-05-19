

#include <vector>
#include <random>
#include <type_traits>

namespace milvus_storage {
template <typename T>
class DataGen {
  public:
  DataGen() = default;
  ~DataGen() = default;

  std::vector<T> yield(size_t count) {
    std::vector<T> vec;
    vec.reserve(count);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Determine the type of distribution based on the type of T
    std::uniform_int_distribution<T> dist_int(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    std::uniform_real_distribution<T> dist_real(0.0, 1.0);

    // Use the appropriate distribution based on whether T is an integral or floating-point type
    if constexpr (std::is_integral<T>::value) {
      for (size_t i = 0; i < count; ++i) {
        vec.push_back(dist_int(gen));
      }
    } else if constexpr (std::is_floating_point<T>::value) {
      for (size_t i = 0; i < count; ++i) {
        vec.push_back(dist_real(gen));
      }
      // } else if constexpr (std::is_array<T>::value) {
      //     for (size_t i = 0; i < count; ++i) {
      //         vec.push_back(dist_real(gen));
      //     }
    } else {
      // If T is neither integral nor floating-point, throw an exception
      throw std::invalid_argument("Type T must be integral or floating-point");
    }

    return vec;
  }

  private:
};
}  // namespace milvus_storage