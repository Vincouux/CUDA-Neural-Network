#include <iostream>

#include "Matrix/matrix.hpp"

int main() {
  Matrix<int> m(4, 5);
  std::cout << m.getWidth() << ", " << m.getHeight();
  return 0;
}
