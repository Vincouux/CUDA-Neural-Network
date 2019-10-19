#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

template <class T>
class Matrix {
public:
  /* Constructors */
  Matrix<T>(size_t height, size_t width);

  /* Getter */
  size_t getWidth() const;
  size_t getHeight() const;

  /* Operation */
  Matrix<T> add(const Matrix<T>& m) const;
  Matrix<T> dot(const Matrix<T>& m) const;
  Matrix<T> transpose() const;

private:
  size_t height;
  size_t width;
  std::vector<std::vector<T>> array;
};

#endif
