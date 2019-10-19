#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

template <class T>
class Matrix {
public:
  /* Constructors */
  Matrix<T>(size_t heights, size_t width);

  /* Getters */
  size_t getWidth() const;
  size_t getHeight() const;

  /* Operations */
  Matrix<T> add(const Matrix<T>& m) const;
  Matrix<T> dot(const Matrix<T>& m) const;
  Matrix<T> transpose() const;

private:
  int height;
  int width;
  std::vector<std::vector<T>> array;
};

#endif
