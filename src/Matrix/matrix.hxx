template <class T>
Matrix<T>::Matrix(size_t height, size_t width) {
  this->height = height;
  this->width = width;
  this->array = std::vector<std::vector<T>>(height, std::vector<T>(width));
}

template <class T>
size_t Matrix<T>::getHeight() const {
  return this->height;
}

template <class T>
size_t Matrix<T>::getWidth() const {
  return this->height;
}

template <class T>
Matrix<T> Matrix<T>::add(const Matrix& m) const{
  Matrix result(height, width);
  for (int i=0; i < height; i++){
    for (int j=0; j < width; j++){
      result.array[i][j] = array[i][j] + m.array[i][j];
    }
  }
  return result;
}

template <class T>
Matrix<T> Matrix<T>::dot(const Matrix& m) const {
  T val=0;
  Matrix<T> result(height, m.width);
  for (int i=0; i < height; i++){
    for (int j=0; j < m.width; j++){
      for (int h=0; h < width; h++){
        val += array[i][h] * m.array[h][j];
      }
      result.array[i][j] = val;
      val=0;
    }
  }
  return result;
}

template <class T>
Matrix<T> Matrix<T>::transpose() const {
  Matrix<T> result(width, height);
  for (int i=0; i < width; i++){
    for (int j=0; j < height; j++){
      result.array[i][j] = array[j][i];
    }
  }
  return result;
}
