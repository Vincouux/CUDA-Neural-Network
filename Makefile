CXX = g++
NVCC = nvcc
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17 -g
TRASH = main

all: run

.PHONY: run
run:
	$(NVCC) $(NVFLAGS) -c src/Matrix/kernels.cu
	$(CXX) $(CXXFLAGS) -c -I/usr/local/cuda-5.5/include src/Activation/activation.cpp \
														src/Layer/dense.cpp \
														src/Layer/input.cpp \
														src/Model/model.cpp \
														src/main.cpp
	$(CXX) -o main activation.o dense.o input.o model.o main.o kernels.o -L/usr/local/cuda-5.5/lib64 -lcudart -lcurand -lcuda




.PHONY: clean
clean:
	$(RM) $(TRASH)
	rm *.o
