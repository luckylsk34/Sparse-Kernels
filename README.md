# Sparse-Kernels

Implementation of "Sparse GPU Kernels for Deep Learning" paper.

This code is based on [Google Sputnik](https://github.com/google-research/sputnik).

## Build

In the ```kernels``` folder, run the following commands to build and run the benchmark.

```mkdir build && cd build```

```cmake .. -DCMAKE_BUILD_TYPE=Release```

```make -j12```

Use can this [Colab link](https://colab.research.google.com/drive/1ws-o_njQ9Esi8p7EM1aFw8-dcMIFQXaw?usp=sharing) to Build and run the project.
