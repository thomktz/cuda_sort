#include <iostream>
#include <stdio.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include "utils.cpp"

void testCUDA(cudaError_t error, const char* file, int line) {
  // To catch errors

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void merge_small_k(int *A, int sizeA, int *B, int sizeB, int *M) {
    // Find value for M[i], for thread i.

    int i = threadIdx.x;
    int Kx, Ky, Py, offset, Qx, Qy;

    // No need to define Px, since it's never used
    if (i>sizeA) {
        Kx = i-sizeA;
        Ky = sizeA;
        Py = i-sizeA;
    } else {
        Kx = 0;
        Ky = i;
        Py = 0;
    }

    while (true){
        offset = abs(Ky-Py)/2;
        Qx = Kx+offset;
        Qy = Ky-offset;

        if (
            (Qy >= 0) && (Qx <= sizeB) && ((Qy == sizeA) || (Qx == 0) || (A[Qy] > B[Qx - 1]))
        ) {
            if (
                (Qx == sizeB) || (Qy == 0) || (A[Qy - 1] <= B[Qx])
            ) {
                if (
                    (Qy < sizeA) && ((Qx == sizeB) || (A[Qy] <= B[Qx]))
                ) {
                    M[i] = A[Qy];
                } else {
                    M[i] = B[Qx];
                }
                break;
            } else {
                Kx = Qx+1;
                Ky = Qy-1;
            }
        } else {
            Py = Qy+1;
        }
    }
}

__global__ void merge_small_k_shared(int *A, int sizeA, int *B, int sizeB, int *M) {
    // Find value for M[i], for thread i.

    int i = threadIdx.x;
    int Kx, Ky, Py, offset, Qx, Qy;

    // Array shared between threads for efficiency
    __shared__ int shared_M[1024];

    // No need to define Px, since it's never used
    if (i > sizeA) {
        Kx = i-sizeA;
        Ky = sizeA;
        Py = i-sizeA;
    } else {
        Kx = 0;
        Ky = i;
        Py = 0;
    }

    while (true) {
        offset = abs(Ky-Py)/2;
        Qx = Kx+offset;
        Qy = Ky-offset;

        if (
            (Qy >= 0) && (Qx <= sizeB) && ((Qy == sizeA) || (Qx == 0) || (A[Qy] > B[Qx - 1]))
        ) {
            if (
                (Qx == sizeB) || (Qy == 0) || (A[Qy - 1] <= B[Qx])
            ) {
                if (
                    (Qy < sizeA) && ((Qx == sizeB) || (A[Qy] <= B[Qx]))
                ) {
                    shared_M[i] = A[Qy];
                } else {
                    shared_M[i] = B[Qx];
                }
                break;
            } else {
                Kx = Qx+1;
                Ky = Qy-1;
            }
        } else {
            Py = Qy+1;
        }
    }

    __syncthreads();
    M[i] = shared_M[i];
}


void benchmarkMerge(int N, int *A, int sizeA, int *B, int sizeB, int *M, int *A_gpu, int *B_gpu, int *M_gpu) {
    std::vector<double> times_global, times_shared;
    for (int i = 0; i < N; ++i) {
        // Timing for global memory kernel
        auto start = std::chrono::high_resolution_clock::now();
        merge_small_k<<<1, sizeA + sizeB>>>(A_gpu, sizeA, B_gpu, sizeB, M_gpu);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_global.push_back(elapsed.count());

        // Timing for shared memory kernel
        start = std::chrono::high_resolution_clock::now();
        merge_small_k_shared<<<1, sizeA + sizeB>>>(A_gpu, sizeA, B_gpu, sizeB, M_gpu);
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        times_shared.push_back(elapsed.count());
    }

    // Calculating mean and standard deviation
    double mean_global = std::accumulate(times_global.begin(), times_global.end(), 0.0) / N;
    double mean_shared = std::accumulate(times_shared.begin(), times_shared.end(), 0.0) / N;

    double std_dev_global = std::sqrt(std::inner_product(times_global.begin(), times_global.end(), times_global.begin(), 0.0) / N - mean_global * mean_global);
    double std_dev_shared = std::sqrt(std::inner_product(times_shared.begin(), times_shared.end(), times_shared.begin(), 0.0) / N - mean_shared * mean_shared);

    std::cout << "Global Memory Kernel: Mean = " << mean_global << " ms, StdDev = " << std_dev_global << " ms\n";
    std::cout << "Shared Memory Kernel: Mean = " << mean_shared << " ms, StdDev = " << std_dev_shared << " ms\n";
}
