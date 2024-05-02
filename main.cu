#include <algorithm>
#include <chrono>
#include <fstream>
#include "utils.cpp"
#include "large_merge_path.cu"
#include "seq_merge_path.cpp"


double time_merge_cuda(int* A, int* B, int size, void (*kernel)(int*, int, int*, int, int*)) {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    int *M;
    int *A_gpu, *B_gpu, *M_gpu;

    int sizeA = size;
    int sizeB = size;
    int sizeMax = std::max(sizeA, sizeB);
    int sizeM = sizeA + sizeB;
    int blockSize = 1024;

    // Calculate the number of threads and blocks for the kernel
    int nThreads = std::min(sizeM, blockSize);
    int nBlocks = (sizeM + blockSize - 1) / blockSize;

    cudaMalloc(&A_gpu, sizeA * sizeof(int));
    cudaMalloc(&B_gpu, sizeB * sizeof(int));
    cudaMalloc(&M_gpu, sizeM * sizeof(int));

    // Copy data to device
    cudaMemcpy(A_gpu, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    // Create and start CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Kernel execution
    kernel<<<nBlocks, nThreads>>>(A_gpu, sizeA, B_gpu, sizeB, M_gpu);
    kernel<<<nBlocks, nThreads>>>(A_gpu, sizeA, B_gpu, sizeB, M_gpu);

    // Stop event and calculate the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(M_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

double time_merge_sequential(int* A, int* B, int size, void (*merge_func)(int*, int, int*, int, int*)) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    int sizeA = size;
    int sizeB = size;
    int sizeM = sizeA + sizeB;
    int* M = (int*)malloc(sizeM * sizeof(int));

    start = std::chrono::high_resolution_clock::now();

    merge_func(A, sizeA, B, sizeB, M);

    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = end - start;

    free(M);

    return 1000 * time.count(); // Convert seconds to milliseconds
}

double time_naive_large_merge(int*A, int *B, int size) {
    return time_merge_cuda(A, B, size, naive_merge_big_k);
}

double time_seq_merge(int*A, int *B, int size) {
    return time_merge_sequential(A, B, size, mergePath);
}

double time_optimal_large_merge(int *A, int *B, int size) {
    return time_merge_cuda(A, B, size, parallel_partition);
}


int main(void){
    const int min_power = 5;
    const int n_powers = 30;
    const int max_size = pow(2, n_powers - 1);
    const int max_iterations_per_size = 2048;
    const int max_iterations = pow(2, n_powers - 1);

    int *A = (int*)malloc(max_size * sizeof(int));
    int *B = (int*)malloc(max_size * sizeof(int));
    double time_taken;

    // Generate data for the largest needed size
    fastGenerateRandomSortedArray(A, 10, max_size);
    fastGenerateRandomSortedArray(B, 10, max_size);

    std::ofstream results_optimal("timing_optimal.csv");
    results_optimal << "Size,Iteration,Time(ms)\n";

    for (int i = min_power; i < n_powers; i++) {
        int size = pow(2, i);
        int num_iterations = std::min(max_iterations_per_size, max_iterations / size);

        for (int j = 0; j < num_iterations; j++) {
            time_taken = time_optimal_large_merge(A, B, size);
            results_optimal << size << "," << j << "," << time_taken << "\n";
        }
    }
    results_optimal.close();

    std::ofstream results_naive("timing_naive.csv");
    results_naive << "Size,Iteration,Time(ms)\n";

    for (int i = min_power; i < n_powers; i++) {
        int size = pow(2, i);
        int num_iterations = std::min(max_iterations_per_size, max_iterations / size);

        for (int j = 0; j < num_iterations; j++) {
            time_taken = time_naive_large_merge(A, B, size);
            results_naive << size << "," << j << "," << time_taken << "\n";
        }
    }
    results_naive.close();

    std::ofstream results_seq("timing_seq.csv");
    results_seq << "Size,Iteration,Time(ms)\n";

    for (int i = min_power; i < n_powers; i++) {
        int size = pow(2, i);
        int num_iterations = std::min(max_iterations_per_size, max_iterations / size);

        for (int j = 0; j < num_iterations; j++) {
            time_taken = time_seq_merge(A, B, size);
            results_seq << size << "," << j << "," << time_taken << "\n";
        }
    }
    results_seq.close();

    free(A);
    free(B);

    return 0;
}