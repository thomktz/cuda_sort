#include <algorithm>
#include <chrono>
#include "large_merge_path.cu"
#include "utils.cpp"
#include "seq_merge_path.cpp"

double time_optimal_large_merge(int size) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    
    // Main script to merge two sorted arrays
    int *A, *B, *M;
    int *A_gpu, *B_gpu, *M_gpu;
    int max_value = 20;
    int sizeA = size;
    int sizeB = size; // Different sizes
    int sizeMax = max(sizeA, sizeB);
    int sizeM = sizeA + sizeB;
    int blockSize = 1024;
    
    // Pick adequate number of blocks and threads
    int nThreads = std::min(sizeM, blockSize);
    int nBlocks =  (sizeM + blockSize - 1) / blockSize;

    A = (int*)malloc(sizeA * sizeof(int));
    B = (int*)malloc(sizeB * sizeof(int));
    M = (int*)malloc(sizeM * sizeof(int));

    cudaMalloc(&A_gpu, sizeA * sizeof(int));
    cudaMalloc(&B_gpu, sizeB * sizeof(int));
    cudaMalloc(&M_gpu, sizeM * sizeof(int));

    fastGenerateRandomSortedArray(A, max_value, sizeA);
    fastGenerateRandomSortedArray(B, max_value, sizeB);

    cudaMemcpy(A_gpu, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);


    start = std::chrono::high_resolution_clock::now();

    parallel_partition <<<nBlocks, nThreads>>> (A_gpu, sizeA, B_gpu, sizeB, M_gpu);

    end = std::chrono::high_resolution_clock::now();


    cudaMemcpy(M, M_gpu, sizeM * sizeof(int), cudaMemcpyDeviceToHost);

    // std::cout << "Array A: ";
    // printArray(A, sizeA);
    // std::cout << "Array B: ";
    // printArray(B, sizeB);

    // std::cout << "Merged array: ";
    // printArray(M, sizeM);
    
    // std::cout << isSorted(M, sizeM);

    std::chrono::duration<double> temps = end - start;

    // std::cout << "Temps d'exécution : " << temps.count() << " secondes\n";

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(M_gpu);
    free(A);
    free(B);
    free(M);

    return temps.count();
}

int main(void){
  int max = 20;
  double *tps = new double[max];

  for (int i = 0; i<max; i++){
    tps[i] = time_sequential_merge_path(pow(2,i));
  }
  printArray(tps, max);
  return 0;
}
