#include <algorithm>
#include <chrono>
#include "large_merge_path.cu"
#include "utils.cpp"
#include "seq_merge_path.cpp"

double time_naive_large_merge(int size){
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

    naive_merge_big_k <<<nBlocks, nThreads>>> (A_gpu, sizeA, B_gpu, sizeB, M_gpu);
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();

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

    return 1000*temps.count();
}

int main(void){
  // std::cout << "hello";

  

  int max = 25;
  double *tps = new double[max];

  for (int i = 0; i<max; i++){
  //   // std::cout << i;
    tps[i] = time_optimal_large_merge(pow(2,i));
  }
  printArrayDouble(tps, max);
  
  return 0;
}
