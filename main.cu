#include <algorithm>
#include "large_merge_path.cu"
#include "utils.cpp"

int main(void) {
    // Main script to merge two sorted arrays

    int *A, *B, *M;
    int *A_gpu, *B_gpu, *M_gpu;
    int max_value = 2000;
    int sizeA = 100;
    int sizeB = 120; // Different sizes
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

    generateRandomSortedArray(A, max_value, sizeA);
    generateRandomSortedArray(B, max_value, sizeB);

    cudaMemcpy(A_gpu, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

    parallel_partition <<<nBlocks, nThreads>>> (A_gpu, sizeA, B_gpu, sizeB, M_gpu);

    cudaMemcpy(M, M_gpu, sizeM * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Array A: ";
    printArray(A, sizeA);
    std::cout << "Array B: ";
    printArray(B, sizeB);

    std::cout << "Merged array: ";
    printArray(M, sizeM);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(M_gpu);
    free(A);
    free(B);
    free(M);

    return 0;
}