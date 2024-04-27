#include <iostream>
#include <stdio.h>
#include "utils.cpp"

void testCUDA(cudaError_t error, const char* file, int line) {
  // To catch errors

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void merge_k(int *A, int sizeA, int *B, int sizeB, int *M) {
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

int main(void) {
  // Main script to merge two sorted arrays

  int *A, *B, *M;
  int *A_gpu, *B_gpu, *M_gpu;
  int max_value = 200;
  int sizeA = 10;
  int sizeB = 12; // Different sizes
  int sizeMax = max(sizeA, sizeB);
  int sizeM = sizeA + sizeB;

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

	merge_k <<<1, sizeM>>> (A_gpu, sizeA, B_gpu, sizeB, M_gpu);


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
