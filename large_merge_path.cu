#include <iostream>
#include <stdio.h>

void testCUDA(cudaError_t error, const char* file, int line) {
  // To catch errors

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void naive_merge_big_k(int *A, int sizeA, int *B, int sizeB, int *M) {
    // Find value for M[i], for thread i.

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i>=sizeA+sizeB) return;
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

__device__ void merge_big_k(int *A, int sizeA, int *B, int sizeB, int *M) {
    // Find value for M[i], for (block, thread) i.

    int i = threadIdx.x;
    int Kx, Ky, Py, offset, Qx, Qy;

    // Prevent entering the loop if i is out of bounds
    if (i >= (sizeA+sizeB)) { return; } 

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

__global__ void parallel_partition(int* A, int sizeA, int* B, int sizeB, int* M){
    int i = blockIdx.x;
    int A_start, A_top, A_bottom, B_start, B_top, M_start, a_i, b_i;
  
    A_start = sizeA;
    B_start = sizeB;

    M_start = i * 1024;

    if (M_start > sizeA) {
        B_top = M_start - sizeA;
        A_top = sizeA; 
    }
    else {
        B_top = 0;
        A_top = M_start;
    }

    A_bottom = B_top;

    while (true) {

        int offset = abs(A_top - A_bottom)/2;

        a_i = A_top - offset;
        b_i = B_top + offset;

         if (
            (a_i >= 0) && (b_i <= sizeB) && ((a_i == sizeA) || (b_i == 0) || (A[a_i] > B[b_i-1]))
        ) {
            if( 
                (b_i == sizeB) || a_i == 0 || A[a_i-1] <= B[b_i]
            ) {
                A_start = a_i;
                B_start = b_i;
                break;
            } else {
                A_top = a_i - 1;
                B_top = b_i + 1;
            }
        } else {
            A_bottom = a_i +1;
        }
    }

    __syncthreads();
    merge_big_k(
        &A[A_start],  // Sub-array A for block i
        sizeA - A_start,  // Size of sub-array A
        &B[B_start],  // Sub-array B for block i
        sizeB - B_start,  // Size of sub-array B
        &M[M_start]  // Sub-array M for block i
    );
    
}
