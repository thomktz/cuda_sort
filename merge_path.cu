#include <stdio.h>
#include "utils.cpp"

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void empty_k(void) {}

__global__ void merge_k(int *A, int sizeA, int *B, int sizeB, int *M, int length) {
  /*A, B, M : arrays, length : max(|A|, |B|)*/

  int i =  threadIdx.x;
  int Kx, Ky, Px, Py;

  if (i>length) {
    Kx=i-length;
    Ky=length;
    Px=length;
    Py=i-length;
  } else {
    Kx=0;
    Ky=i;
    Px=i;
    Py=0;
  }

  int offset, Qx, Qy;
  while (true){
    offset=abs(Ky-Py)/2;
    Qx=Kx+offset;
    Qy=Ky-offset;

    printf("%d\n\n", Qx);

    if (Qy >= 0 && Qx <= sizeB && (Qy == sizeA || Qx == 0 || A[Qy] > B[Qx - 1])) {
      if (Qx == sizeB || Qy == 0 || A[Qy - 1] <= B[Qx]) {
        if (Qy < sizeA && (Qx == sizeB || A[Qy] <= B[Qx])) {
          M[i]=A[Qy];
        } else {
          M[i]=B[Qx];
        }
        break;
      } else{
        Kx=Qx+1;
        Ky=Qy-1;
      }
    } else {
      Px=Qx-1;
      Py=Qy+1;
    }
  }

	

}

int main(void) {

  int *a, *b, *m;
  int *d_a, *d_b, *d_m; // Vecteurs sur le GPU
  int max_value = 100;
  int sizeA = 5;
  int sizeB = 7;
  int sizeMax = max(sizeA, sizeB);
  int sizeM = sizeA + sizeB;

  a = (int*)malloc(sizeA*sizeof(int));
  b = (int*)malloc(sizeB*sizeof(int));
	m = (int*)malloc(sizeM*sizeof(int));

  cudaMalloc(&d_a, sizeA*sizeof(int));
  cudaMalloc(&d_b, sizeB*sizeof(int));
  cudaMalloc(&d_m, sizeM*sizeof(int));

  generateRandomSortedArray(a, max_value, sizeA);
  generateRandomSortedArray(b, max_value, sizeB);

  cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

	merge_k <<<1, 1024>>> (d_a, sizeA, d_b, sizeB, d_m, sizeMax);

  // Copie du résultat du GPU vers le CPU
  cudaMemcpy(m, d_m, sizeM, cudaMemcpyDeviceToHost);

  // Libération de la mémoire sur le GPU et le CPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_m);
  free(a);
  free(b);
  free(m);

	return 0;
}
