/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>

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

__global__ void merge_k(int *A, int *B, int *M, int length) {
  /*A, B, M : arrays, length : max(|A|, |B|)*/

  i =  threadIdx.x;
  if (i>length) {
    int Kx=i-length;
    int Ky=length;
    int Px=length;
    int Py=i-length;
  } else {
    int Kx=0;
    int Ky=i;
    int Px=i;
    int Py=0;
  }


  while (true){
    int offset=abs(Ky-Py)/2;
    int Qx=Kx+offset;
    int Qy=Ky-offset;
    
    if (Qy >= 0 && Qx <= B.size() && (Qy == A.size() || Qx == 0 || A[Qy] > B[Qx - 1])) {
      if (Qx == B.size() || Qy == 0 || A[Qy - 1] <= B[Qx]) {
        if (Qy < A.size() && (Qx == B.size() || A[Qy] <= B[Qx])) {
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

	empty_k <<<1, 1>>> ();

	/*************************************************************

	Once requested, replace this comment by the appropriate code

	*************************************************************/


	return 0;
}
