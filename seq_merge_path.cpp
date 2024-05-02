#include <iostream>
// #include "utils.cpp"

void mergePath(int A[], int sizeA, int B[], int sizeB, int M[]) {
  // Sequential merge algorithm

  int i = 0, j = 0;

  while ((i + j) < (sizeA + sizeB)) {
    if (i >= sizeA) {
      M[i+j] = B[j];
      j++;
    } else if ((j >= sizeB) || (A[i] < B[j])) {
      M[i+j] = A[i];
      i++;
    } else {
      M[i+j] = B[j];
      j++;
    }
  }
}
