#include <iostream>

void mergePath(int A[], int B[], int M[], int sizeA, int sizeB) {
  int i = 0, j = 0, k = 0;

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

int main() {
  int max_value = 100;
  int sizeA = 5;
  int sizeB = 7;

  // Allocate memory for arrays
  int A[sizeA];
  int B[sizeB];
  int M[sizeA + sizeB];

  // Generate random sorted arrays A and B
  generateRandomSortedArray(A, max_value, sizeA);
  generateRandomSortedArray(B, max_value, sizeB);

  // Print the original arrays
  std::cout << "Array A: ";
  printArray(A, sizeA);
  std::cout << "Array B: ";
  printArray(B, sizeB);

  // Merge the arrays
  mergePath(A, B, M, sizeA, sizeB);

  // Print the merged array
  std::cout << "Merged array: ";
  printArray(M, sizeA + sizeB);

  return 0;
}