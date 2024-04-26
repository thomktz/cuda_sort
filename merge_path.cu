#include <stdio.h>
#include "utils.cpp"

int main() {
  int size = 10;
  int max_value = 100;

  int array[size];

  generateRandomSortedArray(array, max_value, size);

  printf("Sorted array: ");
  printArray(array, size);

  return 0;
}
