#include <cstdlib>

void insertionSort(int arr[], int n) {
  // Insertion sort algorithm, to sort the arrays to be merged

  int i, key, j;
  for (i = 1; i < n; i++) {
    key = arr[i];
    j = i - 1;

    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = key;
  }
}

void printArray(int *array, int size) {
  // Utils function to print an array

  printf("[");
  // (Assumes length is > 0)
  for (int i = 0; i < size-1; i++) {
    printf("%d, ", array[i]);
  }
  printf("%d]", array[size-1]);
  printf("\n");
}

void generateRandomSortedArray(int *array, int max_value, int size) {
  // Utils function to generate a random sorted array

  for (int i = 0; i < size; i++) {
    array[i] = rand() % (max_value + 1);
  }
  insertionSort(array, size);
}