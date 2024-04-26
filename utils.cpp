#include <cstdlib>

void insertionSort(int arr[], int n) {
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
  printf("[");
  // Assumes length is > 0
  for (int i = 0; i < size-1; i++) {
    printf("%d, ", array[i]);
  }
  printf("%d]", array[size-1]);
  printf("\n");
}

void generateRandomSortedArray(int *array, int max_value, int size) {
  srand(time(NULL));

  for (int i = 0; i < size; i++) {
    array[i] = rand() % (max_value + 1);
  }

  insertionSort(array, size);
}