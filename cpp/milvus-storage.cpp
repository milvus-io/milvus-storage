#include "arrow/dataset/file_parquet.h"
#include "parquet/file_writer.h"
#include <algorithm>
#include <iostream>
#include <iterator>
void say_hello() { std::cout << "Hello, from milvus-storage!\n"; }

int main() {
  int a[5] = {0};
  std::find(std::begin(a), std::end(a), 1);
}