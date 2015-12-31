#include <iostream>
#include <ctime>
#include <fstream>
#include <valarray>
#include <math.h>

#include "lstm-compress.h"

using namespace std;

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Wrong number of arguments.\n");
    return 0;
  }
  clock_t start = clock();
  std::ifstream is;
  is.open(argv[1], std::ios::binary);
  is.seekg(0, std::ios::end);
  unsigned long long len = is.tellg();
  is.seekg(0, std::ios::beg);
  char* buffer = new char[len];
  is.read(buffer, len);
  is.close();

  LstmCompress lstm(10, 0.1);
  valarray<float> probs = lstm.Perceive(buffer[0]);
  double entropy = log2(1.0/256);
  for (unsigned int pos = 1; pos < len; ++pos) {
    entropy += log2(probs[(unsigned char)buffer[pos]]);
    probs = lstm.Perceive(buffer[pos]);
  }
  entropy = -entropy / len;
  unsigned long long output_bytes = entropy * len / 8;
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      len, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
  printf("cross entropy: %.4f\n", entropy);
  delete[] buffer;
}
