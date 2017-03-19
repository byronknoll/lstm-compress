#include <iostream>
#include <ctime>
#include <fstream>
#include <valarray>
#include <math.h>

#include "lstm.h"

using namespace std;

inline float Rand() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Wrong number of arguments.\n");
    return 0;
  }
  srand(0xDEADBEEF);
  clock_t start = clock();
  std::ifstream is;
  is.open(argv[1], std::ios::binary);
  if (!is.is_open()) return -1;
  is.seekg(0, std::ios::end);
  unsigned long long len = is.tellg();
  is.seekg(0, std::ios::beg);

  Lstm lstm(0, 40, 3, 10, 0.1);
  valarray<float> probs = lstm.Perceive(is.get());
  unsigned long long percent = 1 + (len / 100);
  double entropy = log2(1.0/256);
  for (unsigned int pos = 0; pos < len - 1; ++pos) {
    int c = is.get();
    entropy += log2(probs[(unsigned char)c]);
    probs = lstm.Perceive(c);
    if (pos % percent == 0) {
      printf("\rprogress: %lld%%", pos / percent);
      fflush(stdout);
    }
  }
  entropy = -entropy / len;
  unsigned long long output_bytes = entropy * len / 8;
  printf("\r%lld bytes -> %lld bytes in %1.2f s.\n",
      len, output_bytes,
      ((double)clock() - start) / CLOCKS_PER_SEC);
  printf("cross entropy: %.4f\n", entropy);
  is.close();

  std::ofstream os(argv[2], std::ios::out | std::ios::binary);
  if (!os.is_open()) return -1;
  for (int i = 0; i < 20000; ++i) {
    float r = Rand();
    int c = 0;
    for (; c < 256; ++c) {
      r -= probs[c];
      if (r < 0) break;
    }
    probs = lstm.Predict(c);
    os.put(c);
  }
  os.close();
}
