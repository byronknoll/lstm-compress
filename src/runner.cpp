#include <iostream>
#include <fstream>
#include <valarray>
#include <math.h>

#include "lstm-compress.h"

using namespace std;

int main(int argc, char* argv[]) {
  std::ifstream is;
  is.open(argv[1], std::ios::binary);
  is.seekg(0, std::ios::end);
  unsigned long long len = is.tellg();
  is.seekg(0, std::ios::beg);
  char* buffer = new char[len];
  is.read(buffer, len);
  is.close();

  LstmCompress lstm;
  valarray<float> probs = lstm.Perceive(buffer[0]);
  double entropy = log2(1.0/256);
  for (unsigned int pos = 1; pos < len; ++pos) {
    entropy += log2(probs[(unsigned char)buffer[pos]]);
    probs = lstm.Perceive(buffer[pos]);
  }
  printf("cross entropy: %.4f\n", -entropy/len);
  delete[] buffer;
}
