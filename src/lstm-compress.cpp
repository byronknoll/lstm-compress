#include "lstm-compress.h"

LstmCompress::LstmCompress() : output_(256) {
  output_ = 1.0 / 256;
}

std::valarray<float>& LstmCompress::Perceive(unsigned char val) {
  return output_;
}

