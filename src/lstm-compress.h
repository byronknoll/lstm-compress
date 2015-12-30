#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>

class LstmCompress {
 public:
  LstmCompress();
  std::valarray<float>& Perceive(unsigned char val);

 private:
  std::valarray<float> output_;
};

#endif

