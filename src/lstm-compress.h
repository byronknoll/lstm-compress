#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>

class LstmCompress {
 public:
  LstmCompress(unsigned int num_cells);
  std::valarray<float>& Perceive(unsigned char input);

 private:
  void ForwardPass(unsigned char input);
  std::valarray<float> output_, state_, hidden_, input_;
  std::valarray<std::valarray<float>> forget_gate_, input_gate_,
      candidate_gate_, output_gate_, output_layer_;
};

#endif

