#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>

class LstmCompress {
 public:
  LstmCompress(unsigned int num_cells, float learning_rate);
  LstmCompress(unsigned int num_cells, float learning_rate, unsigned int seed);
  std::valarray<float>& Perceive(unsigned char input);
  std::valarray<float>& Predict(unsigned char input);

 private:
  void ForwardPass(unsigned char input);
  void BackwardPass(unsigned char input);
  std::valarray<float> output_, state_, hidden_, input_, probs_;
  std::valarray<std::valarray<float>> forget_gate_, input_node_,
      input_gate_, output_gate_, output_layer_;
  float learning_rate_;
};

#endif

