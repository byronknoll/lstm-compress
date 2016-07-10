#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>
#include <memory>

#include "layer.h"

class LstmCompress {
 public:
  LstmCompress(unsigned int num_cells, unsigned int num_layers,
      float learning_rate);
  std::valarray<float>& Perceive(unsigned char input);
  std::valarray<float>& Predict(unsigned char input);

 private:
  std::vector<std::unique_ptr<Layer>> layers_;
  std::valarray<float> output_, probs_, hidden_, hidden_error_;
  std::valarray<std::valarray<float>> layer_input_, output_layer_;
  float learning_rate_;
  unsigned int num_cells_;
};

#endif

