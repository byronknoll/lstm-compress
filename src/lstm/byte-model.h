#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "lstm.h"

#include <valarray>

class ByteModel {
 public:
  ByteModel(unsigned int num_cells, unsigned int num_layers, int horizon,
      float learning_rate, float gradient_clip, const std::vector<bool>& vocab,
      unsigned int vocab_size);
  float Predict();
  void Perceive(int bit);

 protected:
  int top_, mid_, bot_;
  std::valarray<int> byte_map_;
  std::valarray<float> probs_;
  unsigned int bit_context_;
  Lstm lstm_;
  const std::vector<bool>& vocab_;
};

#endif
