#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "lstm.h"

#include <valarray>
#include <memory>

class ByteModel {
 public:
  ByteModel(const std::vector<bool>& vocab, Lstm* lstm);
  float Predict();
  void Perceive(int bit);

 protected:
  int top_, mid_, bot_;
  std::valarray<int> byte_map_;
  std::valarray<float> probs_;
  unsigned int bit_context_;
  std::unique_ptr<Lstm> lstm_;
  const std::vector<bool>& vocab_;
};

#endif
